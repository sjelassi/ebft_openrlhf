"""
Strided Actor Model - Simplified generation with strided block computation.

Maintains the same forward pass structure as ed_actor.py but without Ray infrastructure.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

from openrlhf.models.utils import build_strided_attention_mask_and_positions


@dataclass
class GenerationOutput:
    """Output from strided generation."""
    prompts: torch.Tensor  # (batch_size, prompt_length)
    full_sequences: torch.Tensor  # (batch_size * n_samples, full_length)
    doc_ids: torch.Tensor  # Document IDs for masking
    qa_masks: torch.Tensor  # QA masks


class StridedActorModel:
    """
    Simplified actor model for evaluation with strided block computation.

    Uses the same forward pass structure as ed_actor.py:
    - Strided attention masks
    - Block-based generation
    - Custom position IDs
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_name_or_path: Path to model checkpoint or HuggingFace model name
            device: Device to load model on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.dtype = torch_dtype

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: torch.Tensor,
        doc_ids: torch.Tensor,
        qa_masks: torch.Tensor,
        n_samples_per_prompt: int = 4,
        generate_max_len: int = 128,
        context_length: int = 16,
        stride: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        document_masking: bool = False,
        qa_masking: bool = False,
        group_by_prompt: bool = False,
    ) -> GenerationOutput:
        """
        Generate multiple samples per prompt using strided block generation.

        This matches the generate_strided_blocks method from ed_actor.py:
        - Generates one token at a time for each block
        - Uses custom strided attention masks
        - Samples from temperature-scaled distributions

        Args:
            prompts: Tokenized prompts (batch_size, prompt_length)
            doc_ids: Document IDs for masking
            qa_masks: QA masks
            n_samples_per_prompt: Number of samples to generate per prompt
            generate_max_len: Maximum number of tokens to generate
            context_length: Context window length for each block
            stride: Stride between blocks
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            document_masking: Whether to use document masking
            qa_masking: Whether to use QA masking
            group_by_prompt: If True, run generation in prompt-local groups of
                size n_samples_per_prompt to mirror trainer batching behavior.

        Returns:
            GenerationOutput containing prompts and generated sequences
        """
        _, prompt_length = prompts.shape

        # Expand for multiple samples per prompt
        prompts_expanded = prompts.repeat_interleave(n_samples_per_prompt, dim=0).to(self.device)
        doc_ids_expanded = doc_ids.repeat_interleave(n_samples_per_prompt, dim=0).to(self.device)
        qa_masks_expanded = qa_masks.repeat_interleave(n_samples_per_prompt, dim=0).to(self.device)

        # Calculate number of blocks
        assert (prompt_length - generate_max_len - context_length) % stride == 0, (
            f"prompt_length {prompt_length - generate_max_len - context_length} must be divisible by stride {stride}"
        )
        num_blocks = (prompt_length - generate_max_len - context_length) // stride + 1

        def _generate_for_batch(batch_prompts: torch.Tensor, batch_doc_ids: torch.Tensor) -> torch.Tensor:
            full_sequence = batch_prompts.clone()
            batch_size = full_sequence.size(0)

            # Generate tokens iteratively - at each step, generate one token per block
            for generation_step in range(generate_max_len):
                # Build strided attention mask for current sequence
                attention_mask, position_ids = build_strided_attention_mask_and_positions(
                    full_sequence_length=full_sequence.shape[1],
                    prompt_length=prompt_length,
                    context_length=context_length,
                    generation_step=generation_step,
                    max_generation_length=generate_max_len,
                    stride=stride,
                    num_blocks=num_blocks,
                    device=self.device,
                    doc_ids=batch_doc_ids[:, :prompt_length],
                    document_masking=document_masking,
                    dtype=self.dtype,
                )

                outputs = self.model(
                    input_ids=full_sequence,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                # Match trainer actor precision path: sample from float32 logits.
                all_logits = outputs.logits.to(torch.float32)  # [B, sequence_length, vocab_size]

                # Calculate which positions to extract logits from (one per block)
                logit_positions = []
                for block_idx in range(num_blocks):
                    if generation_step == 0:
                        prediction_position = block_idx * stride + context_length - 1
                    else:
                        prediction_position = prompt_length + (generation_step - 1) * num_blocks + block_idx
                    logit_positions.append(prediction_position)

                position_indices = torch.tensor(logit_positions, dtype=torch.long, device=self.device)
                block_logits = all_logits.index_select(1, position_indices)  # [B, num_blocks, V]

                if top_p < 1.0:
                    if temperature > 0:
                        block_logits = block_logits / temperature
                        probabilities = torch.softmax(block_logits, dim=-1)  # [B, num_blocks, V]
                    else:
                        probabilities = torch.softmax(block_logits, dim=-1)

                    _, num_blocks_local, vocab_size = probabilities.shape
                    probs_flat = probabilities.view(-1, vocab_size)

                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs_flat, descending=True, dim=-1)

                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Find where cumulative probability exceeds top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = torch.zeros_like(probs_flat, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    probs_flat[indices_to_remove] = 0
                    probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)

                    sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                    sampled_tokens = sampled_tokens_flat.view(batch_size, -1)
                else:
                    if temperature > 0:
                        block_logits = block_logits / temperature
                        probabilities = torch.softmax(block_logits, dim=-1)
                        probs_flat = probabilities.view(batch_size * num_blocks, -1)
                        sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                        sampled_tokens = sampled_tokens_flat.view(batch_size, -1)
                    else:
                        # Greedy decoding (temperature=0): take argmax
                        sampled_tokens = torch.argmax(block_logits, dim=-1)

                full_sequence = torch.cat([full_sequence, sampled_tokens], dim=1)

            return full_sequence

        if group_by_prompt:
            total_expanded = prompts_expanded.size(0)
            if total_expanded % n_samples_per_prompt != 0:
                raise ValueError(
                    f"Expanded batch {total_expanded} must be divisible by n_samples_per_prompt={n_samples_per_prompt}"
                )
            grouped_outputs = []
            for start in range(0, total_expanded, n_samples_per_prompt):
                end = start + n_samples_per_prompt
                grouped_outputs.append(
                    _generate_for_batch(
                        prompts_expanded[start:end],
                        doc_ids_expanded[start:end],
                    )
                )
            full_sequence = torch.cat(grouped_outputs, dim=0)
        else:
            full_sequence = _generate_for_batch(prompts_expanded, doc_ids_expanded)

        # Match trainer behavior: generated region masks/doc ids are built via a strided unfold
        # over the original prompt-side metadata (not all-ones).
        strided_qa_masks = (
            qa_masks_expanded[:, context_length:]
            .unfold(1, generate_max_len, stride)  # (B, num_blocks, G)
            .transpose(1, 2)  # (B, G, num_blocks)
            .reshape(qa_masks_expanded.shape[0], -1)  # (B, G * num_blocks)
        )
        qa_masks_full = torch.cat([qa_masks_expanded, strided_qa_masks], dim=1)

        strided_doc_ids = (
            doc_ids_expanded[:, context_length:]
            .unfold(1, generate_max_len, stride)  # (B, num_blocks, G)
            .transpose(1, 2)  # (B, G, num_blocks)
            .reshape(doc_ids_expanded.shape[0], -1)  # (B, G * num_blocks)
        )
        doc_ids_full = torch.cat([doc_ids_expanded, strided_doc_ids], dim=1)

        if qa_masks_full.size(1) != full_sequence.size(1):
            raise ValueError(
                f"qa_masks length {qa_masks_full.size(1)} must match full_sequence length {full_sequence.size(1)}"
            )
        if doc_ids_full.size(1) != full_sequence.size(1):
            raise ValueError(
                f"doc_ids length {doc_ids_full.size(1)} must match full_sequence length {full_sequence.size(1)}"
            )

        return GenerationOutput(
            prompts=prompts,
            full_sequences=full_sequence.cpu(),
            doc_ids=doc_ids_full.cpu(),
            qa_masks=qa_masks_full.cpu(),
        )

    @torch.no_grad()
    def compute_logprobs_strided(
        self,
        sequences: torch.Tensor,
        prompt_length: int,
        context_length: int,
        generate_max_len: int,
        stride: int,
        num_blocks: int,
        temperature: float = 1.0,
        doc_ids: Optional[torch.Tensor] = None,
        document_masking: bool = False,
        qa_masks: Optional[torch.Tensor] = None,
        qa_masking: bool = False,
    ) -> torch.Tensor:
        """
        Compute log probabilities using strided attention masks.

        This maintains the same forward pass structure as ed_actor.py.

        Args:
            sequences: Full sequences (batch_size, full_length)
            prompt_length: Length of prompt
            context_length: Context window length
            generate_max_len: Generation length
            stride: Stride for blocks
            num_blocks: Number of blocks
            temperature: Temperature for scaling logits (default: 1.0)
            doc_ids: Document IDs for masking
            document_masking: Whether to use document masking
            qa_masks: QA masks
            qa_masking: Whether to use QA masking

        Returns:
            Log probabilities for generated tokens
        """
        # Move sequences to device
        sequences = sequences.to(self.device)
        if doc_ids is not None:
            doc_ids = doc_ids.to(self.device)

        # Build strided attention mask
        attention_mask, pos_ids = build_strided_attention_mask_and_positions(
            full_sequence_length=sequences.size(1),
            prompt_length=prompt_length,
            context_length=context_length,
            generation_step=generate_max_len,
            max_generation_length=generate_max_len,
            stride=stride,
            num_blocks=num_blocks,
            device=self.device,
            doc_ids=doc_ids[:, :prompt_length] if doc_ids is not None else None,
            document_masking=document_masking,
            dtype=self.dtype,
        )

        # Forward pass through model
        outputs = self.model(
            input_ids=sequences,
            attention_mask=attention_mask,
            position_ids=pos_ids,
        )

        # Match trainer actor precision path used for log-prob computation.
        logits = outputs.logits.to(torch.float32)

        # Get log probs for generated tokens only
        gen_logits = logits[:, prompt_length-1:-1, :]
        gen_tokens = sequences[:, prompt_length:]

        # Apply temperature scaling to match generation
        if temperature != 1.0:
            gen_logits = gen_logits / temperature

        # Compute log probabilities
        log_probs = F.log_softmax(gen_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=gen_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    @torch.no_grad()
    def compute_perplexity(
        self,
        sequences: torch.Tensor,
        prompt_length: int,
        context_length: int,
        generate_max_len: int,
        stride: int,
        num_blocks: int,
        temperature: float = 1.0,
        doc_ids: Optional[torch.Tensor] = None,
        document_masking: bool = False,
        qa_masks: Optional[torch.Tensor] = None,
        qa_masking: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute cross-entropy loss and perplexity with strided attention.

        Args:
            sequences: Full sequences
            prompt_length: Length of prompt
            context_length: Context window length
            generate_max_len: Generation length
            stride: Stride for blocks
            num_blocks: Number of blocks
            temperature: Temperature for scaling logits (default: 1.0)
            doc_ids: Document IDs
            document_masking: Use document masking
            qa_masks: QA masks
            qa_masking: Use QA masking

        Returns:
            (ce_loss, perplexity)
        """
        log_probs = self.compute_logprobs_strided(
            sequences=sequences,
            prompt_length=prompt_length,
            context_length=context_length,
            generate_max_len=generate_max_len,
            stride=stride,
            num_blocks=num_blocks,
            temperature=temperature,
            doc_ids=doc_ids,
            document_masking=document_masking,
            qa_masks=qa_masks,
            qa_masking=qa_masking,
        )

        # Compute average negative log likelihood
        ce_loss = -log_probs.mean().item()
        perplexity = torch.exp(-log_probs.mean()).item()

        return ce_loss, perplexity
