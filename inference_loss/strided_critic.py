"""
Strided Critic Model - Simplified embedding extraction with strided block computation.

Maintains the same forward pass structure as ed_critic.py but without Ray infrastructure.
"""

import math
import torch
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from openrlhf.models.utils import build_strided_attention_mask_and_positions


class StridedCriticModel:
    """
    Simplified critic model for evaluation with strided block computation.

    Uses the same forward pass structure as ed_critic.py:
    - Strided attention masks
    - Block-based hidden state extraction
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

        # Load model using AutoModel
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @torch.no_grad()
    def extract_gt_and_gen_embeddings(
        self,
        sequences: torch.Tensor,
        prompt_length: int,
        context_length: int,
        generate_max_len: int,
        stride: int,
        num_blocks: int,
        hidden_state_method: str = "concat",
        doc_ids: Optional[torch.Tensor] = None,
        document_masking: bool = False,
        qa_masks: Optional[torch.Tensor] = None,
        qa_masking: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ground truth and generated embeddings with gen_len dimension intact.

        This matches ebft_trainer.py flow where embeddings are returned WITHOUT pooling.
        The embed_method pooling and whitening are deferred to evaluation_metrics.py.

        Args:
            sequences: Full sequences (batch_size, full_length)
            prompt_length: Length of prompt
            context_length: Context window length
            generate_max_len: Generation length
            stride: Stride for blocks
            num_blocks: Number of blocks
            hidden_state_method: Method to extract hidden states ("concat", "last_layer")
            doc_ids: Document IDs
            document_masking: Use document masking
            qa_masks: QA masks
            qa_masking: Use QA masking

        Returns:
            (gen_embedding, gt_embedding) with shape (batch, num_blocks, gen_len, num_feat, hidden)
        """
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

        # Forward pass through model to get hidden states
        outputs = self.model(
            input_ids=sequences.to(self.device),
            attention_mask=attention_mask.to(self.device),
            position_ids=pos_ids.to(self.device),
            output_hidden_states=True,
        )

        # Extract hidden states based on method (matching openrlhf.models.critic.Critic.forward)
        if hidden_state_method in {"last_layer", "last_only"}:
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "mean":
            hidden_states = torch.mean(torch.stack(outputs.hidden_states[1:]), dim=0)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle":
            num_layers = len(outputs.hidden_states) - 1
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.mean(torch.stack([outputs.hidden_states[mid1], outputs.hidden_states[mid2]]), dim=0)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle_concat":
            num_layers = len(outputs.hidden_states) - 1
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.cat([outputs.hidden_states[mid1], outputs.hidden_states[mid2]], dim=-1)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle_stack":
            num_layers = len(outputs.hidden_states) - 1
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.stack([outputs.hidden_states[mid1], outputs.hidden_states[mid2]], dim=-2)
        elif hidden_state_method == "concat":
            num_layers = len(outputs.hidden_states) - 1
            idxs = [
                max(1, min(num_layers, math.floor(num_layers * 0.25))),
                max(1, min(num_layers, math.floor(num_layers * 0.50))),
                max(1, min(num_layers, math.floor(num_layers * 0.75))),
            ]
            selected = [outputs.hidden_states[i] for i in idxs]
            hidden_states = torch.cat(selected, dim=-1)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "stack":
            num_layers = len(outputs.hidden_states) - 1
            idxs = [
                max(1, min(num_layers, math.floor(num_layers * 0.25))),
                max(1, min(num_layers, math.floor(num_layers * 0.50))),
                max(1, min(num_layers, math.floor(num_layers * 0.75))),
            ]
            selected = [outputs.hidden_states[i] for i in idxs]
            hidden_states = torch.stack(selected, dim=-2)
        elif hidden_state_method.startswith("layer_"):
            try:
                layer_idx = int(hidden_state_method.split("_")[1])
                hidden_states = outputs.hidden_states[layer_idx]
                hidden_states = hidden_states.unsqueeze(-2)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {exc}") from exc
        elif hidden_state_method.startswith("concat_layers_"):
            try:
                layer_indices = [int(x) for x in hidden_state_method.replace("concat_layers_", "").split("_")]
                selected_layers = [outputs.hidden_states[idx] for idx in layer_indices]
                hidden_states = torch.cat(selected_layers, dim=-1)
                hidden_states = hidden_states.unsqueeze(-2)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {exc}") from exc
        elif hidden_state_method.startswith("stack_layers_"):
            try:
                layer_indices = [int(x) for x in hidden_state_method.replace("stack_layers_", "").split("_")]
                selected_layers = [outputs.hidden_states[idx] for idx in layer_indices]
                hidden_states = torch.stack(selected_layers, dim=-2)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {exc}") from exc
        else:
            raise ValueError(f"Unknown hidden_state_method: {hidden_state_method}")

        # Apply QA masking if needed
        if qa_masks is not None:
            # qa_masks shape: (batch_size, seq_len)
            # hidden_states shape: (batch_size, seq_len, num_features, hidden_dim)
            if not qa_masking:
                qa_masks_expanded = torch.ones_like(qa_masks)
            else:
                qa_masks_expanded = qa_masks

            # Expand qa_masks to match hidden_states dimensions
            mask_expanded = qa_masks_expanded.unsqueeze(-1).unsqueeze(-1)  # (batch, seq, 1, 1)
            hidden_states = hidden_states * mask_expanded

        # L2 normalization
        # Normalize per-token representations so dot products behave like cosine similarity
        hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)

        # Split into GT and Gen portions
        # GT: context_length:prompt_length, Gen: prompt_length:
        gt_hidden_states = hidden_states[:, context_length:prompt_length, :, :]
        gen_hidden_states = hidden_states[:, prompt_length:, :, :]

        # Reshape GT into blocks using unfold
        # gt_hidden_states: (batch, prompt_len - context_len, num_feat, hidden)
        # Need to reshape to: (batch, num_blocks, gen_len, num_feat, hidden)
        gt_hidden_states = gt_hidden_states.unfold(-3, generate_max_len, stride)
        # After unfold: (batch, num_blocks, num_feat, hidden, gen_len)
        # Permute to: (batch, num_blocks, gen_len, num_feat, hidden)
        gt_hidden_states = gt_hidden_states.permute(0, 1, 4, 2, 3)

        # Reshape Gen into blocks
        # gen_hidden_states: (batch, gen_len * num_blocks, num_feat, hidden)
        # Generation is interleaved: [tok0_blk0, tok0_blk1, tok0_blk2, tok1_blk0, tok1_blk1, tok1_blk2, ...]
        # Need to reshape to: (batch, num_blocks, gen_len, num_feat, hidden)
        # Strategy: reshape to (batch, gen_len, num_blocks, num_feat, hidden) then transpose
        batch_size = sequences.size(0)
        num_features = hidden_states.size(-2)
        hidden_dim = hidden_states.size(-1)
        gen_hidden_states = gen_hidden_states.reshape(
            batch_size, generate_max_len, num_blocks, num_features, hidden_dim
        ).transpose(1, 2)  # Swap gen_len and num_blocks dimensions
        # Result: (batch, num_blocks, gen_len, num_feat, hidden)

        gt_embedding = gt_hidden_states  # (batch, num_blocks, gen_len, num_feat, hidden)
        gen_embedding = gen_hidden_states  # (batch, num_blocks, gen_len, num_feat, hidden)

        # NOTE: Embed pooling, whitening, and final flattening now all handled in evaluation_metrics.py

        # Trainer critic forward returns float32 tensors to the caller.
        return gen_embedding.to(torch.float32), gt_embedding.to(torch.float32)
