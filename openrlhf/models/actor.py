from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import compute_entropy, log_probs_from_logits, compute_squared_loss
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logger.info("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @property
    def config(self):
        """Return the config of the model for compatibility with save_model."""
        return self.model.config

    def prepare_logprobs(self, logits, prompt_len, context_len, num_blocks, stride):
        """Prepare logits for log-probability computation.

        Args:
            logits: Model output logits tensor [batch_size, seq_len, vocab_size]
            prompt_len: Length of the prompt sequence
            context_len: Length of the context
            num_blocks: Number of blocks to process
            stride: Step size between blocks

        Returns:
            Concatenated logits tensor for log-probability computation
        """
        # Extract logits for the prompt region (excluding the last token)
        prompt_logits = logits[:, :prompt_len, :]
        prompt_logits_for_tokens = prompt_logits[:, :-1, :]  # Logits that produced tokens in the prompt

        # Calculate indices for block boundaries (using torch for gradient flow)
        block_boundary_indices = torch.arange(0, num_blocks, device=logits.device) * stride + context_len - 1

        # Extract logits at block boundaries (logits that produced first token in each block)
        block_boundary_logits = logits[:, block_boundary_indices, :]

        # Extract logits for the continuation region (after prompt, excluding last num_blocks)
        continuation_logits = logits[:, prompt_len:-num_blocks, :]

        # Concatenate all components for final log-probability computation
        concatenated_logits = torch.cat([
            prompt_logits_for_tokens,
            block_boundary_logits,
            continuation_logits
        ], dim=1)

        return concatenated_logits
    
    def prepare_labels(self, full_seq):
        """Prepare labels for log-probability computation."""
        labels = full_seq[:, 1:]
        return labels
    


    def generate_for_downstream(
            self,
            sequences, #: torch.LongTensor,
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            eos_token_id: int,
            pad_token_id: int,

    ) -> torch.Tensor:
        output = self.model.generate(
                **sequences,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return output
 

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.LongTensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        return_squared_loss=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        return_entropy=False,
        eval_full_ppl_mode=False,
        prompt_len=Optional[int],
        context_len=Optional[int],
        num_blocks=Optional[int],
        stride=Optional[int],
    ) -> torch.Tensor:
        """Forward pass through the actor model.

        Args:
            sequences: Input token sequences [batch_size, seq_length]
            action_mask: Mask for computing action log probabilities
            attention_mask: Attention mask for the transformer
            pos_ids: Position IDs for custom positional encoding
            return_output: Whether to return the full model output
            allgather_logits: Whether to gather logits across processes (unused)
            return_logprobs: Whether to return log probabilities
            ring_attn_group: Process group for ring attention (unused)
            return_entropy: Whether to compute and return entropy
            eval_full_ppl_mode: Evaluation mode for full perplexity computation
            prompt_len: Length of the prompt for logprob preparation
            context_len: Length of the context for logprob preparation
            num_blocks: Number of blocks for prediction
            stride: Step size between blocks
        Returns:
            Action log probabilities or model output depending on flags
        """
        batch_size, seq_length = sequences.size()
        forward_attention_mask = attention_mask

        # Prepare position IDs based on evaluation mode
        if eval_full_ppl_mode:
            # For full perplexity evaluation, compute position IDs from attention mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # Use provided position IDs for standard forward pass
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = pos_ids
        # Run model forward pass
        output = self.model(sequences, attention_mask=forward_attention_mask, position_ids=position_ids)

        # Convert logits to float32 for numerical stability
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        # Compute entropy if requested
        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            setattr(output, "entropy", entropy[:, :-1])

        # Check if we need to compute action log probabilities
        should_return_action_log_probs = action_mask is not None

        # Early return if only model output is needed
        if not should_return_action_log_probs and not return_logprobs:
            assert return_output
            return output

        
        # Prepare logits and labels based on evaluation mode
        if eval_full_ppl_mode:
            processed_logits = output["logits"] 
            target_labels = torch.roll(sequences, shifts=-1, dims=1)
            if return_squared_loss:
                loss = compute_squared_loss(processed_logits, target_labels)
                return loss
        else:
            # Standard mode: prepare logits using the custom preparation function
            processed_logits = self.prepare_logprobs(output["logits"], prompt_len, context_len, num_blocks, stride)
            target_labels = self.prepare_labels(sequences)
        
        # Compute log probabilities from logits and target labels
        log_probs = log_probs_from_logits(processed_logits, target_labels, temperature=self.temperature, prompt_len=prompt_len-1)
    
        if eval_full_ppl_mode:
            log_probs = log_probs[:, :-1]


        # Handle different return modes
        # If only log probabilities are requested (not action-specific)
        if not should_return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        # Extract action-specific log probabilities using the action mask
        # Align with action positions by taking the last action_mask.shape[1] positions
        action_log_probs = log_probs[:, -action_mask.shape[1]:] * action_mask.float()

        # Return action log probabilities with optional model output
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
    
    def save_pretrained(self, output_dir: str, state_dict=None, **kwargs):
        """
        Save the model to a directory in HuggingFace format.
        
        Args:
            output_dir: Directory to save the model
            state_dict: Optional state dict to save
            **kwargs: Additional arguments for save_pretrained
        """
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir, state_dict=state_dict, **kwargs)
        else:
            # Fallback to torch.save if model doesn't have save_pretrained
            import os
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(state_dict if state_dict is not None else self.model.state_dict(), save_path)