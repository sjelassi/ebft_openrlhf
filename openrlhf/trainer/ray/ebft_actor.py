import math
import os
from abc import ABC
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler
from openrlhf.models import Actor, EBFTPolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean, build_strided_attention_mask_and_positions
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.run_config_utils import write_run_config
from ..ppo_utils import EBFTNaiveReplayBuffer
from openrlhf.utils.utils import zero_pad_sequences


logger = init_logger(__name__)

from .launcher import BaseModelActor


class ActorEBFTTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        **kwargs,
    ):
        """EBFTTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size

        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.max_epochs = self.args.max_epochs

        self.actor_loss_fn = EBFTPolicyLoss(
            policy_loss_type=self.args.policy_loss_type,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.replay_buffer = EBFTNaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.args.colocate_all_models and not self.args.async_train:
            self.use_cuda_ipc = True

        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, rl_ctl: float, ce_ctl: float, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = (
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle, #False,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}

        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):

                experience.to_device(device)
                status = self.training_step(experience, rl_ctl, ce_ctl, kl_ctl, step)
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]
                short_status = {
                    "act_loss": status["policy_loss"],
                    "reward": status["reward"],
                    "return": status["return"],
                    "gen_len": status["response_length"],
                    "tot_len": status["total_length"],
                    "kl": status["kl"],
                    "act_lr": status["actor_lr"],
                }

                if "entropy_loss" in status:
                    short_status["ent_loss"] = status["entropy_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, rl_ctl: float, ce_ctl: float, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()
        device = torch.cuda.current_device()

        # Extract sequence tensors from experience
        prompts = experience.prompts  # X: Original full prompt
        full_sequences = experience.full_sequences  # Full sequence (prompt + generated)

        doc_ids = experience.doc_ids  # B, S
        qa_masks = experience.qa_masks  # B, S
        
        # Calculate sequence dimensions
        prompt_length = prompts.shape[1]  # Length of original prompt
        generate_max_len = self.args.generate_max_len  # Total tokens to generate
        context_length = self.args.context_max_len   # Total context length used for generating each block
        stride = self.args.stride  # Stride between blocks
        # Each block covers a stride-offset window; +1 because both endpoints are inclusive.
        num_blocks = (prompt_length - generate_max_len - context_length )// stride + 1  # Number of prediction blocks

        # Extract training data from experience
        action_mask = experience.action_mask
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        # Build strided attention mask and position IDs for the full sequence
        # This creates the attention pattern for multi-token prediction
        attention_mask, pos_ids = build_strided_attention_mask_and_positions(
            full_sequence_length=full_sequences.size(1),  # Total sequence length
            prompt_length=prompts.size(1),  # Original prompt length
            context_length=context_length,
            generation_step=generate_max_len,  # Number of tokens generated
            max_generation_length=generate_max_len,  # Total number of tokens to generate
            stride=stride,
            num_blocks=num_blocks,
            device=device,
            doc_ids=doc_ids[:,:prompts.size(1)],
            document_masking=self.args.document_masking,
        )

        # Compute actor forward pass with strided attention
        # This calculates the action log probabilities for the full sequence
        action_log_probs, output = self.actor(
            full_sequences.to(device),  # Full sequence (prompt + generated)
            torch.ones_like(action_mask).to(device),  # Action mask for generated tokens
            attention_mask.to(device),  # Strided attention mask
            pos_ids=pos_ids,  # Position IDs for proper positional encoding
            return_logprobs=True,
            ring_attn_group=self.strategy.ring_attn_group,
            return_output=True,
            return_entropy=self.args.entropy_loss_coef is not None,
            prompt_len=prompt_length,  # Original prompt length
            context_len=context_length,
            num_blocks=num_blocks,  # Number of prediction blocks
            stride=stride,  # Context stride
        )


        # loss function
        actor_loss, ce_loss = self.actor_loss_fn(
            action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            qa_masks=qa_masks[:, 1:],
            qa_masking=self.args.qa_masking,
        )

        
        if self.args.use_kl_loss:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
            kl_loss = masked_mean(kl, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
        else:
            kl_loss = 0

        loss = actor_loss * rl_ctl + ce_loss * ce_ctl + kl_loss * kl_ctl
        

        # mixtral
        if self.aux_loss:
            loss += output.aux_loss * self.args.aux_loss_coef
        
        # entropy loss
        if self.args.entropy_loss_coef is not None:
            entropy_loss = masked_mean(output.entropy[:, -experience.action_mask.shape[1] :], experience.action_mask)
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef
 
 
        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]
           

        self.strategy.backward(loss, self.actor, self.actor_optim)

        grad_norm_value: Optional[float] = None
        if getattr(self.args, "log_gradients", False):
            should_log = True
            if self.args.use_dynamic_batch:
                should_log = bool(self.replay_buffer.dynamic_optimizer_step[step])

            if should_log:
                grad_norm_value = self._compute_actor_grad_norm()

        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        # status
        status = {"policy_loss": actor_loss.detach().item(), "ce_loss": ce_loss.detach().item(), "actor_lr": self.actor_scheduler.get_last_lr()[0], "exp_log_probs": torch.mean(torch.exp(action_log_probs)).detach().item(), "ce_ctl": ce_ctl, "rl_ctl": rl_ctl}
        status['actor_loss'] = loss.detach().item()
        if self.args.use_kl_loss:
            status['kl_ctl_loss'] = kl_loss.detach().item()  * kl_ctl
        else:
            status['kl_ctl_loss'] = kl_loss  * kl_ctl
        if self.args.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()
        # merge logs from info field
        for k, v in experience.info.items():
            if isinstance(v, list):
                try:
                    status[k] = torch.tensor(v, dtype=torch.float).mean().item()
                except:
                    logger.warning(f"WARNING: cannot convert {v} to tensor")
            elif isinstance(v, torch.Tensor):
                status[k] = v.float().mean().item()
        if grad_norm_value is not None:
            scheduler_step = getattr(self.actor_scheduler, "global_step", None)
            status["actor_grad_norm"] = grad_norm_value
        else:
            status["actor_grad_norm"] = 0.0

        return status

    def _compute_actor_grad_norm(self) -> Optional[float]:
        """Return the global gradient norm of the actor, handling ZeRO stages gracefully."""
        engine = getattr(self.actor, "model", None)
        grad_norm: Optional[float] = None
        try:
            if engine is not None:
                if hasattr(engine, "get_global_grad_norm"):
                    grad_norm = engine.get_global_grad_norm()
                elif hasattr(engine, "get_global_norm"):
                    grad_norm = engine.get_global_norm()
            if grad_norm is None:
                total_norm_sq = 0.0
                has_grad = False
                stage = getattr(self.strategy, "stage", None)
                gather_enabled = stage == 3
                for param in self.actor.parameters():
                    ctx = deepspeed.zero.GatheredParameters([param], enabled=gather_enabled) if gather_enabled else nullcontext()
                    with ctx:
                        grad = param.grad
                        if grad is None:
                            continue
                        has_grad = True
                        param_grad = grad.detach()
                        if not torch.is_floating_point(param_grad):
                            param_grad = param_grad.float()
                        total_norm_sq += param_grad.norm(2).item() ** 2
                if has_grad:
                    grad_norm = math.sqrt(total_norm_sq)
        except Exception as exc:
            logger.warning(f"Failed to compute actor gradient norm: {exc}")
            grad_norm = None
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().cpu().item()
        return grad_norm

   
@ray.remote(num_gpus=1)
class EBFTPolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        # self.vllm_engines = vllm_engines
        self.max_steps = max_steps
        self.debug = args.debug 

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,  
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        # configure optimizer
        if args.actor_learning_rate == 0:
            # Create optimizer with lr=0 (no critic training)
            actor_optim = strategy.create_optimizer(
                actor, lr=0.0, betas=args.adam_betas, weight_decay=args.l2
            )

            # Create dummy scheduler that maintains lr=0
            class DummyScheduler:
                """Dummy scheduler that keeps learning rate at 0."""
                def __init__(self, optimizer):
                    self.optimizer = optimizer

                def step(self):
                    # Ensure lr stays at 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 0.0

                def get_last_lr(self):
                    return [0.0]  # Two learning rates: backbone and head

                def state_dict(self):
                    return {"lr": [group["lr"] for group in self.optimizer.param_groups]}

                def load_state_dict(self, state_dict):
                    lr_list = state_dict.get("lr")
                    if lr_list is not None:
                        for group, lr in zip(self.optimizer.param_groups, lr_list):
                            group["lr"] = lr

            actor_scheduler = DummyScheduler(actor_optim)
        else:
            actor_optim = strategy.create_optimizer(
                actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
            )

            actor_scheduler = get_scheduler(
                args.lr_scheduler,
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * args.actor_lr_warmup_ratio) if args.actor_lr_warmup_ratio is not None else math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
            )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_actor_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states["data_loader_state_dict"]

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorEBFTTrainer(
            strategy,
            self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
        )

       
    def fit(self, rl_ctl: float = 0, ce_ctl: float = 0, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(rl_ctl, ce_ctl, kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.actor, # self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        eval_full_ppl_mode: bool = False,
        return_squared_loss: bool = False,
        prompt_lens=int,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                eval_full_ppl_mode=eval_full_ppl_mode,
                return_squared_loss=return_squared_loss,
                prompt_len=prompt_lens,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")
    

  

    @torch.no_grad()
    def forward_strided_blocks(
        self,
        full_sequences: torch.LongTensor,  # Full sequence (prompt + generated tokens)
        action_mask: Union[int, list[int]],  # Mask for generated tokens
        prompt_length: int,  # Original full prompt length (X length)
        generation_step: int,  # Total number of tokens to generate
        num_blocks: int,  # Number of parallel prediction blocks
        stride: int,  # Context stride between consecutive blocks,
        context_length: int,
    ) -> torch.Tensor:
        """
        Compute action log probabilities for full sequences using strided attention.

        This function computes forward passes for sequences generated using the strided
        multi-token prediction approach. It builds appropriate attention masks that allow
        each prediction block to only see its designated context window.

        The full sequence consists of:
        - Original prompt (X) of length prompt_length
        - Generated tokens of length generation_step

        Args:
            full_sequences: The full sequence tensor (XZ) containing prompt + generated tokens
            action_mask: Mask indicating which positions are actions (generated tokens)
            prompt_length: Length of the original full prompt (X)
            generation_step: Total number of tokens generated (generation length)
            num_blocks: Number of parallel prediction blocks
            stride: Context stride between consecutive blocks
            context length: Length of context used for each block

        Returns:
            Action log probabilities for the full sequence
        """
        device = torch.cuda.current_device()
        self.actor.eval()

        with torch.inference_mode():
            # Ensure full_sequences is a proper tensor with batch dimension
            if isinstance(full_sequences, list):
                full_sequences_batch = torch.tensor(full_sequences, dtype=torch.long)
            else:
                full_sequences_batch = full_sequences

            if full_sequences_batch.dim() == 1:
                full_sequences_batch = full_sequences_batch.unsqueeze(0)
            # Build the strided attention mask for the full sequence
            # This mask ensures each block only attends to its designated context window
            # The mask is built based on the full sequence length and generation parameters
            attention_mask, position_ids = build_strided_attention_mask_and_positions(
                full_sequence_length=full_sequences_batch.size(1),  # Total sequence length
                prompt_length=prompt_length,  # Original prompt length,
                context_length=context_length,
                generation_step=generation_step,  # Number of tokens generated
                max_generation_length=generation_step,  # Total number of tokens to generate
                stride=stride,
                num_blocks=num_blocks,
                device=device,
                document_masking=self.args.document_masking,
            )

            # Forward pass through the actor model to compute log probabilities
            # Use the full sequence with strided attention for multi-token prediction
            action_log_probs = self.actor(
                full_sequences_batch.to(device),
                None,  # action_mask parameter (handled internally by the model)
                attention_mask.to(device),
                pos_ids=position_ids,
                return_logprobs=True,
                ring_attn_group=self.strategy.ring_attn_group,
                prompt_len=prompt_length,
                context_len=context_length,
                num_blocks=num_blocks,
                stride=stride,
            )
            del full_sequences_batch
            del attention_mask, position_ids
            torch.cuda.empty_cache()

        self.actor.train()  # Reset model to training mode
        return action_log_probs.to("cpu")




    def generate_standard_ar(
        self,
        prompts,
        # prompt_token_ids: List[int],
        generate_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """
        Standard autoregressive generation - generate one token at a time.

        This function implements classical autoregressive generation where:
        - We generate one token at a time
        - Each token sees all previous tokens (full causal attention)
        - Tokens are appended sequentially to the prompt

        Args:
            prompt_token_ids: List of token IDs for the original prompt (batch of sequences)
            generate_length: Number of tokens to generate
            temperature: Sampling temperature for generation (0 = greedy)
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            Complete sequence containing prompt + generated tokens
        """
        device = torch.cuda.current_device()
        self.actor.eval()

        with torch.inference_mode():
            # Generate tokens one at a time (standard autoregressive)
            out = self.actor.generate_for_downstream(
                # sequences=sequence,
                sequences=prompts.to(device),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=generate_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Return as CPU tensor
        self.actor.train()
        return out.detach().to("cpu")




    def generate_strided_blocks(
        self,
        prompt_token_ids: List[int],
        doc_ids: List[int],
        stride: int,
        context_length: int,
        generate_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        document_masking: bool = False,
    ):
        """
        Generate multiple blocks in parallel using multi-token prediction with stride.

        This function implements the strided generation approach where:
        - We generate multiple tokens in parallel, one for each stride-sized block
        - Each block uses a different prefix of the original prompt as context
        - The generated tokens are accumulated to form an augmented sequence

        Args:
            prompt_token_ids: List of token IDs for the original full prompt (X)
            stride: Context stride between blocks (number of tokens offset between blocks)
            context_length: Total number of tokens to provide as context for generation
            generate_length: Total number of tokens to generate across all blocks
            temperature: Sampling temperature for generation
            top_p: Top-p sampling threshold (not currently used)

        Returns:
            Augmented sequence (Z) stored in self.Z containing:
            - Partial prompt (X minus last stride tokens)
            - All generated tokens
        """
        device = torch.cuda.current_device()
        self.actor.eval()

        # ---- Input normalization & debug ----
        if isinstance(prompt_token_ids, torch.Tensor):
            # allow tensor input, but ensure it's 2D
            if prompt_token_ids.dim() != 2:
                raise ValueError(f"prompt_token_ids must be 2-D; got shape {tuple(prompt_token_ids.shape)}")
            original_prompt_tensor = prompt_token_ids.to(device=device, dtype=torch.long, non_blocking=True)
            doc_ids = doc_ids.to(device=device, dtype=torch.long, non_blocking=True)
        else:
            # expect List[List[int]]
            if not prompt_token_ids or not isinstance(prompt_token_ids[0], (list, tuple)):
                raise ValueError("prompt_token_ids must be List[List[int]] (2-D).")
            original_prompt_tensor = torch.as_tensor(prompt_token_ids, device=device, dtype=torch.long)
            doc_ids = torch.as_tensor(doc_ids, device=device, dtype=torch.long)

        batch_size = original_prompt_tensor.shape[0]
        prompt_length    = original_prompt_tensor.shape[1]
        if prompt_length - generate_length <= 0:
            raise ValueError(f"generate_length ({generate_length}) must be < seq_len ({prompt_length}).")

        # stride/grid checks
        assert (prompt_length - generate_length - context_length) % stride == 0, (
            f"prompt_length {prompt_length - generate_length - context_length} must be divisible by stride {stride}"
        )
        num_blocks = (prompt_length - generate_length - context_length) // stride + 1

        # Z starts as prefix
        full_sequence = original_prompt_tensor.clone()

        # Store generated tokens for each block (for debugging/analysis)
        generated_blocks = []
        for _ in range(num_blocks):
            generated_blocks.append([])

        with torch.inference_mode():  # stronger than no_grad; also skips version counters
            # Generate tokens iteratively - at each step, we generate one token per block
            for generation_step in range(generate_length):
                # Build the strided attention mask for the current augmented sequence
                # This mask allows each block to see only its designated context window
                attention_mask, position_ids = build_strided_attention_mask_and_positions(
                    full_sequence_length=full_sequence.shape[1],  # Current length of full sequence
                    prompt_length=prompt_length,                       # Original prompt length
                    context_length=context_length,                     # Context length for each block
                    generation_step=generation_step,                   # Current generation step
                    max_generation_length=generate_length,             # Total number of tokens to generate
                    stride=stride,
                    num_blocks=num_blocks,
                    device=device,
                    doc_ids=doc_ids,
                    document_masking=document_masking,
                )

                # Shape checks
                B, L = full_sequence.shape
                assert position_ids.shape == (B, L), \
                    f"position_ids shape {position_ids.shape} != full_sequence {full_sequence.shape}"

                if attention_mask.dim() == 2:
                    # e.g. [B, L]
                    assert attention_mask.shape == (B, L), \
                        f"2D mask shape {attention_mask.shape} != (B={B}, L={L})"
                elif attention_mask.dim() == 4:
                    # e.g. [B, 1, L, L] additive mask
                    b_m, one_m, q_len, k_len = attention_mask.shape
                    assert b_m == B, f"mask batch {b_m} != B {B}"
                    assert q_len == L and k_len == L, \
                        f"mask Q/K dims {attention_mask.shape} != L={L}"
                else:
                    raise ValueError(f"Unexpected attention_mask dim: {attention_mask.dim()}")

                # Forward pass through the model to get logits for all positions
                output = self.actor(
                    full_sequence,
                    attention_mask=attention_mask,
                    pos_ids=position_ids,
                    return_output=True,
                )
                all_logits = output.logits  # Shape: [batch_size, sequence_length, vocab_size]
                del output

                # Calculate which positions to extract logits from
                # Each block predicts the next token from a different position
                logit_positions = []
                for block_idx in range(num_blocks):
                    if generation_step == 0:
                        prediction_position = block_idx * stride + context_length - 1
                    else:
                        prediction_position = prompt_length + (generation_step - 1) * num_blocks + block_idx
                    logit_positions.append(prediction_position)

                # Extract logits from the calculated positions
                position_indices = torch.tensor(logit_positions, dtype=torch.long, device=device)

                # Gather one logit vector per block: [B, seq_len, V] → [B, num_blocks, V]
                block_logits = all_logits.index_select(1, position_indices)  # [B, num_blocks, V]

                # Apply temperature scaling to logits
                if temperature > 0:
                    block_logits = block_logits / temperature
                    # Convert logits to probabilities
                    probabilities = torch.softmax(block_logits, dim=-1) # [B, num_blocks, V]

                    # Apply top-p (nucleus) sampling if specified
                    if top_p < 1.0:
                        batch_size, num_blocks_local, vocab_size = probabilities.shape
                        # Reshape for processing: [batch_size * num_blocks, vocab_size]
                        probs_flat = probabilities.view(-1, vocab_size)

                        # Sort probabilities in descending order
                        sorted_probs, sorted_indices = torch.sort(probs_flat, descending=True, dim=-1)

                        # Calculate cumulative probabilities
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        # Find where cumulative probability exceeds top_p
                        # Shift right by 1 to include the token that crosses the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False

                        # Set probabilities of tokens to remove to 0
                        indices_to_remove = torch.zeros_like(probs_flat, dtype=torch.bool)
                        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                        del sorted_probs, cumulative_probs, sorted_indices_to_remove
                        probs_flat[indices_to_remove] = 0
                        del indices_to_remove, sorted_indices

                        # Renormalize the remaining probabilities
                        probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
                    else:
                        batch_size = probabilities.shape[0]
                        # Flatten for sampling: [batch_size*num_blocks, vocab_size]
                        probs_flat = probabilities.view(-1, probabilities.shape[-1])

                    # Sample next tokens for all blocks
                    sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*NB]
                    # Reshape from flat [B*NB] back to [B, num_blocks]
                    sampled_tokens = sampled_tokens_flat.view(batch_size, -1)  # [B, NB]
                    # Clean up memory
                    del all_logits, block_logits, probabilities, probs_flat, sampled_tokens_flat
                    del attention_mask, position_ids
                else:
                    # Greedy decoding (temperature=0): take argmax
                    sampled_tokens = torch.argmax(block_logits, dim=-1)  # [B, num_blocks]
                    

                # Append num_blocks new tokens to the sequence: [B, L] → [B, L+num_blocks]
                full_sequence = torch.cat([full_sequence, sampled_tokens], dim=1)
                del sampled_tokens
                torch.cuda.empty_cache()
               
                
        # Return per-sample CPU lists (aligned with input order)    
        out = full_sequence.detach().to("cpu")
        self.actor.train()
        return out
    
    
    
    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)
    
    def flush_buffer(self):
        """Clear the replay buffer."""
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states, save_hf=False):
        args = self.strategy.args
        if not args.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
            if self.strategy.is_rank_0():
                write_run_config(
                    os.path.join(args.ckpt_path, "_actor", tag),
                    args,
                    tag=tag,
                    client_states=client_states,
                )
        if self.save_hf_ckpt or save_hf:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.actor, #self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
            if self.strategy.is_rank_0():
                write_run_config(save_path, args, tag=tag, client_states=client_states)
        # wait
        torch_dist_barrier_and_cuda_sync()

