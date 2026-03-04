import math
import os
import socket
from abc import ABC
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler
from openrlhf.datasets.utils import blending_datasets, create_eval_data
from openrlhf.models import OriginalActor, PolicyLoss, SFTLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger
from datatrove.utils.dataset import DatatroveFolderDataset
from openrlhf.datasets import SFTDataset, DatatroveSFTDataset
from ..ppo_utils import NaiveReplayBuffer
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: OriginalActor,
        ema_model: OriginalActor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
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

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and each of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, kl_ctl: float):
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
            shuffle=not not_shuffle,
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
                status = self.training_step(experience, kl_ctl, step)
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

    def training_step(self, experience: Experience, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.entropy_loss_coef is not None,
            override_temperature=True
        )

        # loss function
        actor_loss, clip_ratio, ppo_kl, ratio_mean, ratio_std = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()

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

        loss = actor_loss + kl_loss * kl_ctl
        
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
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # status
        status = {"policy_loss": actor_loss.detach().item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        status['actor_loss'] = loss.detach().item()
        status['ratio_mean'] = ratio_mean.detach().item()
        status['ratio_std'] = ratio_std 
        if self.args.use_kl_loss:
            status['kl_ctl_loss'] = kl_loss.detach().item()  * kl_ctl
        else:
            status['kl_ctl_loss'] = kl_loss  * kl_ctl
        if self.args.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()

        # merge logs from info field
        for k, v in experience.info.items():
            if isinstance(v, list):
                status[k] = torch.tensor(v, dtype=torch.float).mean().item()
            elif isinstance(v, torch.Tensor):
                status[k] = v.float().mean().item()
        return status

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = OriginalActor(
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

        if args.enable_ema:
            ema_model = OriginalActor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
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

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states["data_loader_state_dict"]

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

        ## eval perplexity dataloader 
        # if getattr(args, "eval_dataset", None):
        #     eval_data = create_eval_data(args, self.tokenizer, strategy)
        #     print(f"EVALDATA (PPOACTOR): {eval_data[0]}")
        #     # Use the shared function to create eval_data
        #     eval_perplexity_dataset = DatatroveSFTDataset(
        #                     eval_data,
        #                     self.tokenizer,
        #                     1024,  # Use the full sequence length from the dataset
        #                     args.eval_max_samples,
        #                     strategy,
        #                     pretrain_mode=True # Always use pretrain_mode=True for perplexity evaluation
        #                 )
            
           
               
        #     self.eval_perplexity_dataloader = self.strategy.setup_dataloader(
        #         eval_perplexity_dataset,
        #         args.micro_train_batch_size,
        #         True,
        #         False,
        #         eval_perplexity_dataset.collate_fn,
        #     )
        #     self.sft_loss_fn = SFTLoss()

        #     # --- NEW: ED eval dataloader (windows with stride=1) ---
        #     # Uses the SAME eval_data constructed above.
            
            
        #     eval_ed_dataset = EDDataset(
        #         eval_data,
        #         self.tokenizer,
        #         args.prompt_max_len,
        #         args.generate_max_len,
        #         args.stride,           # stride
        #         args.eval_max_samples # respect your config
        #     )
    
        #     self.eval_ed_dataloader = self.strategy.setup_dataloader(
        #         eval_ed_dataset,
        #         args.micro_train_batch_size,
        #         True,
        #         False,
        #         eval_ed_dataset.collate_fn,
        #     )
        #     print(f"BATCH EVAL PPOACTOR: {next(iter(self.eval_ed_dataloader))}")

    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        override_temperature: bool = False,
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
                override_temperature=override_temperature,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        self.trainer._broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_cuda_sync()

    # def compute_perplexity(self, steps=0):

    #     self.actor.eval()
    #     with torch.no_grad():
    #         total_loss = 0
    #         total_tokens = 0
    #         step_bar = tqdm(
    #             range(self.eval_perplexity_dataloader.__len__()),
    #             desc="Eval stage of steps %d" % steps,
    #             disable=not self.strategy.is_rank_0(),
    #         )

    #         for inputs, attention_masks, loss_masks in self.eval_perplexity_dataloader:
    #             inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
    #             attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
    #             loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
         
    #             per_token_log_probs = self.actor(
    #                 inputs,
    #                 attention_mask=attention_mask,
    #                 return_logprobs=True,
    #                 ring_attn_group=self.strategy.ring_attn_group,
    #                 override_temperature=True
    #             )
    #             # self.actor.temperature = original_temp  # Restore original temperature

    #             # Calculate loss for this batch
    #             loss = self.sft_loss_fn(per_token_log_probs, loss_mask[:, :-1])

    #             # Count the number of tokens we're computing loss over
    #             num_tokens = loss_mask[:, :-1].sum().item()

 
    #             # Accumulate weighted loss (loss is already averaged over tokens in the batch)
    #             total_loss += loss.item() * num_tokens
    #             total_tokens += num_tokens

    #             # Calculate running average
    #             if total_tokens > 0:
    #                 avg_loss = total_loss / total_tokens
    #                 perplexity = math.exp(avg_loss)
                    
    #             else:
    #                 raise ValueError("total_tokens = 0 in eval perplexity computation")
                    
    #             bar_dict = {"eval/gpt_loss": avg_loss, "eval/perplexity": perplexity}
    #             step_bar.update()
    #             logs = self.strategy.all_reduce(bar_dict)
    #             step_bar.set_postfix(logs)

    #     self.actor.train()
    #     return logs

    # def compute_perplexity_ed_dataset(self, steps=0):
    #     import math
    #     self.actor.eval()
    #     device = torch.cuda.current_device()

    #     # Preconditions: ensure eval dataloader and loss fn are initialized
    #     if not hasattr(self, "eval_ed_dataloader") or not hasattr(self, "sft_loss_fn"):
    #         raise RuntimeError(
    #             "compute_perplexity_ed_dataset requires eval_ed_dataloader and sft_loss_fn to be initialized. "
    #             "Ensure args.eval_dataset is set and init_model_from_pretrained has built the eval loaders."
    #         )

    #     total_loss = 0.0
    #     total_tokens = 0

    #     step_bar = tqdm(
    #         self.eval_ed_dataloader,
    #         desc=f"Eval ED-Completion PPL steps {steps}",
    #         disable=not self.strategy.is_rank_0(),
    #     )

    #     for batch in step_bar:
    #         # Convert lists-of-ids to dense tensors explicitly to avoid ragged/ambiguous constructions
    #         contexts_list = batch["context_ids"]
    #         targets_list = batch["real_sequence_ids"]
    #         ctx = torch.stack([torch.tensor(x, dtype=torch.long, device=device) for x in contexts_list], dim=0)
    #         tgt = torch.stack([torch.tensor(x, dtype=torch.long, device=device) for x in targets_list], dim=0)
    #         # print(f"TARGET TOKEN: {tgt}")
    #         B, L = ctx.shape
    #         _, C = tgt.shape
    #         assert L >= 1 and C > 0, "ED perplexity requires prompt_max_len >= 1 and generate_max_len > 0."

    #         # Build inputs and masks
    #         inputs = torch.cat([ctx, tgt], dim=1)                  # (B, L+C)
    #         attn   = torch.ones_like(inputs, dtype=torch.long)     # (B, L+C)
    #         lmask  = torch.zeros_like(inputs, dtype=torch.float32) # (B, L+C)
    #         lmask[:, L - 1 : L - 1 + C] = 1.0                      # score only completion tokens

    #         with torch.no_grad():
    #             per_token_logp = self.actor(
    #                 inputs,
    #                 attention_mask=attn,
    #                 return_logprobs=True,
    #                 ring_attn_group=self.strategy.ring_attn_group,
    #                 override_temperature=True,
    #             )  # (B, L+C-1)

    #         loss = self.sft_loss_fn(per_token_logp, lmask[:, :-1])    # masked mean NLL over completions
    #         num_tokens = float(lmask[:, :-1].sum().item())

    #         # accumulate as sums
    #         loss_val = float(loss.item()) if math.isfinite(loss.item()) else 0.0
    #         total_loss += loss_val * num_tokens
    #         total_tokens += num_tokens

    #         # running display (local)
    #         if total_tokens > 0:
    #             avg_loss = total_loss / total_tokens
    #             ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float("nan")
    #         else:
    #             avg_loss, ppl = float("nan"), float("nan")
    #         step_bar.set_postfix({"ed_comp_tok_batch": int(num_tokens), "ed_comp_ppl": ppl})

    #     # DDP reduction (use float64 for stability)
    #     reduced = self.strategy.all_reduce({
    #         "ed_comp_loss_sum": torch.tensor(total_loss, dtype=torch.float64, device=device),
    #         "ed_comp_tok_sum": torch.tensor(total_tokens, dtype=torch.float64, device=device),
    #     })
    #     tok_sum = float(reduced["ed_comp_tok_sum"].detach().cpu())
    #     if tok_sum > 0:
    #         loss_mean = float(reduced["ed_comp_loss_sum"].detach().cpu()) / tok_sum
    #         ppl = math.exp(loss_mean)
    #     else:
    #         loss_mean, ppl = float("nan"), float("nan")

    #     self.actor.train()
    #     return {"eval/ed_loss": loss_mean, "eval/ed_ppl": ppl}

    

















    ## old compute ppl for ed dataset
    # def old_compute_perplexity_ed_dataset(self, steps=0):
    #     """
    #     Compute perplexity on completion tokens using the ED eval dataset directly.
    #     This mirrors compute_perplexity() but restricts loss to completion positions only.
    #     """
    #     import math
    #     self.actor.eval()
    #     device = torch.cuda.current_device()

    #     total_loss = 0.0
    #     total_tokens = 0

    #     step_bar = tqdm(
    #         self.eval_ed_dataloader,
    #         desc="Eval ED-Completion PPL steps %d" % steps,
    #         disable=not self.strategy.is_rank_0(),
    #     )

    #     pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (
    #         getattr(self.tokenizer, "eos_token_id", 0)
    #     )

    #     for batch in step_bar:
    #         contexts = batch["context_ids"]
    #         targets  = batch["real_sequence_ids"]

    #         sequences = []
    #         attention_masks = []
    #         loss_masks = []

    #         for ctx_ids, tgt_ids in zip(contexts, targets):
    #             ctx_tensor = torch.tensor(ctx_ids, dtype=torch.long)
    #             tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    #             seq = torch.cat([ctx_tensor, tgt_tensor], dim=0)  # [S]
    #             sequences.append(seq.unsqueeze(0))                 # [1,S]

    #             attn = torch.ones_like(seq, dtype=torch.long)
    #             attention_masks.append(attn.unsqueeze(0))

    #             # Build mask for next-token prediction over completion tokens only
    #             m = torch.zeros_like(seq, dtype=torch.float32)
    #             L = ctx_tensor.numel()
    #             C = tgt_tensor.numel()
    #             if L >= 1 and C > 0:
    #                 start = L - 1
    #                 end = start + C
    #                 # print(f"L: {L}; C: {C}; start: {start}; end: {end}; SEQ: {seq.shape}")
    #                 m[start:end] = 1.0
    #             loss_masks.append(m.unsqueeze(0))
            

    #         # Pad to tensors [B,1,S] → [B,S]
    #         inputs = zero_pad_sequences(sequences, "right", pad_id).squeeze(1).to(torch.long).contiguous().to(device)
    #         attn   = zero_pad_sequences(attention_masks, "right").squeeze(1).to(torch.long).contiguous().to(device)
    #         lmask  = zero_pad_sequences(loss_masks, "right").squeeze(1).to(torch.float32).contiguous().to(device)

    #         # Ensure valid attention rows
    #         row_sums = attn.sum(dim=-1)
    #         if (row_sums == 0).any():
    #             bad = (row_sums == 0).nonzero(as_tuple=False).flatten()
    #             attn[bad, 0] = 1

    #         with torch.no_grad():
    #             # Save original temperature and use 1.0 for perplexity calculation
                
    #             per_token_logp = self.actor(
    #                 inputs,
    #                 attention_mask=attn,
    #                 return_logprobs=True,
    #                 ring_attn_group=self.strategy.ring_attn_group,
    #                 override_temperature=True
    #             )
                
                
    #         # Masked loss on completion positions only
    #         loss = self.sft_loss_fn(per_token_logp, lmask[:, :-1])
    #         num_tokens = lmask[:, :-1].sum().item()

    #         # Numerical stability checks
    #         loss_val = loss.item()
    #         if not torch.isfinite(loss).all() or not math.isfinite(loss_val):
    #             print(f"WARNING: Non-finite loss detected: {loss_val}, setting to 0")
    #             loss_val = 0.0

    #         total_loss += loss_val * num_tokens
    #         total_tokens += num_tokens

    #         # print(f"PERTOKEN: {per_token_logp}; LMASK: {lmask}; PERTOKENSHAPE: {per_token_logp.shape}; MASKSHAPE: {lmask.shape}; LOSS: {loss_val} NUMTOK: {num_tokens}; ")

    #         if total_tokens > 0:
    #             avg_loss = total_loss / total_tokens
    #             if math.isfinite(avg_loss):
    #                 ppl = math.exp(avg_loss)
    #             else:
    #                 print(f"WARNING: Non-finite average loss: {avg_loss}, setting perplexity to nan")
    #                 ppl = float("nan")
    #         else:
    #             avg_loss, ppl = float("nan"), float("nan")
    #         step_bar.set_postfix({"ed_comp_tok_batch": int(num_tokens), "ed_comp_ppl": ppl})

    #     reduced = self.strategy.all_reduce({
    #         "ed_comp_loss_sum": torch.tensor(total_loss, dtype=torch.float32, device=device),
    #         "ed_comp_tok_sum": torch.tensor(total_tokens, dtype=torch.float32, device=device),
    #     })
    #     tok_sum = float(reduced["ed_comp_tok_sum"].detach().cpu())
    #     if tok_sum > 0:
    #         loss_mean = float(reduced["ed_comp_loss_sum"].detach().cpu()) / tok_sum
    #         ppl = math.exp(loss_mean)
    #     else:
    #         loss_mean, ppl = float("nan"), float("nan")

    #     self.actor.train()
    #     return {"eval/ed_loss": loss_mean, "eval/ed_ppl": ppl}
    