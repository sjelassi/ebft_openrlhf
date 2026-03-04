import argparse
from datetime import datetime
import os
from pathlib import Path
import ray
from ray.util.placement_group import placement_group

from openrlhf.utils.logging_utils import init_logger


def nullable_float(value):
    """Custom argparse type for float arguments that can be None."""
    if value is None or value == "None" or value == "null" or value == "":
        return None
    return float(value)


def nullable_int(value):
    """Custom argparse type for int arguments that can be None."""
    if value is None or value == "None" or value == "null" or value == "":
        return None
    return int(value)

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    EBFTReferenceModelActor,
    RewardModelActor,
    EBFTRewardModelActor,
    EBFTCometRewardModelActor,
)
from openrlhf.trainer.ray.ebft_actor import EBFTPolicyModelActor
from openrlhf.trainer.ray.ebft_critic import EBFTCriticModelActor
from openrlhf.utils import get_strategy


def train(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # init vllm / actor /critic /ref /reward model
    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
    
    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        EBFTPolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )

    if args.init_kl_coef <= 0:
        ref_model = None
    else:
        ref_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            EBFTReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )

    if not args.colocate_all_models:
        pg = None

    reward_pretrains = (
        [pretrain.strip() for pretrain in args.reward_pretrain.split(",") if pretrain.strip()]
        if args.reward_pretrain
        else []
    )
    num_reward_models = len(reward_pretrains)
    reward_world_size = args.reward_num_nodes * args.reward_num_gpus_per_node
    reward_pg = None
    reward_pg_bundle_offset = 0

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        # bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        # Only one shared reward world, even if multiple reward models
        reward_bundle_count = reward_world_size if num_reward_models > 0 else 0
        total_bundles = critic_world_size + reward_bundle_count
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_bundles if total_bundles > 0 else critic_world_size)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        if num_reward_models > 0:
            reward_pg = pg
            reward_pg_bundle_offset = critic_world_size

    if args.critic_pretrain:
        critic_model = RayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            EBFTCriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        critic_model = None

    # multiple reward models
    if reward_pg is None and num_reward_models > 0 and args.colocate_reward_models:
        # We want all reward models to share the same GPU "world"
        total_reward_bundles = reward_world_size if reward_world_size > 0 else 1
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_reward_bundles)]
        reward_pg = placement_group(bundles, strategy="PACK")
        ray.get(reward_pg.ready())

    # Initialize reward models (optionally multiple, comma-separated)
    reward_models = []
    pg_for_reward = reward_pg if reward_pg is not None else pg
    reward_pg_active = reward_pg is not None
    for idx, r_pretrain in enumerate(reward_pretrains):
        is_comet = "comet" in r_pretrain.lower()
        # All reward models share the same PG bundles, starting from reward_pg_bundle_offset
        bundle_start = reward_pg_bundle_offset if reward_pg_active else 0
        reward_model = RayActorGroup(
            args.reward_num_nodes,
            args.reward_num_gpus_per_node,
            EBFTCometRewardModelActor if is_comet else EBFTRewardModelActor,
            pg=pg_for_reward,
            num_gpus_per_actor=0.2 if pg_for_reward else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
            pg_bundle_start=bundle_start,
        )
        # Annotate the group so trainers can distinguish comet vs. embedding models.
        reward_model.pretrain_name = r_pretrain
        reward_model.is_comet = is_comet
        reward_model.group_index = idx
        reward_models.append(reward_model)
    reward_models = reward_models if reward_models else None
    

    from openrlhf.trainer.ebft_trainer import EBFTTrainer
 
    # init EBFT trainer (Single controller)
    # NOTE: EBFTTrainer runs evaluation logic that may load additional models (e.g., factuality NLI,
    # ParaDetox detox metrics, etc). If those metrics should run on GPU, EBFTTrainer must be scheduled
    # with GPU resources so Ray exposes CUDA_VISIBLE_DEVICES inside that actor.
    eval_ds = str(getattr(args, "eval_dataset", "") or "").strip().lower()
    eval_down_enabled = int(getattr(args, "eval_down_steps", -1) or -1) != -1 and bool(getattr(args, "eval_dataset", None))

    trainer_needs_gpu = False

    trainer_options = {}
    if trainer_needs_gpu:
        # Default: lightweight eval metrics can share a GPU.
        trainer_options.update({"num_gpus": 0.2, "num_cpus": 0.2})
        if reward_pg is not None:
            trainer_options.update(
                {
                    "placement_group": reward_pg,
                    "placement_group_bundle_index": reward_pg_bundle_offset,
                }
            )
        elif pg is not None:
            trainer_options.update({"placement_group": pg, "placement_group_bundle_index": 0})

    ebft_trainer_ctor = EBFTTrainer.options(**trainer_options) if trainer_options else EBFTTrainer
    ebft_trainer = ebft_trainer_ctor.remote(
        args.pretrain,
        strategy,
        actor_model,
        critic_model,
        reward_models,
        ref_model,
        prompt_split=args.prompt_split,
        eval_split=args.eval_split,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # training update steps
    max_steps = ray.get(ebft_trainer.get_max_steps.remote())

    # init reference/reward/actor model
    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps))
    if reward_models is not None:
        for i, r_model in enumerate(reward_models):
            refs.extend(r_model.async_init_model_from_pretrained(strategy, reward_pretrains[i]))
    ray.get(refs)

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # train actor and critic model
    ray.get(ebft_trainer.fit.remote())

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -----------------------------------------------------------------------
    # Infrastructure: Ray node/GPU layout and colocation
    # -----------------------------------------------------------------------
    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument("--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model")
    parser.add_argument("--colocate_actor_ref", action="store_true", default=False, help="share GPUs between actor and reference model")
    parser.add_argument("--colocate_critic_reward", action="store_true", default=False, help="share GPUs between critic and reward model")
    parser.add_argument("--colocate_reward_models", action="store_true", default=False, help="place all reward models on the same device/placement group")
    parser.add_argument("--colocate_all_models", action="store_true", default=False, help="share GPUs across all models including vLLM engines")
    parser.add_argument("--async_train", action="store_true", default=False, help="Enable async training")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride", type=int, default=1,
        help="Number of heads per ring-attention step; larger is faster but uses more memory",
    )

    # -----------------------------------------------------------------------
    # Infrastructure: vLLM
    # -----------------------------------------------------------------------
    parser.add_argument("--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="tensor parallel size of vLLM Engine for multi-GPU inference")
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95, help="vLLM gpu_memory_utilization")
    parser.add_argument("--vllm_enable_sleep", action="store_true", default=False, help="Enable sleep mode for vLLM when using --colocate_all_models")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")

    # -----------------------------------------------------------------------
    # Infrastructure: DeepSpeed
    # -----------------------------------------------------------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--deepspeed_enable_sleep", action="store_true", default=False, help="Enable sleep mode for deepspeed when using --colocate_all_models")
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")
    parser.add_argument("--packing_samples", action="store_true", default=False, help="Pack samples using Flash Attention 2")
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    # -----------------------------------------------------------------------
    # Models: pretrained checkpoints and architecture flags
    # -----------------------------------------------------------------------
    parser.add_argument("--pretrain", type=str, default=None, help="Actor HF model name or path")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="Critic HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="Reward model HF name or path (comma-separated for multiple)")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # EMA
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--ema_beta", type=float, default=0.992, help="EMA beta coefficient")

    # -----------------------------------------------------------------------
    # Training schedule: LRs, schedulers
    # -----------------------------------------------------------------------
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--critic_lr_head", type=float, default=None, help="LR for critic classifier head; if None, uses critic_learning_rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--critic_lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--actor_lr_warmup_ratio", type=float, default=None, help="Optionally set actor warmup separately")
    parser.add_argument("--critic_lr_warmup_ratio", type=float, default=None, help="Optionally set critic warmup separately")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")

    # -----------------------------------------------------------------------
    # Batch sizes
    # -----------------------------------------------------------------------
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024, help="Number of prompts per rollout")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="Per-GPU training batch size")
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8, help="Per-GPU rollout batch size")
    parser.add_argument("--micro_reward_batch_size", type=int, default=8, help="Per-GPU reward model batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Global eval batch size")
    parser.add_argument("--eval_down_batch_size", type=int, default=128, help="Batch size for downstream evaluation")
    parser.add_argument("--vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples")

    # -----------------------------------------------------------------------
    # Sequence / generation parameters
    # -----------------------------------------------------------------------
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--context_max_len", type=int, default=1024, help="Context window length used per generation block")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Tokens generated per block (G)")
    parser.add_argument("--stride", type=int, default=None, help="Stride between block windows. Defaults to generate_max_len.")
    parser.add_argument("--n_samples_per_prompt", type=int, default=1, help="Number of generated responses per prompt")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False, help="Enable reproducible behavior during distributed training")

    # -----------------------------------------------------------------------
    # RL / loss coefficients
    # -----------------------------------------------------------------------
    parser.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"])
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Advantage estimation method",
    )
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None, help="Asymmetric PPO-clip (low, high)")
    parser.add_argument("--dual_clip", type=float, default=None, help="Dual-clip PPO lower bound")
    parser.add_argument("--value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1, help="GAE lambda")
    parser.add_argument("--gamma", type=float, default=1, help="GAE gamma")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable reward normalization")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="Use KL as a training loss (GRPO-style)")
    parser.add_argument("--kl_estimator", type=str, default="k1", choices=["k1", "k2", "k3"], help="KL divergence estimator")
    parser.add_argument("--kl_loss_coef", type=float, default=None)
    parser.add_argument("--kl_loss_decay_steps", type=int, default=None)
    parser.add_argument("--rl_loss_coef", type=float, default=None)
    parser.add_argument("--rl_loss_warmup_start", type=int, default=None)
    parser.add_argument("--rl_loss_warmup_steps", type=int, default=None)
    parser.add_argument("--ce_loss_coef", type=float, default=0.01, help="CE loss coefficient for actor")
    parser.add_argument("--ce_loss_decay_steps", type=nullable_int, default=None)
    parser.add_argument("--entropy_loss_coef", type=float, default=None, help="Entropy loss coefficient; 0 enables logging only")
    parser.add_argument("--no_advantage_std_norm", action="store_true", default=False, help="Disable std normalization of advantages")
    parser.add_argument("--overlong_buffer_len", type=float, default=None, help="Buffer length before overlong penalty kicks in")
    parser.add_argument("--overlong_penalty_factor", type=float, default=1, help="Overlong penalty factor")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE router balancing loss coefficient")
    parser.add_argument("--use_whitening", action="store_true", default=False, help="Whiten actor reward embeddings")
    parser.add_argument("--dynamic_filtering", action="store_true", default=False, help="Enable dynamic filtering")
    parser.add_argument("--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float, help="Reward range for dynamic filtering")

    # Reward composition
    parser.add_argument("--alignment_rew_coef", type=float, default=1.0, help="Weight for embedding alignment reward")
    parser.add_argument("--diversity_rew_coef", type=float, default=1.0, help="Weight for diversity reward")

    # -----------------------------------------------------------------------
    # Critic losses and architecture
    # -----------------------------------------------------------------------
    parser.add_argument("--critic_sequence_level", type=str, default="token", choices=["token", "last_token", "mean_pooling", "concat"], help="Aggregation level for classifier logits")
    parser.add_argument("--hidden_state_method", type=str, default="last_only", help="Which transformer layers to use for critic embeddings (last_only, mean, concat, layer_N, ...)")
    parser.add_argument("--embed_method", type=str, default="concat", help="Embedding method for generator")
    parser.add_argument("--classifier_sequence_selection", type=str, default="first", choices=["first", "all", "closest", "only_different"], help="Which generated sample to contrast against GT in classifier loss")
    parser.add_argument("--document_masking", action="store_true", default=False, help="Restrict attention to same-document tokens")
    parser.add_argument("--qa_masking", action="store_true", default=False, help="Mask question tokens from loss")

    # Critic loss coefficient
    parser.add_argument("--critic_classifier_loss_coef", type=float, default=1.0, help="Classifier log loss coefficient")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--prompt_data_probs", type=str, default=None, help="Sampling probabilities for datasets (comma-separated)")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Evaluation dataset HF name or path")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="Dataset field for model input")
    parser.add_argument("--output_key", type=str, default=None, help="Dataset field for model output")
    parser.add_argument("--label_key", type=str, default=None, help="Dataset field for ground-truth label")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template")
    parser.add_argument("--no_chat_template", action="store_true", default=False, help="Disable chat template even if available")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max training samples (-1 = all)")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    parser.add_argument("--eval_steps", type=int, default=-1, help="Run online eval every N steps (-1 = disabled)")
    parser.add_argument("--eval_down_steps", type=int, default=-1, help="Run downstream eval every N steps (-1 = disabled)")
    parser.add_argument("--eval_generate_max_len", type=int, default=1024, help="Max tokens to generate during evaluation")
    parser.add_argument("--eval_temperature", type=float, default=0.6, help="Sampling temperature during evaluation")
    parser.add_argument("--eval_temperature_down", type=float, default=0.6, help="Sampling temperature for downstream evaluation")
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=4, help="Samples per prompt during evaluation")
    parser.add_argument("--eval_n_samples_per_prompt_down", type=int, default=16, help="Samples per prompt for downstream evaluation")
    parser.add_argument("--eval_max_samples", type=int, default=1e8, help="Max prompts for online evaluation")
    parser.add_argument("--eval_down_max_samples", type=int, default=1e8, help="Max prompts for downstream evaluation")

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------
    parser.add_argument("--save_path", type=str, default="./ckpt", help="Directory to save model checkpoints")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every N steps (-1 = disabled)")
    parser.add_argument("--save_log_scale_count", type=int, default=-1, help="Number of log-scale checkpoints (-1 = disabled)")
    parser.add_argument("--save_even_count", type=int, default=10, help="Number of evenly spaced checkpoints when other save triggers are disabled")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_actor_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_critic_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False, help="Use DeepSpeed universal checkpoint")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    parser.add_argument("--use_wandb", type=str, default=None, help="WandB API key")
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_debug")
    parser.add_argument("--wandb_run_name", type=str, default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"))
    parser.add_argument("--slurm_job", type=str, default=None, help="Slurm job identifier for wandb tracking ({slurm_job_id}_{slurm_task_id})")
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--log_gradients", action="store_true", default=False)

    # -----------------------------------------------------------------------
    # Misc / debug
    # -----------------------------------------------------------------------
    parser.add_argument("--debug", action="store_true", default=False, help="Enable verbose debug output")
    parser.add_argument("--use_ms", action="store_true", default=False, help="Use ModelScope hub")

    args = parser.parse_args()

    # Validate arguments
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ring_attn_size > 1:
        if not args.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.packing_samples = True

    if args.use_dynamic_batch:
        if not args.packing_samples:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.packing_samples = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.colocate_all_models and args.async_train:
        print("[Warning] Using --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if args.async_train:
        assert not args.vllm_enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size

    if args.dynamic_filtering:
        assert (
            args.dynamic_filtering_reward_range[0] < args.dynamic_filtering_reward_range[1]
        ), "reward_clip_range[0] must be less than reward_clip_range[1]"
        assert (
            args.remote_rm_url or args.agent_func_path
        ), "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        assert (
            args.n_samples_per_prompt > 1
        ), "n_samples_per_prompt must be greater than 1 when using dynamic filtering"

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    assert (
        args.train_batch_size == args.n_samples_per_prompt * args.rollout_batch_size
    ), f"ED approach only works for train_batch_size == n_samples_per_prompt * rollout_batch_size but found train_batch_size={args.train_batch_size}, n_samples_per_prompt={args.n_samples_per_prompt} and rollout_batch_size={args.rollout_batch_size}"

    assert (
        args.micro_rollout_batch_size % args.n_samples_per_prompt == 0
    ), f"Micro rollout batch size {args.micro_rollout_batch_size} must be divided by n_samples_per_prompt {args.n_samples_per_prompt}"


    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    if args.packing_samples:
        print("[Info] Disabling --packing_samples when --use_carles_mask is enabled.")
        args.packing_samples = False

    assert not (
        args.remote_rm_url and args.reward_pretrain
    ), "remote_rm_url and reward_pretrain are mutually exclusive"

    train(args)
