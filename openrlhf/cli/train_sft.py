import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from datatrove.utils.dataset import DatatroveFolderDataset
from openrlhf.datasets import SFTDataset, DatatroveSFTDataset, PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import OriginalActor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = OriginalActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    if args.dataset.count("/") > 1:
        train_data = DatatroveFolderDataset(
                folder_path=args.dataset,
                seq_len=args.max_len,
                token_size=(2 if tokenizer.vocab_size < 65535 else 4),
                shuffle=True,
                seed=args.seed,
                return_positions=False,      # we don’t need them
            )
        train_dataset = DatatroveSFTDataset(
                    train_data,
                    tokenizer,
                    args.max_len,
                    args.max_samples,
                    strategy,
                    pretrain_mode=args.pretrain_mode,
                )
                   # args.prompt_max_len,
                    #     args.generate_max_len,
                

    else:
        train_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=args.dataset_split,
        )
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        train_dataset = SFTDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            pretrain_mode=args.pretrain_mode,
            input_template=args.input_template,
            multiturn=args.multiturn,
        )
    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )


    eval_dataloader = None
    eval_perplexity_dataloader = None
    if getattr(args, "eval_dataset", None):


        if args.dataset.count("/") > 1:
            eval_data = DatatroveFolderDataset(
                    folder_path=args.eval_dataset,
                    seq_len=args.max_len,
                    token_size=(2 if tokenizer.vocab_size < 65535 else 4),
                    shuffle=True,
                    seed=args.seed,
                    return_positions=False,      # we don’t need them
                )
            eval_perplexity_dataset = DatatroveSFTDataset(
                        eval_data,
                        tokenizer,
                        args.max_len,
                        args.eval_max_samples,
                        strategy,
                        pretrain_mode=args.pretrain_mode,
                    )

                    # args.prompt_max_len,
                    #     args.generate_max_len,
        else:
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=args.eval_split,
            )
            
            eval_data = eval_data.select(range(min(args.eval_max_samples, len(eval_data))))
            eval_perplexity_dataset = SFTDataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                pretrain_mode=args.pretrain_mode,
                input_template=args.input_template,
                multiturn=args.multiturn,
            )
            eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, args.eval_batch_size, True, False)#1, True, False)

        eval_perplexity_dataloader = strategy.setup_dataloader(
            eval_perplexity_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_perplexity_dataset.collate_fn,
        )



    # scheduler
    
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    # print(f"LEN: {len(train_dataset)}; BS: {args.train_batch_size}; NUM UPDATE: {num_update_steps_per_epoch}",flush=True)
    # wqqw
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_perplexity_dataloader=eval_perplexity_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # vLLM configuration
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM engine")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7,
                       help="GPU memory utilization for vLLM engine")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate during evaluation")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for sampling during evaluation")
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=1,
                       help="Number of samples to generate per prompt for pass@k calculation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter for generation")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum number of tokens to generate")
    
    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--eval_max_samples", type=int, default=1e8, help="Max number of eval samples")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_debug")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.multiturn:
        assert args.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)