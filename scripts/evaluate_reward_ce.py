#!/usr/bin/env python3
"""
Simplified standalone evaluation script using inference_loss module.

This script requires minimal arguments compared to the full Ray-based evaluation:
- Model checkpoints
- Dataset
- Basic evaluation parameters
"""

import argparse
import json
import math
import os
import random
import time
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.utils.logging_utils import init_logger
from inference_loss import StridedActorModel, StridedCriticModel, EvaluationMetrics

logger = init_logger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified standalone evaluation script")

    # Model checkpoints (only essential args)
    parser.add_argument("--actor_checkpoint", type=str, required=True, help="Path to actor checkpoint")
    parser.add_argument("--critic_checkpoint", type=str, required=True, help="Path to critic checkpoint")

    # Dataset
    parser.add_argument("--eval_dataset", type=str, required=True, help="Evaluation dataset name")
    parser.add_argument("--eval_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--eval_max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--input_key", type=str, default="question", help="Key for input in dataset")
    parser.add_argument("--label_key", type=str, default="answer", help="Key for label/answer in dataset")

    # Evaluation parameters
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")

    # Sequence parameters
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max prompt length")
    parser.add_argument("--generate_max_len", type=int, default=16, help="Max generation length")
    parser.add_argument("--context_max_len", type=int, default=16, help="Context length")
    parser.add_argument("--stride", type=int, default=16, help="Stride for block processing")

    # Reward settings
    parser.add_argument("--alignment_rew_coef", type=float, default=1.0,
                       help="Coefficient for alignment reward")
    parser.add_argument("--diversity_rew_coef", type=float, default=1.0,
                       help="Coefficient for diversity penalty")

    # Embedding settings
    parser.add_argument("--embed_method", type=str, default="concat",
                       choices=["mean_pooling", "last_token", "concat", "token"],
                       help="Embedding method (used in ebft_trainer.py evaluation)")
    parser.add_argument("--hidden_state_method", type=str, default="last_only",
                       help="Hidden state extraction method")
    parser.add_argument("--use_whitening", action="store_true", help="Use whitening on embeddings")

    # Model settings
    parser.add_argument("--document_masking", action="store_true", help="Use document masking")
    parser.add_argument("--qa_masking", action="store_true", help="Use QA masking")

    # Device settings
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                       help="Data type for models")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Output
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Set torch dtype
    if args.dtype == "float32":
        args.torch_dtype = torch.float32
    elif args.dtype == "float16":
        args.torch_dtype = torch.float16
    else:
        args.torch_dtype = torch.bfloat16

    return args


def pack_to_fixed_chunks(
    tokenizer,
    qa_pairs: List[Tuple[str, str]],
    seq_len: int = 1024,
    add_eos_between: bool = True,
    pad_last: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Concatenate-tokenize-pack QA pairs into fixed-length chunks.
    Produces a list of (seq_len,) tensors. No padding except the final chunk.

    Args:
        tokenizer: Tokenizer to use
        qa_pairs: List of (question, answer) tuples
        seq_len: Fixed sequence length for each chunk
        add_eos_between: Add EOS token between QA pairs
        pad_last: Pad the last chunk if needed

    Returns:
        chunks:             List[Tensor] with shape (seq_len,) containing token IDs
        doc_id_chunks:      List[Tensor] with shape (seq_len,),
                            where doc_id_chunks[k][t] = QA pair index of token t
                            (0,1,2,...) and -1 for padding positions.
        answer_mask_chunks: List[Tensor] with shape (seq_len,),
                            1 for answer tokens, 0 otherwise (question/separators/padding).
    """
    # Separator (EOS) between QA pairs
    sep = []
    if add_eos_between:
        if tokenizer.eos_token_id is not None:
            sep = [tokenizer.eos_token_id]
        else:
            raise ValueError("tokenizer must have eos_token_id to use as separator")

    # Pad id for final chunk
    pad_id = None
    if pad_last:
        if tokenizer.pad_token_id is not None:
            pad_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id
        else:
            raise ValueError("Need pad_token_id or eos_token_id for padding")

    # Tokenize questions and answers separately
    prompts = [p for p, _ in qa_pairs]
    answers = [a for _, a in qa_pairs]

    prompt_tok = tokenizer(
        prompts,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    answer_tok = tokenizer(
        answers,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    # Build per-example token ids and corresponding answer masks
    ids_per_ex: List[List[int]] = []
    ans_mask_per_ex: List[List[int]] = []
    for p_ids, a_ids in zip(prompt_tok, answer_tok):
        ids = list(p_ids) + list(a_ids)
        mask = [0] * len(p_ids) + [1] * len(a_ids)  # 0 for question, 1 for answer
        ids_per_ex.append(ids)
        ans_mask_per_ex.append(mask)

    # Flatten with separators AND parallel doc_ids and answer masks
    flat_ids: List[int] = []
    flat_doc_ids: List[int] = []
    flat_ans_mask: List[int] = []

    for doc_idx, ids in enumerate(ids_per_ex):
        # actual tokens
        flat_ids.extend(ids)
        flat_doc_ids.extend([doc_idx] * len(ids))
        flat_ans_mask.extend(ans_mask_per_ex[doc_idx])

        # separator tokens (not part of Q/A; mask=0)
        if sep and doc_idx != len(ids_per_ex) - 1:
            flat_ids.extend(sep)
            flat_doc_ids.extend([doc_idx] * len(sep))
            flat_ans_mask.extend([0] * len(sep))

    if not flat_ids:
        return [], [], []

    stream = torch.tensor(flat_ids, dtype=torch.long)
    stream_doc = torch.tensor(flat_doc_ids, dtype=torch.long)
    stream_mask = torch.tensor(flat_ans_mask, dtype=torch.long)
    L = seq_len
    n_full = stream.numel() // L

    chunks: List[torch.Tensor] = []
    doc_id_chunks: List[torch.Tensor] = []
    answer_mask_chunks: List[torch.Tensor] = []

    if n_full:
        chunks.extend(stream[: n_full * L].view(n_full, L).unbind(0))
        doc_id_chunks.extend(stream_doc[: n_full * L].view(n_full, L).unbind(0))
        answer_mask_chunks.extend(stream_mask[: n_full * L].view(n_full, L).unbind(0))

    # remainder
    rem = stream[n_full * L :]
    rem_doc = stream_doc[n_full * L :]
    rem_mask = stream_mask[n_full * L :]

    if rem.numel():
        if rem.numel() < L and pad_last:
            # pad tokens
            rem = F.pad(rem, (0, L - rem.numel()), value=pad_id)
            # mark padding with doc_id = -1 and mask=0
            rem_doc = F.pad(rem_doc, (0, L - rem_doc.numel()), value=-1)
            rem_mask = F.pad(rem_mask, (0, L - rem_mask.numel()), value=0)
        chunks.append(rem)
        doc_id_chunks.append(rem_doc)
        answer_mask_chunks.append(rem_mask)

    return chunks, doc_id_chunks, answer_mask_chunks


def load_eval_dataset(args):
    """Load evaluation dataset."""
    # Load dataset
    if args.eval_dataset == "openai/gsm8k":
        eval_data = load_dataset(args.eval_dataset, name='main')[args.eval_split]
    else:
        eval_data = load_dataset(args.eval_dataset)[args.eval_split]


    return eval_data


def preprocess_data(sample, input_key, label_key):
    """Simple preprocessing function."""
    if input_key and label_key:
        prompt = sample[input_key]
        label = sample[label_key]
    else:
        # Fallback to default keys
        prompt = sample.get("question", sample.get("prompt", ""))
        label = sample.get("answer", sample.get("completion", ""))

    return prompt, label


def collate_packed_fn(batch):
    """
    Collate function for packed dataset.
    Each batch item is a tuple of (input_ids, doc_ids, qa_masks).

    Returns:
        Dict with batched tensors
    """
    # Unpack the batch
    sequences, doc_ids, qa_masks = zip(*batch)

    # Stack into batches
    return {
        "input_ids": torch.stack(sequences, dim=0),
        "doc_ids": torch.stack(doc_ids, dim=0),
        "qa_masks": torch.stack(qa_masks, dim=0),
    }


def evaluate(args):
    """Main evaluation function."""
    start_time = time.time()
    logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("="*80)
    logger.info("Simplified Standalone Evaluation")
    logger.info("="*80)
    logger.info(f"Actor checkpoint: {args.actor_checkpoint}")
    logger.info(f"Critic checkpoint: {args.critic_checkpoint}")
    logger.info(f"Evaluation dataset: {args.eval_dataset} ({args.eval_split})")
    logger.info(f"N samples per prompt: {args.n_samples_per_prompt}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info("="*80)

    # Initialize models
    logger.info("Loading actor model...")
    actor_model = StridedActorModel(
        model_name_or_path=args.actor_checkpoint,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    actor_model.model.eval()  # Set to eval mode

    logger.info("Loading critic model...")
    critic_model = StridedCriticModel(
        model_name_or_path=args.critic_checkpoint,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    critic_model.model.eval()  # Set to eval mode

    # Initialize metrics computer
    metrics_computer = EvaluationMetrics(
        alignment_rew_coef=args.alignment_rew_coef,
        diversity_rew_coef=args.diversity_rew_coef,
        embed_method=args.embed_method,
        use_whitening=args.use_whitening,  # Enable whitening in evaluation_metrics (happens after pooling)
    )

    # Load dataset
    logger.info("Loading evaluation dataset...")
    eval_data = load_eval_dataset(args)

    # Prepare QA pairs
    logger.info("Processing QA pairs...")
    qa_pairs = []
    for sample in eval_data:
        prompt, label = preprocess_data(sample, args.input_key, args.label_key)
        qa_pairs.append((prompt, label))

    # Pack QA pairs into fixed-length chunks
    logger.info(f"Packing {len(qa_pairs)} QA pairs into chunks of length {args.prompt_max_len}...")
    packed_sequences, packed_doc_ids, packed_qa_masks = pack_to_fixed_chunks(
        tokenizer=actor_model.tokenizer,
        qa_pairs=qa_pairs,
        seq_len=args.prompt_max_len,
        add_eos_between=True,
        pad_last=True,
    )

    logger.info(f"Total QA pairs: {len(qa_pairs)}")
    logger.info(f"Total packed chunks before limiting: {len(packed_sequences)}")

    # Apply max_samples AFTER packing to limit number of packed sequences
    if args.eval_max_samples > 0 and len(packed_sequences) > args.eval_max_samples:
        logger.info(f"Limiting to first {args.eval_max_samples} packed sequences (from {len(packed_sequences)})")
        packed_sequences = packed_sequences[:args.eval_max_samples]
        packed_doc_ids = packed_doc_ids[:args.eval_max_samples]
        packed_qa_masks = packed_qa_masks[:args.eval_max_samples]

    logger.info(f"Total packed chunks after limiting: {len(packed_sequences)}")

    # Create dataset from packed chunks
    class PackedDataset(torch.utils.data.Dataset):
        def __init__(self, sequences, doc_ids, qa_masks):
            self.sequences = sequences
            self.doc_ids = doc_ids
            self.qa_masks = qa_masks

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.doc_ids[idx], self.qa_masks[idx]

    packed_dataset = PackedDataset(packed_sequences, packed_doc_ids, packed_qa_masks)

    # Create dataloader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_packed_fn,
    )

    # Calculate num_blocks
    prompt_length = args.prompt_max_len
    num_blocks = (prompt_length - args.generate_max_len - args.context_max_len) // args.stride + 1

    # Accumulate metrics
    all_gen_embeddings = []
    all_gt_embeddings = []
    all_ppl_nll_sums = []
    all_ppl_token_counts = []
    all_answer_nll_sums = []
    all_answer_token_counts = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(args.device)
            doc_ids = batch["doc_ids"].to(args.device)
            qa_masks = batch["qa_masks"].to(args.device)

            # Use regular causal attention to compute log probs on the original packed sequences
            # This computes perplexity on the ground truth answer tokens only (using qa_masks)

            # Create attention mask (all ones for causal attention)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=args.device)

            # Create position IDs (standard sequential positions)
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=args.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            # Forward pass through model with regular causal attention
            outputs = actor_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Get logits and compute log probs for next-token prediction
            logits = outputs.logits[:, :-1, :]  # Shape: (batch, seq_len-1, vocab)
            targets = input_ids[:, 1:]  # Shape: (batch, seq_len-1)

            # Compute log probabilities
            log_probs_all = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs_all,
                dim=-1,
                index=targets.unsqueeze(-1)
            ).squeeze(-1)  # Shape: (batch, seq_len-1)

            # Free memory immediately after gathering
            del logits, log_probs_all

            # Full loss mask: all valid next-token positions (matches trainer CE evaluation)
            full_loss_mask = torch.ones_like(targets, dtype=torch.float)
            # Answer-only loss mask: restrict to answer tokens (qa_masks shifted by 1 for next-token prediction)
            answer_loss_mask = qa_masks[:, 1:].float()

            # Accumulate NLL sums and token counts for both masks
            all_ppl_nll_sums.append((-token_log_probs * full_loss_mask).sum().cpu())
            all_ppl_token_counts.append(full_loss_mask.sum().cpu())
            all_answer_nll_sums.append((-token_log_probs * answer_loss_mask).sum().cpu())
            all_answer_token_counts.append(answer_loss_mask.sum().cpu())

            # Free CE computation memory
            del outputs, token_log_probs, full_loss_mask, answer_loss_mask, attention_mask, position_ids
            torch.cuda.empty_cache()

            # ========================================
            # STEP 2: Generate samples and compute rewards
            # ========================================
            gen_output = actor_model.generate_samples(
                prompts=input_ids,
                doc_ids=doc_ids,
                qa_masks=qa_masks,
                n_samples_per_prompt=args.n_samples_per_prompt,
                generate_max_len=args.generate_max_len,
                context_length=args.context_max_len,
                stride=args.stride,
                temperature=args.temperature,
                top_p=args.top_p,
                document_masking=args.document_masking,
                qa_masking=args.qa_masking,
                group_by_prompt=True,
            )

            # Ensure all tensors from gen_output are on the correct device
            gen_output.full_sequences = gen_output.full_sequences.to(args.device)
            gen_output.doc_ids = gen_output.doc_ids.to(args.device)
            gen_output.qa_masks = gen_output.qa_masks.to(args.device)

            # Extract embeddings from critic for reward computation
            gen_embedding, gt_embedding = critic_model.extract_gt_and_gen_embeddings(
                sequences=gen_output.full_sequences,
                prompt_length=prompt_length,
                context_length=args.context_max_len,
                generate_max_len=args.generate_max_len,
                stride=args.stride,
                num_blocks=num_blocks,
                hidden_state_method=args.hidden_state_method,
                doc_ids=gen_output.doc_ids,
                document_masking=args.document_masking,
                qa_masks=gen_output.qa_masks,
                qa_masking=args.qa_masking,
            )

            all_gen_embeddings.append(gen_embedding.cpu())
            all_gt_embeddings.append(gt_embedding.cpu())


    # Concatenate all results
    logger.info("Computing final metrics...")
    gen_embeddings = torch.cat(all_gen_embeddings, dim=0)
    gt_embeddings = torch.cat(all_gt_embeddings, dim=0)

    # NOTE: Whitening now happens INSIDE evaluation_metrics.py compute_rewards() AFTER pooling
    # Verify grouping is correct
    total_sequences = gen_embeddings.size(0)
    assert total_sequences % args.n_samples_per_prompt == 0, (
        f"Total sequences {total_sequences} must be divisible by n_samples_per_prompt {args.n_samples_per_prompt}"
    )
    num_prompts_total = total_sequences // args.n_samples_per_prompt
    logger.info(f"Total sequences: {total_sequences}, Total prompts: {num_prompts_total}, Samples per prompt: {args.n_samples_per_prompt}")

    # Compute perplexity from accumulated NLL sums
    total_nll = sum(all_ppl_nll_sums)
    total_tokens = sum(all_ppl_token_counts)
    ce_loss = (total_nll / total_tokens).item()
    perplexity = math.exp(ce_loss)

    total_answer_nll = sum(all_answer_nll_sums)
    total_answer_tokens = sum(all_answer_token_counts)
    answer_ce_loss = (total_answer_nll / total_answer_tokens).item()
    answer_perplexity = math.exp(answer_ce_loss)

    # Compute reward metrics (pass empty log_probs since we computed perplexity separately)
    reward_dict = metrics_computer.compute_rewards(
        gen_embedding=gen_embeddings,
        gt_embedding=gt_embeddings,
        n_samples=args.n_samples_per_prompt,
    )

    # Debug: Print per-sample reward statistics
    rewards = reward_dict["rewards"]
    gt_rewards = reward_dict["gt_rewards"]
    diversity_rewards = reward_dict["diversity_rewards"]
    effective_rewards = reward_dict["effective_rewards"]

    metrics = metrics_computer.compute_pass_metrics(reward_dict, args.n_samples_per_prompt)

    # Add perplexity metrics
    metrics.update({
        "full_ce_loss": ce_loss,
        "full_perplexity": perplexity,
        "answer_ce_loss": answer_ce_loss,
        "answer_perplexity": answer_perplexity,
    })

    end_time = time.time()
    duration = end_time - start_time
    time_str = str(timedelta(seconds=duration)).split(".")[0]
    logger.info(f"✨ Evaluation completed in {time_str}")

    return metrics


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Run evaluation
    results = evaluate(args)

    # Add metadata to results
    results_with_meta = {
        "metadata": {
            "actor_checkpoint": args.actor_checkpoint,
            "critic_checkpoint": args.critic_checkpoint,
            "eval_dataset": args.eval_dataset,
            "eval_split": args.eval_split,
            "n_samples_per_prompt": args.n_samples_per_prompt,
            "temperature": args.temperature,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": results,
    }

    # Save results
    output_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*80}")

    # Print results in organized format
    logger.info("\n📊 REWARD METRICS (Pass@1):")
    logger.info(f"  reward_pass1_gt:                 {results['reward_pass1_gt']:.6f}")
    logger.info(f"  reward_pass1_diversity:          {results['reward_pass1_diversity']:.6f}")
    logger.info(f"  reward_pass1_effective:          {results['reward_pass1_effective']:.6f}")
    logger.info(f"  reward_pass1_effective_alpha_1:  {results['reward_pass1_effective_alpha_1']:.6f}")
    logger.info(f"  reward_pass1 (full):             {results['reward_pass1']:.6f}")

    logger.info(f"\n📊 REWARD METRICS (Pass@{args.n_samples_per_prompt}):")
    logger.info(f"  reward_passk_gt:                 {results['reward_passk_gt']:.6f}")
    logger.info(f"  reward_passk_diversity:          {results['reward_passk_diversity']:.6f}")
    logger.info(f"  reward_passk_effective:          {results['reward_passk_effective']:.6f}")
    logger.info(f"  reward_passk_effective_alpha_1:  {results['reward_passk_effective_alpha_1']:.6f}")
    logger.info(f"  reward_passk (full):             {results['reward_passk']:.6f}")

    logger.info("\n📈 PERPLEXITY METRICS:")
    logger.info(f"  full_ce_loss:                    {results['full_ce_loss']:.6f}")
    logger.info(f"  full_perplexity:                 {results['full_perplexity']:.6f}")
    logger.info(f"  answer_ce_loss:                  {results['answer_ce_loss']:.6f}")
    logger.info(f"  answer_perplexity:               {results['answer_perplexity']:.6f}")

    logger.info(f"\n✅ Results saved to: {output_path}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
