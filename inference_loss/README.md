# Inference Loss - Simplified Evaluation Module

This module provides standalone classes for evaluating models without Ray, DeepSpeed, or training infrastructure.

## Overview

The `inference_loss` module extracts the core evaluation logic from `ebft_trainer.py` and provides simple classes that:

1. **StridedActorModel** - Handles generation with strided attention masks
2. **StridedCriticModel** - Handles embedding extraction with strided blocks
3. **EvaluationMetrics** - Computes rewards and perplexity

## Usage

### Simple Evaluation Script

Use `scripts/evaluate_reward_ce.py` for streamlined evaluation:

```bash
python scripts/evaluate_reward_ce.py \
    --actor_checkpoint Qwen/Qwen2.5-1.5B \
    --critic_checkpoint Qwen/Qwen2.5-1.5B \
    --eval_dataset sjelassi/opencode-instruct_100k_200tok \
    --eval_split test \
    --eval_max_samples 100 \
    --n_samples_per_prompt 4 \
    --temperature 0.6 \
    --embed_method last_token \
    --use_whitening \
    --output_file results/eval_results.json
```

### Using the Module Directly

```python
from inference_loss import StridedActorModel, StridedCriticModel, EvaluationMetrics
import torch

# Load models
actor = StridedActorModel("path/to/actor", device="cuda")
critic = StridedCriticModel("path/to/critic", device="cuda")

# Initialize metrics
metrics = EvaluationMetrics(
    alignment_rew_coef=1.0,
    diversity_rew_coef=1.0,
)

# Generate samples
gen_output = actor.generate_samples(
    prompts=input_ids,
    doc_ids=doc_ids,
    qa_masks=qa_masks,
    n_samples_per_prompt=4,
    generate_max_len=128,
    temperature=0.6,
)

# Extract embeddings
gen_emb, gt_emb = critic.extract_gt_and_gen_embeddings(
    sequences=gen_output.full_sequences,
    prompt_length=1024,
    context_length=16,
    generate_max_len=128,
    stride=16,
    num_blocks=60,
    hidden_state_method="concat",
    embed_method="last_token",
    use_whitening=True,
)

# Compute log probs
log_probs = actor.compute_logprobs_strided(
    sequences=gen_output.full_sequences,
    prompt_length=1024,
    context_length=16,
    generate_max_len=128,
    stride=16,
    num_blocks=60,
    temperature=0.6,  # Should match generation temperature
)

# Get all metrics
results = metrics.compute_all_metrics(
    gen_embedding=gen_emb,
    gt_embedding=gt_emb,
    log_probs=log_probs,
    n_samples=4,
)

print(results)
# {
#   'reward_pass1_gt': 0.123,
#   'reward_pass1_diversity': 0.234,
#   'reward_pass1_effective_alpha_1': 0.006,  # gt - div/2
#   'reward_passk_gt': 0.345,
#   'reward_passk_diversity': 0.456,
#   'reward_passk_effective_alpha_1': 0.117,
#   'full_ce_loss': 1.234,
#   'full_perplexity': 3.456,
# }
```

## Architecture

### StridedActorModel

Handles generation and perplexity computation:
- Loads HuggingFace CausalLM models
- Generates multiple samples per prompt
- Computes log probabilities with strided attention masks
- Uses `build_strided_attention_mask_and_positions` from `openrlhf.models.utils`

### StridedCriticModel

Handles embedding extraction:
- Loads HuggingFace models (AutoModel for embeddings)
- Extracts hidden states with strided blocks
- Supports multiple embedding methods (last_token, mean_pooling, concat, token)
- Applies whitening if requested
- Splits into ground truth and generated embeddings

### EvaluationMetrics

Computes all metrics:
- Alignment rewards (ground truth similarity)
- Diversity rewards (inter-sample diversity)
- Combined rewards (effective, effective_alpha_1)
- Pass@1 metrics (average over all samples)
- Pass@k metrics (oracle/best of k per prompt)
- Perplexity and cross-entropy loss

## Metrics Computed

The module computes the same metrics as `ebft_trainer.py` evaluate():

### Pass@1 Metrics (average over all samples)
- `reward_pass1_gt` - Ground truth alignment reward
- `reward_pass1_diversity` - Diversity penalty
- `reward_pass1_effective` - Combined reward with full diversity penalty
- `reward_pass1_effective_alpha_1` - Combined reward: `gt - div/2`
- `reward_pass1` - Full combined reward used in training

### Pass@k Metrics (best of k samples per prompt)
- `reward_passk_gt` - Ground truth alignment (best)
- `reward_passk_diversity` - Diversity penalty (best)
- `reward_passk_effective` - Combined reward (best)
- `reward_passk_effective_alpha_1` - Combined: `gt - div/2` (best)
- `reward_passk` - Full combined reward (best)

### Perplexity Metrics
- `full_ce_loss` - Cross-entropy loss
- `full_perplexity` - Perplexity
- `answer_ce_loss` - Cross-entropy loss (restricted to answer tokens)
- `answer_perplexity` - Perplexity (restricted to answer tokens)

## Notes

## Requirements

- PyTorch
- Transformers
- Datasets
- openrlhf (for utility functions like `build_strided_attention_mask_and_positions`)

## Example Output

```json
{
  "metadata": {
    "actor_checkpoint": "Qwen/Qwen2.5-1.5B",
    "critic_checkpoint": "Qwen/Qwen2.5-1.5B",
    "eval_dataset": "sjelassi/opencode-instruct_100k_200tok",
    "eval_split": "test",
    "n_samples_per_prompt": 4,
    "temperature": 0.6,
    "timestamp": "2026-01-31 12:00:00"
  },
  "metrics": {
    "reward_pass1_gt": 0.123456,
    "reward_pass1_diversity": 0.234567,
    "reward_pass1_effective": 0.345678,
    "reward_pass1_effective_alpha_1": 0.006173,
    "reward_pass1": -0.111111,
    "reward_passk_gt": 0.654321,
    "reward_passk_diversity": 0.543210,
    "reward_passk_effective": 0.432109,
    "reward_passk_effective_alpha_1": 0.382716,
    "reward_passk": 0.111111,
    "full_ce_loss": 1.234567,
    "full_perplexity": 3.456789
  }
}
```
