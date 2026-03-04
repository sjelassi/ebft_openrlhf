#!/bin/bash

python scripts/evaluate_reward_ce.py \
    --actor_checkpoint Qwen/Qwen2.5-1.5B \
    --critic_checkpoint Qwen/Qwen2.5-1.5B \
    --eval_dataset sjelassi/opencode-instruct_100k_200tok \
    --eval_split test \
    --input_key question \
    --label_key answer \
    --eval_batch_size 1 \
    --eval_max_samples 1024 \
    --n_samples_per_prompt 4 \
    --prompt_max_len 1024 \
    --generate_max_len 8 \
    --context_max_len 8 \
    --stride 8 \
    --temperature 1.0 \
    --embed_method last_token \
    --hidden_state_method concat \
    --output_file results/eval_reward_ce_results.json \
    --use_whitening

