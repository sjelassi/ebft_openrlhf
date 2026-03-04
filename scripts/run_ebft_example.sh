#!/bin/bash

# launch the master node of ray in container
conda activate openrlhf
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT_CANDIDATES=(6000 6001 6002 6003)
# check if master port candidate is already being used; if not, use it as master port
for MPC in ${MASTER_PORT_CANDIDATES[@]}; do
    NUM_LISTENING_PROCESSES=$(lsof -Pi :${MPC} -sTCP:LISTEN | wc -l)
    if test $NUM_LISTENING_PROCESSES -eq 0; then
        MASTER_PORT=${MPC}
        export MASTER_PORT=${MPC}
        echo "Setting master port to ${MASTER_PORT}."
        break
    fi
done
if [ -z ${MASTER_PORT+x} ]; then
    echo "Could not find an available master port. Exiting."
    exit
fi

# Find available port for Ray (different range to avoid conflicts)
RAY_PORT_CANDIDATES=(6379 6380 6381 6382)
for RPC in ${RAY_PORT_CANDIDATES[@]}; do
    NUM_LISTENING_PROCESSES=$(lsof -Pi :${RPC} -sTCP:LISTEN | wc -l)
    if test $NUM_LISTENING_PROCESSES -eq 0; then
        RAY_PORT=${RPC}
        echo "Setting Ray port to ${RAY_PORT}."
        break
    fi
done
if [ -z ${RAY_PORT+x} ]; then
    echo "Could not find an available Ray port. Exiting."
    exit
fi

ray start --head \
  --port=${RAY_PORT} \
  --node-ip-address=0.0.0.0 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus 4

mkdir openrlhf_work_dir

ray job submit \
    --address='http://127.0.0.1:8265' \
    --runtime-env-json='{"working_dir":"./openrlhf_work_dir"}' \
    -- python3 -m openrlhf.cli.train_ebft_ray \
    --bf16 \
    --gradient_checkpointing \
    --adam_offload \
    --save_hf_ckpt \
    --pretrain_mode \
    --disable_ds_ckpt \
    --colocate_reward_models \
    --use_kl_loss \
    --use_whitening \
    --log_gradients \
    --enable_ema \
    --eval_n_samples_per_prompt=4 \
    --temperature=1.0 \
    --context_max_len=8 \
    --generate_max_len=8 \
    --stride=8 \
    --n_samples_per_prompt=4 \
    --rollout_batch_size=16 \
    --train_batch_size=64 \
    --ce_loss_coef=0.03 \
    --rl_loss_coef=1.0 \
    --diversity_rew_coef=0.5 \
    --alignment_rew_coef=1.0 \
    --critic_learning_rate=0 \
    --critic_lr_head=0 \
    --actor_learning_rate=1e-06 \
    --ref_num_nodes=1 \
    --ref_num_gpus_per_node=1 \
    --critic_num_nodes=1 \
    --critic_num_gpus_per_node=1 \
    --actor_num_nodes=1 \
    --actor_num_gpus_per_node=1 \
    --reward_num_nodes=1 \
    --reward_num_gpus_per_node=1 \
    --micro_train_batch_size=8 \
    --micro_rollout_batch_size=8 \
    --micro_reward_batch_size=8 \
    --max_samples=-1 \
    --eval_max_samples=128 \
    --eval_down_max_samples=128 \
    --max_epochs=1 \
    --num_episodes=1 \
    --prompt_data=sjelassi/opencode-instruct_100k_200tok \
    --eval_dataset=sjelassi/opencode-instruct_100k_200tok \
    --input_key=question \
    --label_key=answer \
    --output_key=answer \
    --prompt_split=train \
    --eval_split=test \
    --prompt_max_len=1024 \
    --save_steps=-1 \
    --logging_steps=1 \
    --save_log_scale_count=-1 \
    --save_path=./ed_checkpoints/ebft_test \
    --ckpt_path=./ed_checkpoints/ebft_test/ckpt \
    --kl_estimator=k2 \
    --init_kl_coef=0.0 \
    --top_p=1.0 \
    --eval_generate_max_len=512 \
    --eval_temperature=1.0 \
    --eval_batch_size=16 \
    --eval_down_batch_size=128 \
    --eval_n_samples_per_prompt_down=4 \
    --eval_temperature_down=1.0 \
    --eval_steps=10 \
    --eval_down_steps=50 \
    --pretrain=Qwen/Qwen2.5-1.5B \
    --critic_pretrain=Qwen/Qwen2.5-1.5B \
    --zero_stage=2 \
    --lr_warmup_ratio=0.03 \
    --lr_scheduler=constant_with_warmup \
    --advantage_estimator=rloo \
    --seed=43 \
    --ema_beta=0.9 \
    --critic_lr_scheduler=constant_with_warmup \
    --critic_classifier_loss_coef=0.0 \
    --hidden_state_method=concat \
    --embed_method=last_token \
    --critic_sequence_level=last_token \
    --classifier_sequence_selection=closest \
    --wandb_run_name=ebft_test
