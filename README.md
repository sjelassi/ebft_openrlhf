# Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models

Accompanying code for experiments in "[Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models](https://arxiv.org/abs/2603.12248)". We are grateful to the contributors of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), from which this code is built on top of.

## Setup
The code is up-to-date with OpenRLHF 0.8.9. Run the setup script which creates a conda environment with necessary dependencies:
```
bash ebm_openrlhf/scripts/setup_env.sh
```

## Running EBFT
The `ebm_openrlhf/scripts/` directory contains scripts for launching training runs:

- **`run_ebft_example.sh`**: An example bash script that starts a local Ray cluster and submits an EBFT training job via `openrlhf.cli.train_ebft_ray`.\
- **`ebft_sweep.py`**: A Python script for launching SLURM array jobs that sweep over model checkpoints and hyperparameters. It reads a YAML config, expands the hyperparameter grid, maps each SLURM task array index to a specific configuration, and submits the corresponding Ray job.
- **`run_ebft_sweep.sh`**: The `sbatch` script that launches `ebft_sweep.py` as a SLURM array job (e.g., `--array=0-26%9` for up to 9 jobs running in parallel).

The YAML configs used to reproduce our results for the three tasks (structured code, unstructured code, and translation) are in `ebm_openrlhf/configs/`: `qa_code.yaml`, `unstructured_code.yaml`, and `translation.yaml`. Each config specifies dataset paths, model checkpoints, generation parameters, loss coefficients, and a `sweep` section listing the hyperparameter combinations to evaluate.

## Computing Feature-Matching Loss
`ebm_openrlhf/scripts/evaluate_reward_ce.py` is a standalone script for computing feature-matching rewards and perplexity on a dataset. An example is provided in `ebm_openrlhf/scripts/run_eval_reward_ce_example.sh`.

The script uses the `ebm_openrlhf/inference_loss/` module, which packages the core evaluation logic into three self-contained classes:
- **`StridedActorModel`** — loads a causal LM and generates samples using strided attention masks, and computes log-probabilities.
- **`StridedCriticModel`** — loads a model (via `AutoModel`) and extracts hidden-state embeddings with strided blocks.
- **`EvaluationMetrics`** — computes alignment rewards (cosine similarity to ground-truth embeddings), diversity rewards (inter-sample similarity), and perplexity/cross-entropy. Reports both Pass@1 and Pass@k variants.

See `ebm_openrlhf/inference_loss/README.md` for full documentation and usage examples.

## EBFT Code Overview
EBFT is structured similarly to the distributed PPO implementation in OpenRLHF, using Ray for orchestration. The main entry point is `ebm_openrlhf/openrlhf/cli/train_ebft_ray.py`, which initializes the Ray cluster, creates placement groups, and launches the distributed training actors.

The key components are:

- **EBFT Trainer** (`openrlhf/trainer/ebft_trainer.py`): The central training loop. It coordinates rollout generation, embedding extraction, reward computation, advantage estimation, and policy updates. It also runs periodic evaluation (Pass@1, Pass@k, perplexity, BLEU).
- **EBFT Actor** (`openrlhf/trainer/ray/ebft_actor.py`): A Ray actor wrapping the policy model. During training it computes log-probabilities and policy gradients using the strided attention structure.
- **EBFT Critic** (`openrlhf/trainer/ray/ebft_critic.py`): A Ray actor wrapping the critic model. It extracts hidden-state embeddings from the full sequences (prompt + generation) using strided blocks, which are then used to compute alignment and diversity rewards.
- **Strided attention mask** (`openrlhf/models/utils.py`, `build_strided_attention_mask_and_positions`): Constructs the custom 4D attention mask and position IDs that implement parallel strided-block generation. Generation is divided into blocks; block *k* attends to a context window offset by *k × stride* tokens into the prompt, allowing multiple independent context windows to be processed in a single forward pass.
- **EBFT Experience Maker** (`openrlhf/trainer/ppo_utils/ebft_experience_maker.py`): Generates and stores rollout experiences. Each `Experience` holds the full sequences, attention/action masks, doc IDs, QA masks, per-sample rewards (alignment, diversity, combined), and log-probabilities.
- **Loss functions** (`openrlhf/models/loss.py`): `EBFTPolicyLoss` implements the PPO/GSPO policy gradient loss. `ClassifierLoss` implements the binary log-sigmoid real-vs-fake loss used for (optional) critic training. `ClassifierAccuracy` reports associated classifier metrics.

### Embedding method and hidden-state selection

Two orthogonal choices control how embeddings are formed from the transformer's internal representations:

**`--hidden_state_method`** selects which transformer layers to aggregate:
- `last_only` — final layer only
- `mean` — mean of all layers
- `middle` — average of the two middle layers
- `middle_concat` / `middle_stack` — concatenate or stack the two middle layers
- `concat` — concatenate the layers at 25%, 50%, and 75% depth
- `stack` — stack the same three quartile layers as separate features
- `layer_N` — a single specific layer by index
- `concat_layers_N_M_...` / `stack_layers_N_M_...` — concatenate or stack an arbitrary list of layers

**`--embed_method`** controls how the per-token hidden states are pooled into a per-block embedding:
- `last_token` — use only the final token's hidden state
- `mean_pooling` — average across all tokens in the block
- `concat` — concatenate the hidden states at the 25th, 50th, and 75th token positions
- `token` — retain per-token embeddings (used for token-level reward signals)

Optionally, `--use_whitening` applies SVD-based whitening to decorrelate the embeddings before computing cosine-similarity rewards.

### Critic learning (experimental)

In our paper we exclusively use a **frozen critic** (the critic model's weights are fixed throughout training; it is used only to extract embeddings). However, the codebase also supports joint actor–critic training via a binary classifier head attached to the critic. When `--critic_learning_rate` or `--critic_lr_head` are non-zero, the critic is unfrozen and a classifier head is trained to distinguish the model's generated continuations ("fake") from ground-truth continuations ("real") using the extracted embeddings. The relevant arguments are:

| Argument | Description |
|---|---|
| `--critic_learning_rate` | LR for the critic base model (set to `0` to freeze) |
| `--critic_lr_head` | LR for the classifier head (defaults to `critic_learning_rate` if unset) |
| `--critic_lr_scheduler` | LR schedule for the critic |
| `--critic_lr_warmup_ratio` | Optional separate warmup ratio for the critic |
| `--critic_classifier_loss_coef` | Weight of the classifier loss in the total critic loss |
| `--critic_sequence_level` | How classifier logits are aggregated per block: `token`, `last_token`, `mean_pooling`, or `concat` |
| `--classifier_sequence_selection` | Which generated sample to contrast against GT: `first`, `all`, `closest`, or `only_different` |

> **Disclaimer:** We have not rigorously tested joint actor–critic training and cannot vouch for its stability or performance. The frozen-critic setting is the one evaluated in the paper.

## Citation
```bibtex
@misc{jelassi2026matchingfeaturestokensenergybased,
      title={Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models}, 
      author={Samy Jelassi and Mujin Kwun and Rosie Zhao and Yuanzhi Li and Nicolo Fusi and Yilun Du and Sham M. Kakade and Carles Domingo-Enrich},
      year={2026},
      eprint={2603.12248},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.12248}, 
}
```
