"""
Runs a sweep over the specified file.
To use, specify `sweep_config` arguments.
"""
import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
import time
import sys
import logging

KEY_ABBREVS = {
    "actor_learning_rate": "a_lr",
    "critic_learning_rate": "c_lr",
    "critic_lr_head": "c_lr_head",
    "actor_lr_warmup_ratio": "a_warmup",
    "critic_lr_warmup_ratio": "c_warmup",
    "pretrain": "pt",
    "critic_pretrain": "cpt",
    "ce_loss_coef": "ce",
    "init_kl_coef": "kl",
    "rl_loss_coef": "rt",
    "rl_loss_warmup_start": "rs",
    "rl_loss_warmup_steps": "rh",
    "diversity_rew_coef": "pr",
    "alignment_rew_coef": "mrt",
    "use_whitening": "wh",
    "document_masking": "dm",
    "qa_masking": "qm",
    "stride": "str",
    "context_max_len": "ctx",
    "generate_max_len": "gen",
    "n_samples_per_prompt": "nsp",
    "prompt_data": "pd",
    "eval_dataset": "ed",
    "temperature": "temp",
    "eval_temperature_down": "ed_temp",
    "eval_temperature_mt": "mt_temp",
    "eval_temperature": "ev_temp",
    "eval_steps": "ev_st",
    "eval_mt_steps": "ev_mt_st",
    "eval_down_steps": "ev_dwn_st",
    "num_episodes": "ne",
    "critic_sequence_level": "csl",
}

MODEL_ABBREVS = {
    "Qwen/Qwen2.5-7B": "qwen7",
    "Qwen/Qwen2.5-1.5B": "qwen15",
    "meta-llama/Llama-3.2-1B": "llama1",
}

DATASET_ABBREVS = {
    "sjelassi/opencode-instruct_100k_200tok": "code200",
    "sjelassi/opencode-instruct_130k": "code130k",
    "sjelassi/alma": "alma",
    "sjelassi/wmt22_test": "wmt22", 
}

ALWAYS_INCLUDE_IN_SAVE_PATH = {
    "prompt_data",
    "eval_dataset",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ebft_sweep")

def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config

def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v
    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list
    
def abbreviate_value(key, value):
    def _fallback_abbrev_slash_name(s: str) -> str:
        # If the identifier looks like "org_or_namespace/name", abbreviate to "name".
        # (Keeps full string for things like local paths or already-short names.)
        if "/" in s:
            return s.rsplit("/", 1)[-1]
        return s

    # Only abbreviate model identifiers for these keys
    if key in ("pretrain", "critic_pretrain") and isinstance(value, str):
        return MODEL_ABBREVS.get(value, _fallback_abbrev_slash_name(value))
    # Abbreviate known datasets for readability; otherwise use a "org/name" -> "name" fallback.
    if key in ("dataset", "prompt_data", "eval_dataset") and isinstance(value, str):
        return DATASET_ABBREVS.get(value, _fallback_abbrev_slash_name(value))
    return value


def run(cli_args):
    if "debug" in cli_args:
        logger.info("Debug Mode")
        master_addr = 5555
        master_port = 6002
        machine_rank = 0
        num_processes = 4
        num_machines = 1
        slurm_cpus_per_task = 4
        slurm_job_id = 0
        slurm_task_id = 1
        dashboard_port = 8265
    else:
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        machine_rank = os.environ.get("SLURM_PROCID")
        num_processes = os.environ.get("NUM_PROCESSES")
        num_machines = os.environ.get("NNODES")
        slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        dashboard_port = int(os.environ.get("DASHBOARD_PORT", 8265))
    logger.info(f"Starting task ID {slurm_task_id} (Job ID: {slurm_job_id})")
    # Compute overrides
    try:
        base_sweep = OmegaConf.load(cli_args.sweep_config)
        list_of_sweeps = base_sweep.pop("sweep")
        config_list = []
        for sweep in list_of_sweeps:
            sweep_config = OmegaConf.merge(base_sweep, sweep)
            config_list += grid_to_list(sweep_config)
        if slurm_task_id >= len(config_list):
            logger.error(f"Task ID {slurm_task_id} exceeds the number of configurations {len(config_list)}")
            return
            
        overrides = config_list[slurm_task_id]
        base_flat = flatten(base_sweep)
        new_path = ["ed_sweep"]
        # Sort keys to make run names deterministic across machines / yaml ordering differences
        for key in sorted(overrides.keys()):
            if 'ckpt_path' in key:
                continue
            if (
                key not in ALWAYS_INCLUDE_IN_SAVE_PATH
                and key in base_flat
                and overrides[key] == base_flat[key]
            ):
                continue
            if str(key) in KEY_ABBREVS:
                new_path.append(KEY_ABBREVS[str(key)])
            else:
                new_path.append(str(key))
            new_path.append(str(abbreviate_value(key, overrides[key])))

        path_suffix = "_".join(new_path)
        overrides['save_path'] = os.path.join(overrides['save_path'], path_suffix)
        overrides['wandb_run_name'] = path_suffix
        if 'ckpt_path' not in overrides:
            overrides['ckpt_path'] = os.path.join(overrides['save_path'], 'ckpt')
            
        logger.info(f'Save path: {overrides["save_path"]}')
        logger.info(f'Checkpoint path: {overrides["ckpt_path"]}')
        logger.info(f"Configuration for task {slurm_task_id}: dashboard port {dashboard_port}")
        logger.info(overrides)
        
        if "debug" in cli_args:
            logger.info(f"Total configs: {len(config_list)}")
            logger.info(config_list)

        def pop_flag(key):
            if key in overrides:
                v = overrides.pop(key)
                return bool(v)
            return False

        no_chat_template        = pop_flag("no_chat_template")
        pretrain_mode           = pop_flag("pretrain_mode")
        debug                   = pop_flag("debug")
        disable_ds_ckpt         = pop_flag("disable_ds_ckpt")
        load_actor_checkpoint   = pop_flag("load_actor_checkpoint")
        load_critic_checkpoint  = pop_flag("load_critic_checkpoint")
        colocate_all_models     = pop_flag("colocate_all_models")
        colocate_reward_models  = pop_flag("colocate_reward_models")
        use_kl_loss             = pop_flag("use_kl_loss")
        use_whitening           = pop_flag("use_whitening")
        log_gradients           = pop_flag("log_gradients")
        enable_ema              = pop_flag("enable_ema")
        document_masking        = pop_flag("document_masking")
        qa_masking              = pop_flag("qa_masking")

        # Ray job submission command
        launch_args = [
            f"ray job submit --address='http://127.0.0.1:{dashboard_port}'",
            '--runtime-env-json=\'{\"working_dir\":\"./openrlhf_work_dir\"}\'',
            '-- python3 -m openrlhf.cli.train_ebft_ray',
            "--bf16",
            "--gradient_checkpointing",
            "--adam_offload",
            "--save_hf_ckpt",
        ]

        if not no_chat_template:
            launch_args.append("--apply_chat_template")
        if pretrain_mode:
            launch_args.append("--pretrain_mode")
        if debug:
            launch_args.append("--debug")
        if disable_ds_ckpt:
            launch_args.append("--disable_ds_ckpt")
        if load_actor_checkpoint:
            launch_args.append("--load_actor_checkpoint")
        if load_critic_checkpoint:
            launch_args.append("--load_critic_checkpoint")
        if colocate_all_models:
            launch_args.append("--colocate_all_models")
        if colocate_reward_models:
            launch_args.append("--colocate_reward_models")
        if use_kl_loss:
            launch_args.append("--use_kl_loss")
        if use_whitening:
            launch_args.append("--use_whitening")
        if log_gradients:
            launch_args.append("--log_gradients")
        if enable_ema:
            launch_args.append("--enable_ema")
        if document_masking:
            launch_args.append("--document_masking")
        if qa_masking:
            launch_args.append("--qa_masking")

        for k, v in overrides.items():
            launch_args.append(f"--{k}={v}")

        launch_args.append(f"--slurm_job={slurm_job_id}_{slurm_task_id}")

        if "debug" in cli_args:
            logger.info(f"Launch command: {' '.join(launch_args)}")

        # Maximum retries for Ray connection
        max_retries = 3
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Submitting job, attempt {attempt+1}/{max_retries}")
                result = subprocess.run([
                    "bash",
                    "-c", ' '.join(launch_args)
                ], capture_output=True, text=True, check=False)
                
                if result.returncode != 0:
                    logger.error(f"Command failed with return code {result.returncode}")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    
                    # Check for specific error patterns
                    if "No available agent" in result.stderr or "500" in result.stderr:
                        logger.warning("Ray agent not available, retrying...")
                        time.sleep(retry_delay)
                        continue
                else:
                    logger.info("Job submitted successfully")
                    break
            except Exception as e:
                logger.exception(f"Error running command: {e}")
                time.sleep(retry_delay)
        else:
            logger.error(f"Failed to submit job after {max_retries} attempts")
            
    except Exception as e:
        logger.exception(f"Error in run function: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    run(cli_args)