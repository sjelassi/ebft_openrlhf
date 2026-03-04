import time
from abc import ABC
import math
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

import ray
import torch
from torch import distributed as dist

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or {}

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id, n_samples_per_prompt: int = 1) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences
            n_samples_per_prompt: Number of samples per prompt (unused in base implementation)

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def update_samples_with_rewards(rewards_info, samples_list):
    """Process rewards info and update samples with rewards, scores and extra logs.

    Args:
        rewards_info: List of reward information dictionaries
        samples_list: List of Experience objects to update
    """
    # Process rewards and scores
    samples_len = [len(sample.sequences) for sample in samples_list]

    rewards_list = torch.cat([info["rewards"] for info in rewards_info], dim=0).split(samples_len)
    if "scores" in rewards_info[0]:
        scores_list = torch.cat([info["scores"] for info in rewards_info], dim=0).split(samples_len)
    else:
        scores_list = rewards_list

    # Process extra_logs if present
    if "extra_logs" in rewards_info[0]:
        # Merge all extra_logs tensors first
        merged_logs = {
            key: torch.cat([logs[key] for logs in [info["extra_logs"] for info in rewards_info]], dim=0).split(
                samples_len
            )
            for key in rewards_info[0]["extra_logs"].keys()
        }

    # Update samples with rewards, scores and extra logs
    for i, samples in enumerate(samples_list):
        samples.rewards = rewards_list[i]
        samples.scores = scores_list[i]
        samples.info["score"] = scores_list[i]
        samples.info["reward"] = rewards_list[i]
        if "extra_logs" in rewards_info[0]:
            for key, values in merged_logs.items():
                samples.info[key] = values[i]

    return samples_list


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        # Set up sampling parameters
        # max_new = kwargs.get("max_new_tokens", getattr(self.strategy.args, "generate_max_len", 1))
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=self.args.generate_max_len,
            min_tokens=self.args.generate_max_len,
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        #max_response_length = kwargs.get("max_new_tokens", 1024) #commented by Samy
        max_response_length = self.args.generate_max_len
        truncate_length = self.prompt_max_len + max_response_length

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompt_token_ids = sum([[prompt.tolist()] * n_samples_per_prompt for prompt in all_prompts], [])
        all_label_token_ids = sum([[label.tolist()] * n_samples_per_prompt for label in all_labels], [])
        all_prompts = all_prompt_token_ids
        all_labels = all_label_token_ids

        # Distribute requests to engines and collect responses
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        

        output_examples = [self.tokenizer.decode(all_outputs[i].outputs[0].token_ids) for i in range(min(50,len(all_outputs)))]
        print(f"SAMPLE RESPONSES: {output_examples}")
        # Process outputs into Experience objects
        samples_list = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]

            # Concatenate prompt and output tokens
            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            
            ##old code
            # if output.outputs[0].token_ids[-1] != eos_token_id:
            #     input_ids.append(eos_token_id)

            assert (input_ids[:len(prompt)] == prompt)
            attention_mask = [1] * len(input_ids)

            sequences = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # Create action mask based on output token positions
            action_mask = torch.zeros_like(attention_mask)
            response_length = len(output.outputs[0].token_ids)
            action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + response_length] = 1

            # Truncate to max length if needed
            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            # CRITICAL: Shift action_mask by 1 to align with model's predictions
            # The action at position i predicts token i+1, so we need to shift
            # This matches the original PPO implementation
            action_mask = action_mask[1:truncate_length].to("cpu")
 

 
            total_length = attention_mask.float().sum()
            is_clipped = response_length >= max_response_length

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
            }

            rollout_samples = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                prompts=[prompt],
                labels=[label],
                info=info,
            )
            samples_list.append(rollout_samples)
      
        # Get rewards from remote reward models if needed
        # This is required by dynamic sampling
        remote_reward_model = kwargs.get("remote_reward_model", None)
        if remote_reward_model:

            # Flatten to per-sample lists (not per micro-batch)
            all_queries = sum([s.sequences.tolist() for s in samples_list], [])
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards info from remote model
            rewards_info = ray.get(remote_reward_model.get_rewards.remote(all_queries, all_prompts, all_labels))
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

 
    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]


        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Build flattened per-sample inputs for the remote reward model
            queries_list = sum([seq.tolist() for seq in sequences_list], [])
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])


            # Keep the remote call asynchronous
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        else:
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        override_temperature_list = [True] * len(sequences_list)
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
            override_temperature=override_temperature_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))
            override_temperature_list = [True] * len(sequences_list)
            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
                override_temperature=override_temperature_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
                override_temperature=override_temperature_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        # Process rewards based on source
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model
            rewards_info = ray.get(r_refs)
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            # Reward Model
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # after computing action_log_probs in make_experience added by samy
            assert action_log_probs.shape[-1] == samples.action_mask.shape[-1]
            # per-sample check
            per_sample_mask = samples.action_mask.to(torch.int).sum(dim=-1).view(-1)
            per_sample_resp = samples.info["response_length"].view(-1)  # handles [B] or [B,1]
            # print(f"ACTIONMASKSUM: {per_sample_mask}; RESPONSE LEN: {per_sample_resp}")
            assert torch.all(per_sample_mask == per_sample_resp), \
                f"{per_sample_mask.tolist()} vs {per_sample_resp.tolist()}"
            # experience_maker.make_experience (after rewards are set)
           

            # Update experience with new information
            # print(f"ACTION LOG PROBS: {action_log_probs.shape}")
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list


    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # DAPO reward shaping with optional overlong penalty - Apply BEFORE dynamic indices processing
        if args.overlong_buffer_len is not None:
            assert (
                args.generate_max_len >= args.overlong_buffer_len
            ), "generate_max_len must be larger than overlong_buffer_len"
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            # Apply penalty to each experience's rewards based on response length
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    # Cap the exceed_len to overlong_buffer_len to prevent excessive penalty
                    exceed_len = min(valid_response_length - expected_len, overlong_buffer_len)
                    if exceed_len > 0:
                        overlong_penalty = -exceed_len / overlong_buffer_len * overlong_penalty_factor
                        # Apply penalty to the j-th reward in this experience
                        experience.rewards[j] += overlong_penalty

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)


        
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            old_reward = reward
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )
            # print(f"OLDR: {old_reward}; REW: {reward}")
            # Now the shapes must match
            # inside compute_advantages_and_returns, right after compute_reward
            assert reward.shape[-1] == experience.action_mask.shape[-1]
            assert experience.action_mask.sum().item() == int(experience.info["response_length"].sum())


            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            
    
            
            # remove unnecessary info
            experience.kl = None

        ## old code    
        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd
            

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    @torch.no_grad()
    def make_ppls_experience_batch(self, rollout_samples):
        """
        Build inputs needed to compute perplexities using ground-truth completions.

        Returns a dict with per-rank accumulated NLL sums and token counts for:
        - ED-completion tokens only (matches compute_perplexity_ed_dataset)
        - Full sequence tokens (matches compute_perplexity on pretrain-style loss)
        """
        # Deduplicate by (prompt, label) so we evaluate each ground-truth only once
        seen_pairs = set()
        unique_rollout_samples = []
        for sample in rollout_samples:
            # Extract prompt/label token id lists robustly
            p_field = sample.prompts
            l_field = sample.labels
            p_ids = p_field[0] if isinstance(p_field, list) and len(p_field) > 0 and isinstance(p_field[0], (list, tuple)) else p_field
            l_ids = l_field[0] if isinstance(l_field, list) and len(l_field) > 0 and isinstance(l_field[0], (list, tuple)) else l_field
            key = (tuple(p_ids), tuple(l_ids))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unique_rollout_samples.append(sample)

        # Each batch of samples will be scheduled to an effective Ray Actor (i.e., a DP rank)
        samples_list = self.split_rollout_samples(unique_rollout_samples)

        # Prepare tensors for GT sequences and masks (evaluation naming to avoid confusion)
        build = self.make_ppl_experience(samples_list)
        eval_sequences_list       = build["eval_sequences"]        # list of [B, S]
        eval_attention_mask_list  = build["eval_attention_masks"]  # list of [B, S]
        eval_comp_masks_list      = build["eval_comp_masks"]       # list of [B, S-1]
        eval_full_masks_list      = build["eval_full_masks"]       # list of [B, S-1]

        args = self.strategy.args
        override_temperature_list = [True] * len(eval_sequences_list)

        # Query actor for log probs on completion-only mask
        comp_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=eval_sequences_list,
            action_mask=eval_comp_masks_list,
            attention_mask=eval_attention_mask_list,
            override_temperature=override_temperature_list,
        )

        # And also for full-sequence mask
        full_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=eval_sequences_list,
            action_mask=eval_full_masks_list,
            attention_mask=eval_attention_mask_list,
            override_temperature=override_temperature_list,
        )

        # Sync to avoid GPU OOM when colocating models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(comp_log_probs_ref)
            ray.get(full_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Handle duplicated returns from ring/tp groups (keep one copy)
        duplicate_factor   = args.ring_attn_size * args.ds_tensor_parallel_size
        comp_log_probs_list = sum(ray.get(comp_log_probs_ref)[::duplicate_factor], [])
        full_log_probs_list = sum(ray.get(full_log_probs_ref)[::duplicate_factor], [])

        # Accumulate local sums in float64 for stability (on CPU to save GPU mem)
        device        = torch.device("cpu")
        comp_loss_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        comp_tok_sum  = torch.tensor(0.0, dtype=torch.float64, device=device)
        full_loss_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        full_tok_sum  = torch.tensor(0.0, dtype=torch.float64, device=device)

        for comp_lp, full_lp, comp_m, full_m in zip(
            comp_log_probs_list, full_log_probs_list, eval_comp_masks_list, eval_full_masks_list
        ):
            # All in float64 on CPU for robust accumulation
            comp_lp = comp_lp.to(dtype=torch.float64, device=device)
            full_lp = full_lp.to(dtype=torch.float64, device=device)
            comp_m  = comp_m.to(dtype=torch.float64, device=device)
            full_m  = full_m.to(dtype=torch.float64, device=device)

            # Negative log likelihood sums over selected tokens
            comp_loss_sum += (-comp_lp * comp_m).sum()
            comp_tok_sum  += comp_m.sum()
            full_loss_sum += (-full_lp * full_m).sum()
            full_tok_sum  += full_m.sum()

        return {
            "ed_comp_loss_sum":  comp_loss_sum,
            "ed_comp_tok_sum":   comp_tok_sum,
            "sft_full_loss_sum": full_loss_sum,
            "sft_full_tok_sum":  full_tok_sum,
        }


    def make_ppl_experience(self, samples_list: List[Experience]):
        """
        Prepare per-microbatch tensors for ground-truth sequences and corresponding masks
        needed for ED-completion PPL and full-sequence PPL.

        Returns a dict with lists of tensors: sequences [B,S], attention_masks [B,S],
        comp_masks [B,S-1], full_masks [B,S-1].
        """
        start_time = time.time()
        logger.info(f"🚀 Preparing PPL inputs with {sum([len(s.sequences) for s in samples_list])} samples")

        eval_sequences_list      = []
        eval_attention_mask_list = []
        eval_comp_masks_list     = []
        eval_full_masks_list     = []

        # pad id for sequences (fallback to eos if pad is None)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)

        for s in samples_list:
            # After concat, s.prompts / s.labels are lists-of-lists (length B)
            # In the simplest case they may still be a single list; normalize to list-of-lists.
            if isinstance(s.prompts, list) and (len(s.prompts) > 0) and isinstance(s.prompts[0], (list, tuple)):
                prompts_batch = s.prompts
            else:
                prompts_batch = [s.prompts]

            if isinstance(s.labels, list) and (len(s.labels) > 0) and isinstance(s.labels[0], (list, tuple)):
                labels_batch = s.labels
            else:
                labels_batch = [s.labels]

            assert len(prompts_batch) == len(labels_batch), \
                f"prompts/labels batch mismatch: {len(prompts_batch)} vs {len(labels_batch)}"

            # Build per-sample (1D) tensors first
            eval_seq_1d_list   = []
            eval_attn_1d_list  = []
            eval_comp_1d_list  = []
            eval_full_1d_list  = []

            for p_ids, l_ids in zip(prompts_batch, labels_batch):
                p = torch.as_tensor(p_ids, dtype=torch.long)
                l = torch.as_tensor(l_ids, dtype=torch.long)
                assert p.numel() >= 1 and l.numel() > 0, \
                    "ED PPL requires prompt_max_len >= 1 and generate_max_len > 0."

                # Full sequence and attention
                seq = torch.cat([p, l], dim=0)           # [S]
                att = torch.ones_like(seq, dtype=torch.long)

                # Next-token mask length is S-1
                S = seq.numel()
                L = p.numel()
                C = l.numel()

                # Completion-only mask: positions predicting completion tokens
                comp = torch.zeros(S - 1, dtype=torch.long)
                start = L - 1
                end   = start + C          # exclusive
                comp[start:end] = 1

                # Full-sequence mask: predict every next token
                full = torch.ones(S - 1, dtype=torch.long)

                eval_seq_1d_list.append(seq)
                eval_attn_1d_list.append(att)
                eval_comp_1d_list.append(comp)
                eval_full_1d_list.append(full)

            # Now pad to form [B,S] (and [B,S-1]) batches for this micro-batch Experience
            # For 1D inputs, use stack=True to obtain [B, S] instead of a flattened 1D tensor
            eval_seq_batch  = zero_pad_sequences(eval_seq_1d_list,  side="right", value=pad_id, stack=True).to(torch.long)
            eval_att_batch  = zero_pad_sequences(eval_attn_1d_list, side="right", value=0,      stack=True).to(torch.long)

            # For masks we pad with 0 on the right
            eval_comp_batch = zero_pad_sequences(eval_comp_1d_list, side="right", value=0,      stack=True).to(torch.long)
            eval_full_batch = zero_pad_sequences(eval_full_1d_list, side="right", value=0,      stack=True).to(torch.long)

            # Sanity: masks must be one shorter than sequences along last dim
            assert eval_comp_batch.shape[1] == eval_seq_batch.shape[1] - 1, \
                f"comp mask shape {eval_comp_batch.shape} not S-1 for seq {eval_seq_batch.shape}"
            assert eval_full_batch.shape[1] == eval_seq_batch.shape[1] - 1, \
                f"full mask shape {eval_full_batch.shape} not S-1 for seq {eval_seq_batch.shape}"

            eval_sequences_list.append(eval_seq_batch)
            eval_attention_mask_list.append(eval_att_batch)
            eval_comp_masks_list.append(eval_comp_batch)
            eval_full_masks_list.append(eval_full_batch)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ PPL input preparation completed in {time_str}")

        return {
            "eval_sequences":        eval_sequences_list,
            "eval_attention_masks":  eval_attention_mask_list,
            "eval_comp_masks":       eval_comp_masks_list,
            "eval_full_masks":       eval_full_masks_list,
        }


    # @torch.no_grad()
    # def compute_ppl(self, stats_or_exps):
    #     """
    #     Compute perplexities from either:
    #     - A dict of local sums returned by make_ppls_experience_batch, or
    #     - A list of Experience with action_log_probs and action_mask (fallback: ED-completion PPL only).
    #     """
    #     device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    #     if isinstance(stats_or_exps, dict):
    #         # Reduce across processes, CPU-safe: avoid forcing CUDA when unavailable
    #         payload = {
    #             "ed_comp_loss_sum":  stats_or_exps["ed_comp_loss_sum"].to(device=device, dtype=torch.float64),
    #             "ed_comp_tok_sum":   stats_or_exps["ed_comp_tok_sum"].to(device=device, dtype=torch.float64),
    #             "sft_full_loss_sum": stats_or_exps["sft_full_loss_sum"].to(device=device, dtype=torch.float64),
    #             "sft_full_tok_sum":  stats_or_exps["sft_full_tok_sum"].to(device=device, dtype=torch.float64),
    #         }

    #         if torch.cuda.is_available():
    #             reduced = self.strategy.all_reduce(payload)
    #         else:
    #             # CPU-only distributed reduction using current backend (e.g., gloo)
    #             reduced = {}
    #             for k, v in payload.items():
    #                 t = v
    #                 if dist.is_initialized() and dist.get_world_size() > 1:
    #                     if k.endswith("_loss_sum") or k.endswith("_tok_sum"):
    #                         dist.all_reduce(t, op=dist.ReduceOp.SUM)
    #                     else:
    #                         dist.all_reduce(t, op=dist.ReduceOp.SUM)
    #                 reduced[k] = t

    #         ed_tok = float(reduced["ed_comp_tok_sum"].detach().cpu())
    #         sft_tok = float(reduced["sft_full_tok_sum"].detach().cpu())

    #         if ed_tok > 0.0:
    #             ed_loss = float(reduced["ed_comp_loss_sum"].detach().cpu()) / ed_tok
    #             ed_ppl  = math.exp(ed_loss)
    #         else:
    #             ed_loss, ed_ppl = float("nan"), float("nan")

    #         if sft_tok > 0.0:
    #             sft_loss = float(reduced["sft_full_loss_sum"].detach().cpu()) / sft_tok
    #             sft_ppl  = math.exp(sft_loss)
    #         else:
    #             sft_loss, sft_ppl = float("nan"), float("nan")

    #         return {
    #             "ed_loss":    ed_loss,
    #             "ed_ppl":     ed_ppl,
    #             "gpt_loss":   sft_loss,
    #             "perplexity": sft_ppl,
    #         }

    #     # Fallback: compute ED completion ppl directly from Experience list
    #     total_loss = 0.0
    #     total_tokens = 0.0
    #     for exp in stats_or_exps:
    #         lp = exp.action_log_probs.to(dtype=torch.float64)
    #         m  = exp.action_mask.to(dtype=torch.float64)
    #         total_loss  += float((-lp * m).sum().item())
    #         total_tokens += float(m.sum().item())

    #     if torch.cuda.is_available():
    #         reduced = self.strategy.all_reduce({
    #             "ed_comp_loss_sum": torch.tensor(total_loss,  dtype=torch.float64, device=device),
    #             "ed_comp_tok_sum":  torch.tensor(total_tokens, dtype=torch.float64, device=device),
    #         })
    #     else:
    #         # CPU-only reduction
    #         ed_comp_loss_sum = torch.tensor(total_loss,  dtype=torch.float64, device=device)
    #         ed_comp_tok_sum  = torch.tensor(total_tokens, dtype=torch.float64, device=device)
    #         if dist.is_initialized() and dist.get_world_size() > 1:
    #             dist.all_reduce(ed_comp_loss_sum, op=dist.ReduceOp.SUM)
    #             dist.all_reduce(ed_comp_tok_sum,  op=dist.ReduceOp.SUM)
    #         reduced = {
    #             "ed_comp_loss_sum": ed_comp_loss_sum,
    #             "ed_comp_tok_sum":  ed_comp_tok_sum,
    #         }
    #     tok_sum = float(reduced["ed_comp_tok_sum"].detach().cpu())
    #     if tok_sum > 0.0:
    #         loss_mean = float(reduced["ed_comp_loss_sum"].detach().cpu()) / tok_sum
    #         ppl = math.exp(loss_mean)
    #     else:
    #         loss_mean, ppl = float("nan"), float("nan")
    #     return {"ed_loss": loss_mean, "ed_ppl": ppl}