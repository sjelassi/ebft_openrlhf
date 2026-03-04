import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union
import ray
import torch
from torch import distributed as dist
from openrlhf.utils.utils import pad_to_longest

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, extract_and_reorder_rewards
from openrlhf.utils.embedding_utils import (
    prepare_tensors_for_embedding,
    whiten_embeddings_batched,
    get_alignment_rewards,
    get_diversity_rewards
)
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import zero_pad_sequences

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
    
    # Sequence tensors
    generated_sequences: list[str] = None
    prompts: torch.Tensor = None  # X: Original full prompt
    full_sequences: torch.Tensor = None  # XZ: Full sequence (full prompt + generated)
    doc_ids: torch.Tensor = None  # B, S
    qa_masks: torch.Tensor = None  # B, S

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

    prompt_strings: list[str] = None
    label_strings: list[str] = None
    unit_tests: list[str] = None
    entry_points: list[str] = None
    code_contexts: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    diversity_rewards: torch.Tensor = None  # used for advantage calculation
    gt_rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        # New field names
        generated_sequences=None,
        prompts=None,
        full_sequences=None,
        doc_ids=None,
        qa_masks=None,
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
        prompt_strings=None,
        label_strings=None,
        unit_tests=None,
        entry_points=None,
        code_contexts=None,
        rewards=None,
        diversity_rewards=None,
        gt_rewards=None,
        scores=None,
        info=None,
    ):
        # New field names
        self.prompts = prompts 
        self.full_sequences = full_sequences  
        self.generated_sequences = generated_sequences
        self.doc_ids = doc_ids
        self.qa_masks = qa_masks

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
        self.prompt_strings = prompt_strings
        self.label_strings = label_strings
        self.unit_tests = unit_tests
        self.entry_points = entry_points
        self.code_contexts = code_contexts
        self.rewards = rewards
        self.diversity_rewards = diversity_rewards
        self.gt_rewards = gt_rewards
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


class SamplesGenerator:
    def __init__(self, actor_model_group, strategy, tokenizer):
        self.strategy = strategy
        self.args = strategy.args
        self.actor_model_group = actor_model_group
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate_samples(self, all_sequences: List[str], doc_ids: List[str], qa_masks: List[str], **generate_kwargs) -> Tuple[torch.Tensor, List[Experience]]:
        """Generate samples from input sequences using parallel strided generation.

        This is the main entry point for sample generation. It delegates to the
        parallel generation method which implements multi-token prediction with
        stride-based context windows.

        Args:
            all_sequences: List of input prompt sequences to generate from
            **generate_kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            List of Experience objects containing:
                - Generated sequences with full context
                - Original prompts and full sequences
                - Action masks for training
                - Metadata about sequence lengths
        """
        # Generate samples using the parallel strided blocks approach
        # This method handles batching, generation, and Experience creation
        samples_list = self._generate_parallel(all_sequences, doc_ids, qa_masks, **generate_kwargs)

        return samples_list
    
    @torch.no_grad()
    def generate_samples_for_downstream(self, all_prompts: List[str], all_labels: List[str], all_unit_tests: List[str] = None, all_entry_points: List[str] = None, all_code_contexts: List[str] = None, **generate_kwargs) -> Tuple[torch.Tensor, List[Experience]]:
        """Generate samples from input sequences using parallel strided generation.

        This is the main entry point for sample generation. It delegates to the
        parallel generation method which implements multi-token prediction with
        stride-based context windows.

        Args:
            all_sequences: List of input prompt sequences to generate from
            **generate_kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            List of Experience objects containing:
                - Generated sequences with full context
                - Original prompts and full sequences
                - Action masks for training
                - Metadata about sequence lengths
        """
        # Generate samples using the parallel strided blocks approach
        # This method handles batching, generation, and Experience creation
        samples_list = self._generate_for_downstream(all_prompts, all_labels, all_unit_tests=all_unit_tests, all_entry_points=all_entry_points, all_code_contexts=all_code_contexts, **generate_kwargs)

        return samples_list
    
 

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

    @torch.no_grad()
    def _generate_parallel(self, all_sequences: List[List[int]], all_doc_ids: List[List[int]], all_qa_masks: List[List[int]], **kwargs) -> Tuple[torch.Tensor, List[Experience]]:
        """Generate parallel samples using multi-token prediction with stride.

        This method coordinates the generation process:
        1. Sends prompts to actors for strided generation
        2. Collects full sequences (Z) from actors
        3. Reconstructs full sequences and creates Experience objects

        Args:
            all_sequences: List of input prompt tensors (X)
            **kwargs: Additional generation parameters (temperature, top_p)

        Returns:
            List of Experience objects containing:
            - Original prompts (X): full original prompts
            - Full sequences: full prompt + generated tokens
            - Action masks for training
        """
        args = self.strategy.args
        batch_size = args.micro_rollout_batch_size
        if "generate_max_len" in kwargs:
            generation_length = kwargs["generate_max_len"]
        else:
            generation_length = self.args.generate_max_len
        context_length = self.args.context_max_len
        context_stride = self.args.stride
        if 'n_samples_per_prompt' in kwargs:
            n_samples = kwargs['n_samples_per_prompt']
            batch_size = n_samples
        else:
            n_samples = self.args.n_samples_per_prompt
 

        if 'top_p' in kwargs:
            top_p = kwargs['top_p']
        else:
            top_p = self.args.top_p
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
        else:
            temperature = self.args.temperature

        actors = self.actor_model_group._actor_handlers
        if not actors:
            raise RuntimeError("No actors available in actor_model_group.")
        
        all_sequences = [seq.clone() for seq in all_sequences for _ in range(n_samples)]  # Repeat each prompt n_samples times
        if getattr(self.args, "debug", False):
            logger.info(f"doc_ids shape: {all_doc_ids[0].shape}")
        all_doc_ids = [doc_id.clone()[:self.args.prompt_max_len] if doc_id.dim() == 1 else doc_id.clone()[:,:self.args.prompt_max_len] for doc_id in all_doc_ids for _ in range(n_samples)] # repeat
        all_qa_masks = [qa_mask.clone() for qa_mask in all_qa_masks for _ in range(n_samples)] # repeat
        batches = []
        for s in range(0, len(all_sequences), batch_size):
            chunk = torch.stack(all_sequences[s : s + batch_size])
            doc_ids = torch.stack(all_doc_ids[s : s + batch_size])
            qa_masks = torch.stack(all_qa_masks[s : s + batch_size])
            batches.append((s // batch_size, chunk, doc_ids, qa_masks))

        # ---------- 3) Submit one RPC per batch (true async), round-robin across actors ----------
        inflight_refs = []
        for k, (bid, batch_2d, doc_ids, qa_masks) in enumerate(batches):
            a = actors[k % len(actors)]
            # IMPORTANT: pass a single 2D batch, not a list of batches
            ref = a.generate_strided_blocks.remote(
                prompt_token_ids=batch_2d,
                doc_ids=doc_ids,
                stride=context_stride,
                context_length=context_length,
                generate_length=generation_length,
                temperature=temperature,
                top_p=top_p,
                document_masking=self.args.document_masking,
            )
            inflight_refs.append((bid, ref))

        # ---------- 4) Drain asynchronously; store outputs by batch id ----------
        id_by_ref = {ref: bid for (bid, ref) in inflight_refs}
        pending = [ref for (_, ref) in inflight_refs]
        results_by_bid = {}  # bid -> batch_out

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            r = ready[0]
            bid = id_by_ref[r]
            batch_out = ray.get(r)
            results_by_bid[bid] = batch_out
        # ---------- 5) Reassemble in submission order; flatten ----------
        ordered = [results_by_bid[i] for i in range(len(results_by_bid))]
        full_sequences: List[torch.Tensor] = []
        for out in ordered:
            full_sequences.append(out)

        # ---------- 6) Build Experience objects (aligned 1:1) ----------
        samples_list: List[Experience] = []
        for pos, full_sequence in enumerate(full_sequences):
            _, original_prompt, doc_ids, qa_masks = batches[pos]  # batched sample

            prompt_length = original_prompt.shape[1]
            generated_tokens_count = full_sequence.shape[1] - prompt_length
            if generated_tokens_count < 0:
                raise ValueError(
                    f"full sequence shorter than prompt: |full|={full_sequence.shape[1]}, prompt={prompt_length}"
                )
            if full_sequence.shape[1] != prompt_length + generated_tokens_count:
                raise ValueError(
                    f"full sequence length mismatch: |full|={full_sequence.shape[1]}, prompt={prompt_length}, generated={generated_tokens_count}"
                )

            # 1 for generated tokens (shifted by 1 for NTP)
            action_mask = torch.zeros(original_prompt.shape[0], full_sequence.shape[1], dtype=torch.bool)
            action_mask[:,prompt_length:] = 1
            action_mask = action_mask[:,1:]  # CPU


            strided_qa_mask = (
                qa_masks[:, context_length:]
                .unfold(1, generation_length, context_stride)  # (B, num_blocks, G)
                .transpose(1, 2)  # (B, G, num_blocks)
                .reshape(qa_masks.shape[0], -1)  # (B, G * num_blocks)
            )
            qa_mask = torch.cat([qa_masks, strided_qa_mask], dim=1)

            strided_doc_ids = (
                doc_ids[:, context_length:]
                .unfold(1, generation_length, context_stride)  # (B, num_blocks, G)
                .transpose(1, 2)  # (B, G, num_blocks)
                .reshape(doc_ids.shape[0], -1)  # (B, G * num_blocks)
            )
            doc_ids = torch.cat([doc_ids, strided_doc_ids], dim=1)

            sample_info = {
                "response_length": generated_tokens_count,
                "total_length": full_sequence.shape[1],
            }

            samples_list.append(Experience(
                prompts=original_prompt,
                full_sequences=full_sequence,
                action_mask=action_mask,
                info=sample_info,
                doc_ids=doc_ids,
                qa_masks=qa_mask,
            ))
        return samples_list



    @torch.no_grad()
    def _generate_for_downstream(self, all_prompts: List[str], all_labels: List[str], all_unit_tests: List[str] = None, all_entry_points: List[str] = None, all_code_contexts: List[str] = None, **kwargs) -> Tuple[torch.Tensor, List[Experience]]:
        """Generate parallel samples using multi-token prediction with stride.

        This method coordinates the generation process:
        1. Sends prompts to actors for strided generation
        2. Collects full sequences (Z) from actors
        3. Reconstructs full sequences and creates Experience objects

        Args:
            all_prompts: List of input prompts
            all_labels: List of labels corresponding to prompts
            all_unit_tests: Optional list of unit tests for code generation tasks
            **kwargs: Additional generation parameters (temperature, top_p, n_samples_per_prompt, etc.)

        Returns:
            List of Experience objects containing:
            - Original prompts (X): full original prompts
            - Full sequences: full prompt + generated tokens
            - Action masks for training
        """
        args = self.strategy.args
        batch_size = args.eval_down_batch_size
        if "generate_max_len" in kwargs:
            generation_length = kwargs["generate_max_len"]
        else:
            generation_length = self.args.eval_generate_max_len
        if 'n_samples_per_prompt' in kwargs:
            n_samples = kwargs['n_samples_per_prompt']
        else:
            n_samples = self.args.n_samples_per_prompt
        
        assert n_samples <= batch_size, "All eval samples from the same prompt must be in the same batch"
 

        if 'top_p' in kwargs:
            top_p = kwargs['top_p']
        else:
            top_p = self.args.top_p
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
        else:
            temperature = self.args.temperature

        actors = self.actor_model_group._actor_handlers
        if not actors:
            raise RuntimeError("No actors available in actor_model_group.")
        
        all_prompts = [prompt for prompt in all_prompts for _ in range(n_samples)]
        all_labels = [label for label in all_labels for _ in range(n_samples)]
        if all_unit_tests is not None:
            all_unit_tests = [unit_test for unit_test in all_unit_tests for _ in range(n_samples)]
        if all_entry_points is not None:
            all_entry_points = [entry_point for entry_point in all_entry_points for _ in range(n_samples)]
        if all_code_contexts is not None:
            all_code_contexts = [code_ctx for code_ctx in all_code_contexts for _ in range(n_samples)]
        batches = []
        labels = []
        unit_tests = []
        entry_points = []
        code_contexts = []
        for s in range(0, len(all_prompts), batch_size):
            chunk = pad_to_longest(all_prompts[s : s + batch_size], self.tokenizer)
            lbl = all_labels[s : s + batch_size]
            batches.append((s // batch_size, chunk))
            labels.append((s // batch_size, lbl))
            if all_unit_tests is not None:
                ut = all_unit_tests[s : s + batch_size]
                unit_tests.append((s // batch_size, ut))
            if all_entry_points is not None:
                ep = all_entry_points[s : s + batch_size]
                entry_points.append((s // batch_size, ep))
            if all_code_contexts is not None:
                code_ctx = all_code_contexts[s : s + batch_size]
                code_contexts.append((s // batch_size, code_ctx))

        # ---------- 3) Submit one RPC per batch (true async), round-robin across actors ----------
        inflight_refs = []  # List[Tuple[int, ObjectRef, List[int]]]
        for k, (bid, batch_2d) in enumerate(batches):
            a = actors[k % len(actors)]
            ref = a.generate_standard_ar.remote(
                prompts=batch_2d,
                generate_length=generation_length,
                temperature=temperature,
                top_p=top_p,
            )
            inflight_refs.append((bid, ref))

        # ---------- 4) Drain asynchronously; store outputs by batch id ----------
        id_by_ref = {ref: bid for (bid, ref) in inflight_refs}
        pending = [ref for (_, ref) in inflight_refs]
        results_by_bid = {}  # bid -> batch_out

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            r = ready[0]
            bid = id_by_ref[r]
            batch_out = ray.get(r)
            results_by_bid[bid] = batch_out
        # ---------- 5) Reassemble in submission order; flatten ----------
        ordered = [results_by_bid[i] for i in range(len(results_by_bid))]
        generated_sequences: List[torch.Tensor] = []
        for out in ordered:
            generated_sequences.append(out)
        # ---------- 6) Build Experience objects (aligned 1:1) ----------
        samples_list: List[Experience] = []
        for pos, generated_sequence in enumerate(generated_sequences):
            original_prompt = batches[pos][1]  # batched sample
            original_label = labels[pos][1]
            if all_unit_tests is not None:
                original_ut = unit_tests[pos][1]
            else:
                original_ut = None
            if all_entry_points is not None:
                original_ep = entry_points[pos][1]
            else:
                original_ep = None
            if all_code_contexts is not None:
                original_code_ctx = code_contexts[pos][1]
            else:
                original_code_ctx = None
            

            # Remove the prompt tokens from generated_sequence before decoding
            prompt_length = original_prompt['input_ids'].shape[1]  # Get prompt length
            generated_sequence_only = generated_sequence[:, prompt_length:]  # Strip prompt tokens
            generated_sequence = self.tokenizer.batch_decode(generated_sequence_only, skip_special_tokens=True)
            decoded_prompts = self.tokenizer.batch_decode(original_prompt['input_ids'], skip_special_tokens=True) 

            samples_list.append(Experience(
                generated_sequences=generated_sequence, #strings, generations only - no prompt
                prompt_strings=decoded_prompts, #tokenized and padded
                label_strings=original_label, # strs
                unit_tests=original_ut,
                entry_points=original_ep,
                code_contexts=original_code_ctx,
            ))
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
        self.tokenizer = tokenizer


    def assign_sample_indices(self, rollout_samples):
        """Assign sequential indices to samples for tracking through the pipeline."""
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        return rollout_samples

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples):
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        if self.critic_model_group is None:
            rollout_samples = self.assign_sample_indices(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(rollout_samples)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    def groom(self, gen_hidden_states, gen_qa_mask, qa_masking=False):
        # gen_hidden_states: (MB, NG, NS, NB, G, NF, H)
        # gen_qa_mask:       (MB, NG, NS, NB, G, NF, 1) with 1=answer, 0=question
        MB, NG, NS, NB, G, NF, H = gen_hidden_states.shape
        device = gen_hidden_states.device

        if not qa_masking:
            gen_qa_mask = torch.ones_like(gen_qa_mask)

        # (MB, NG, NS, NB, G, NF)
        mask = gen_qa_mask.squeeze(-1).to(torch.bool)

        # Timesteps indices along G
        time_idx = torch.arange(G, device=device).view(1, 1, 1, 1, G, 1).expand(MB, NG, NS, NB, G, NF)

        # Set non-answer to -1, take max over G to get last answer index; -1 means none
        last_idx = time_idx.masked_fill(~mask, -1).amax(dim=4)  # (MB, NG, NS, NB, NF)

        # Build gather indices on dim=4 (time)
        safe_idx = last_idx.clamp_min(0).unsqueeze(4).unsqueeze(-1).expand(MB, NG, NS, NB, 1, NF, H)
        out = gen_hidden_states.gather(dim=4, index=safe_idx)  # (MB, NG, NS, NB, 1, NF, H)

        # Zero where no answer existed
        no_ans = last_idx.eq(-1).unsqueeze(4).unsqueeze(-1).expand_as(out)
        out = out.masked_fill(no_ans, 0.0)

        return out
    
    @torch.no_grad()
    def make_experience(self, samples_list):
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.full_sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        n_samples = self.args.n_samples_per_prompt
        stride = self.args.stride
        context_length = self.args.context_max_len
        generate_length = self.args.generate_max_len
        
        hidden_state_method = self.args.hidden_state_method
        

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        prompts_list = [s.prompts for s in samples_list]
        full_sequences_list = [s.full_sequences for s in samples_list]
        action_mask_list = [torch.ones_like(s.action_mask) for s in samples_list]

        prompt_strings_list = [s.prompt_strings for s in samples_list]

        # Calculate sequence lengths and blocks
        prompt_length = prompts_list[0].shape[1]  # Length of original prompt
        num_blocks = (prompt_length - generate_length - context_length) // stride + 1  # Number of prediction blocks
        doc_ids_list = [s.doc_ids[:prompt_length] if s.doc_ids.dim() == 1 else s.doc_ids[:,:prompt_length] for s in samples_list]
        qa_masks_list = [s.qa_masks for s in samples_list]

        log_token_match_metrics = bool(getattr(self.args, "log_token_match_metrics", False))
        token_match_flat = None
        if log_token_match_metrics:
            gen_tokens_diag, gt_tokens_diag = prepare_tensors_for_embedding(
                prompts_list,
                full_sequences_list,
                prompt_length,
                stride,
                num_blocks,
                n_samples,
                context_length=context_length,
                gen_len=generate_length,
            )
            # (num_micro_batches, num_groups, n_samples, num_blocks)
            token_match_block = (gen_tokens_diag == gt_tokens_diag).float().mean(dim=-1)
            # Flatten to match per-sample layout used for rewards tensors (B = num_groups * n_samples)
            token_match_flat = token_match_block.reshape(
                token_match_block.shape[0], -1, token_match_block.shape[-1]
            )
        
        critic_hidden_states_ref = self.critic_model_group.async_run_method_batch(
            method_name="forward",
            sequences=full_sequences_list,
            prompt_length=[prompt_length] * len(full_sequences_list),
            context_length=[context_length] * len(full_sequences_list),
            generate_max_len=[generate_length] * len(full_sequences_list),
            stride=[stride] * len(full_sequences_list),
            num_blocks=[num_blocks] * len(full_sequences_list),
            hidden_state_method=[hidden_state_method] * len(full_sequences_list),
            doc_ids=doc_ids_list,
            document_masking = [self.args.document_masking] * len(full_sequences_list),
            qa_masks=qa_masks_list,
            qa_masking=[self.args.qa_masking] * len(full_sequences_list),
        )

 

        ray.get(critic_hidden_states_ref)
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        critic_hidden_states_list = sum(ray.get(critic_hidden_states_ref)[::duplicate_factor], []) # list of num_batches tensors of shape (batch_size, full_sequence_length,hidden_size)

        critic_hidden_states_tensor = torch.stack([entry[0] for entry in critic_hidden_states_list])

        gt_embedding = critic_hidden_states_tensor[:,:, context_length:prompt_length, :, :] # (num_batches, batch_size, prompt_length, num_feat, hidden_size)
        gen_embedding = critic_hidden_states_tensor[:,:, prompt_length:, :, :] # (num_batches, batch_size, generate_length, num_feat, hidden_size)
        qa_masks_tensor = torch.stack(qa_masks_list)
        gt_qa_mask = qa_masks_tensor[:, :, context_length:prompt_length].view(gt_embedding.shape[0], gt_embedding.shape[1], gt_embedding.shape[2], 1, 1).repeat(1, 1, 1, gt_embedding.shape[3], 1)
        gen_qa_mask = qa_masks_tensor[:, :, prompt_length:].view(gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2], 1, 1).repeat(1, 1, 1, gen_embedding.shape[3], 1)

        gt_embedding = gt_embedding.unfold(-3, generate_length, stride).permute(0, 1, 2, 5, 3, 4)
        gen_embedding = gen_embedding.reshape(gen_embedding.shape[0], gen_embedding.shape[1], generate_length, num_blocks, gen_embedding.shape[-2], gen_embedding.shape[-1]).transpose(-3,-4) # (num_micro_batches, batch_size, num_blocks, generate_length, hidden_size)
        gt_qa_mask = gt_qa_mask.unfold(-3, generate_length, stride).permute(0, 1, 2, 5, 3, 4)
        gen_qa_mask = gen_qa_mask.reshape(gen_embedding.shape[0], gen_embedding.shape[1], generate_length, num_blocks, gen_embedding.shape[-2], 1).transpose(-3,-4)
        if getattr(self.args, "debug", False):
            logger.info(
                f'gt_embedding shape after reshape: {gt_embedding.shape}, gen_embedding shape after reshape: {gen_embedding.shape}, '
                f'gt_qa_mask shape: {gt_qa_mask.shape}, gen_qa_mask shape: {gen_qa_mask.shape}'
            )
        num_micro_batches = gt_embedding.shape[0]
        num_groups = gt_embedding.shape[1] // n_samples # num groups per micro batch
        num_feat = gt_embedding.shape[-2]
        gt_embedding = gt_embedding.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, gt_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, num_feat, hidden_size)
        gen_embedding = gen_embedding.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, gen_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, num_feat, hidden_size)
        gt_qa_mask = gt_qa_mask.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, 1) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, num_feat, 1)
        gen_qa_mask = gen_qa_mask.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, 1) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, num_feat, 1)
        if getattr(self.args, "debug", False):
            logger.info(
                f'gt_embedding shape after reshape into n_samples per prompt: {gt_embedding.shape}, '
                f'gen_embedding shape after reshape into n_samples per prompt: {gen_embedding.shape}'
            )
            logger.info(f'torch.mean(torch.norm(gen_embedding, dim=-1)): {torch.mean(torch.norm(gen_embedding, dim=-1))}')
            logger.info(f'torch.mean(torch.norm(gt_embedding, dim=-1)): {torch.mean(torch.norm(gt_embedding, dim=-1))}')
        if self.args.embed_method == "mean_pooling":
            gt_embedding = torch.mean(gt_embedding, dim=-3, keepdim=True) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, um1, num_feat, hidden_size)
            gen_embedding = torch.mean(gen_embedding, dim=-3, keepdim=True) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, hidden_size)
        elif self.args.embed_method == "last_token":
            gt_embedding = self.groom(gt_embedding, gt_qa_mask, qa_masking=self.args.qa_masking)
            gen_embedding = self.groom(gen_embedding, gen_qa_mask, qa_masking=self.args.qa_masking)
        elif self.args.embed_method == "concat":
            gt_embedding = gt_embedding.transpose(-2, -3).reshape(num_micro_batches, num_groups, n_samples, num_blocks, 1, num_feat, generate_length * gt_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, generate_length * hidden_size)
            gen_embedding = gen_embedding.transpose(-2, -3).reshape(num_micro_batches, num_groups, n_samples, num_blocks, 1, num_feat, generate_length * gen_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, generate_length * hidden_size)
        elif self.args.embed_method == "token":
            # shape of gt_embedding and gen_embedding should be (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, n hidden_size)
            pass # already have token-level embeddings
        else:
            raise ValueError(f"Unknown embed_method: {self.args.embed_method}")
        
        if self.args.use_whitening:
            gen_embedding, gt_embedding = whiten_embeddings_batched(gen_embedding, gt_embedding, whiten_tol=1e-5, normalize=False)
            
        gt_embedding = gt_embedding.reshape(
            gt_embedding.shape[0], gt_embedding.shape[1], gt_embedding.shape[2],
            gt_embedding.shape[3], gt_embedding.shape[4], gt_embedding.shape[5] * gt_embedding.shape[6]
        )
        gen_embedding = gen_embedding.reshape(
            gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2],
            gen_embedding.shape[3], gen_embedding.shape[4], gen_embedding.shape[5] * gen_embedding.shape[6]
        )

        if self.args.embed_method != "token":
            gt_embedding = gt_embedding.squeeze(-2) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, num_features)
            gen_embedding = gen_embedding.squeeze(-2) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, num_features)

        if self.args.embed_method == "token":
            per_token = True
        else:
            per_token = False
        # ed or cosine
        gt_rewards_tensor = get_alignment_rewards(gen_embedding, gt_embedding)
        diversity_rewards_tensor = get_diversity_rewards(gen_embedding, per_token)
        if per_token:
            # token-level rewards
            gt_rewards_tensor = gt_rewards_tensor.reshape(gt_rewards_tensor.shape[0], -1, gt_rewards_tensor.shape[-2], gt_rewards_tensor.shape[-1])
            diversity_rewards_tensor = diversity_rewards_tensor.reshape(diversity_rewards_tensor.shape[0], -1, diversity_rewards_tensor.shape[-2], diversity_rewards_tensor.shape[-1])
        else:
            gt_rewards_tensor = gt_rewards_tensor.reshape(gt_rewards_tensor.shape[0], -1, gt_rewards_tensor.shape[-1])
            diversity_rewards_tensor = diversity_rewards_tensor.reshape(diversity_rewards_tensor.shape[0], -1, diversity_rewards_tensor.shape[-1])
        # NOTE: We keep the *raw* component tensors (gt/diversity/critic) for logging,
        # and apply scalar coefficients only when forming the final reward used for advantages.
        gt_rewards_tensor *= 2
        diversity_rewards_tensor *= 2

        if getattr(self.args, "debug", False):
            logger.info(
                f'gt_rewards_tensor shape: {gt_rewards_tensor.shape}, '
                f'diversity_rewards_tensor shape: {diversity_rewards_tensor.shape}'
            )
        # Final reward used for advantages (coherent with config knobs):
        #   reward = alignment_rew_coef * alignment
        #          - diversity_rew_coef      * diversity
        rewards_tensor = self.args.alignment_rew_coef * gt_rewards_tensor - self.args.diversity_rew_coef * diversity_rewards_tensor
        
        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward_strided_blocks",
                full_sequences=full_sequences_list,
                action_mask=action_mask_list,
                prompt_length=[prompt_length] * len(full_sequences_list),
                generation_step=[generate_length] * len(full_sequences_list), 
                num_blocks=[num_blocks] * len(full_sequences_list),
                stride=[stride] * len(full_sequences_list),
                context_length=[context_length] * len(full_sequences_list),
                doc_ids=doc_ids_list,
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
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        
        # Assign rewards to each sample
        for i, samples in enumerate(samples_list):
            samples.rewards = rewards_tensor[i].clone().detach()
            samples.diversity_rewards = diversity_rewards_tensor[i].clone().detach()
            samples.gt_rewards = gt_rewards_tensor[i].clone().detach()
            tokmatch_i = token_match_flat[i] if token_match_flat is not None else None  # (batch_size, num_blocks)

            effective_rewards = self.args.alignment_rew_coef * samples.gt_rewards - self.args.diversity_rew_coef * samples.diversity_rewards / 2

            feature_map_reward = samples.gt_rewards - samples.diversity_rewards / 2

            samples.info["reward"] = samples.rewards.mean().detach().cpu().item()
            samples.info["effective_reward"] = effective_rewards.mean().detach().cpu().item()
            samples.info["diversity_reward"] = samples.diversity_rewards.mean().detach().cpu().item()
            samples.info["gt_reward"] = samples.gt_rewards.mean().detach().cpu().item()
            samples.info["feature_map_reward"] = feature_map_reward.mean().detach().cpu().item()
            if tokmatch_i is not None:
                samples.info["tokmatch"] = tokmatch_i.mean().detach().cpu().item()

            reshaped_rewards = samples.rewards
            if n_samples > 1:
                std_rewards = reshaped_rewards.reshape(-1, self.args.n_samples_per_prompt, samples.rewards.shape[-1]).std(1).detach()
                num_zeros_rewards = (samples.gt_rewards.reshape(-1, self.args.n_samples_per_prompt, samples.rewards.shape[-1]).mean(1)==0).float().mean().detach()

                num_elements = std_rewards.numel() 
                samples.info["std_reward"] = std_rewards.mean().cpu().item()
                samples.info["zero_std_reward"] = (std_rewards == 0).float().sum().cpu().item() / num_elements
                samples.info["num_zeros_rewards"] = num_zeros_rewards.cpu().item()

                gt_rewards_raw = samples.gt_rewards
                if gt_rewards_raw.dim() > 2:
                    # token-level rewards -> average over token dimension
                    gt_rewards_raw = gt_rewards_raw.mean(dim=-1)
                # (num_groups, n_samples, num_blocks)
                gt_reward_g = gt_rewards_raw.reshape(-1, self.args.n_samples_per_prompt, gt_rewards_raw.shape[-1])
                top2 = torch.topk(gt_reward_g, k=min(2, self.args.n_samples_per_prompt), dim=1).values
                gt_top1 = top2[:, 0, :]
                gt_top2 = top2[:, 1, :] if top2.shape[1] > 1 else gt_top1
                samples.info["gt_reward_top1"] = gt_top1.mean().cpu().item()
                samples.info["gt_reward_gap_top1_top2"] = (gt_top1 - gt_top2).mean().cpu().item()

                # ------------------------------------------------------------
                # Optional lexical ranking diagnostics (off by default)
                # ------------------------------------------------------------
                if tokmatch_i is None:
                    continue

                tokmatch_g = tokmatch_i.reshape(-1, self.args.n_samples_per_prompt, tokmatch_i.shape[-1])

                # Argmax agreement (top-1 selection quality)
                r_top1 = gt_reward_g.argmax(dim=1)   # (num_groups, num_blocks)
                t_top1 = tokmatch_g.argmax(dim=1)    # (num_groups, num_blocks)
                samples.info["reward_tokmatch_top1_acc"] = (r_top1 == t_top1).float().mean().cpu().item()

                # Token-match achieved by selecting the sample with max reward
                tokmatch_of_r_top1 = tokmatch_g.gather(dim=1, index=r_top1.unsqueeze(1)).squeeze(1)  # (num_groups, num_blocks)
                samples.info["tokmatch_of_reward_top1"] = tokmatch_of_r_top1.mean().cpu().item()

                # Oracle token-match (best possible among samples), and regret
                tokmatch_oracle = tokmatch_g.max(dim=1).values  # (num_groups, num_blocks)
                samples.info["tokmatch_top1_oracle"] = tokmatch_oracle.mean().cpu().item()
                samples.info["tokmatch_regret"] = (tokmatch_oracle - tokmatch_of_r_top1).mean().cpu().item()



             
        assert (
            len(samples_list) == len(base_action_log_probs_list)
        ), f"len(samples_list): {len(samples_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}"

        # Process results for each sample
        for i, (samples, base_action_log_probs) in enumerate(
            zip(samples_list, base_action_log_probs_list)
        ):

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.base_action_log_probs = base_action_log_probs


        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list


    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> List[Experience]:
        """
        Compute advantages and returns for PPO training from raw rewards.

        Processes block-level rewards from multi-token prediction by:
        - Reordering and reshaping rewards for multiple samples per prompt
        - Applying reward shaping with baselines (RLOO or group norm)
        - Expanding block rewards to token-level for the full sequence
        - Computing advantages/returns and normalizing across batch

        Args:
            experiences: List of Experience objects with rewards
            **kwargs: Additional arguments (unused)

        Returns:
            Experiences updated with advantages and returns
        """
        args = self.strategy.args
        full_sequence_length = experiences[0].full_sequences.shape[1]


        # Get sample indices for reordering (identity mapping unless using dynamic batching)
        sample_indices = torch.tensor(sum([experience.index for experience in experiences], []))

        # Extract and reorder all reward types
        raw_rewards = extract_and_reorder_rewards(experiences, 'rewards', sample_indices) # (num_batches, batch_size/nsamp, nsamp, num_blocks)
        raw_diversity_rewards = extract_and_reorder_rewards(experiences, 'diversity_rewards', sample_indices)
        raw_gt_rewards = extract_and_reorder_rewards(experiences, 'gt_rewards', sample_indices)

        # Verify that generations for the same prompt are consecutive
        # This is critical for correct baseline computation in RLOO
        n_samples = args.n_samples_per_prompt
        if n_samples > 1:
            # Each Experience contains batch_size samples
            # Within each Experience, verify consecutive n_samples share the same prompt
            for exp_idx, experience in enumerate(experiences):
                batch_size = experience.prompts.shape[0]
                # Check consecutive groups of n_samples within this batch
                for i in range(0, batch_size, n_samples):
                    # if i + n_samples <= batch_size:
                    # Compare consecutive samples in groups of n_samples
                    first_prompt = experience.prompts[i]
                    for j in range(1, n_samples):
                        sample_prompt = experience.prompts[i + j]
                        if not torch.equal(first_prompt, sample_prompt):
                            raise ValueError(
                                f"Experiences are not properly ordered: samples from the same prompt must be consecutive. "
                                f"Found mismatch in experience[{exp_idx}] at batch position {i}, sample {j}. "
                                f"Expected {n_samples} consecutive samples per prompt."
                            )
        
        
        

        # Apply reward shaping based on the selected advantage estimator
        if args.advantage_estimator == "rloo":
            # RLOO: subtract leave-one-out baseline for variance reduction
            # (num_batches, batch_size, num_blocks)
            baseline = self.compute_baseline(raw_diversity_rewards, raw_gt_rewards, args.n_samples_per_prompt)
            shaped_rewards = raw_rewards - baseline #(num_batches, batch_size/nsamp, nsamp, num_blocks)

        elif args.advantage_estimator in ["reinforce"]:
            shaped_rewards = raw_rewards

        elif args.advantage_estimator == "group_norm":
            # Group normalization: subtract baseline and normalize by standard deviation
            baseline = self.compute_baseline(raw_diversity_rewards, raw_gt_rewards, args.n_samples_per_prompt)
            shaped_rewards = (raw_rewards - baseline) / (raw_rewards.std(-1, keepdim=True) + 1e-9)

        reward_list = list(shaped_rewards)

        # Verify dimensions match
        assert len(experiences) == len(reward_list), f"Number of experiences ({len(experiences)}) doesn't match number of rewards ({len(reward_list)})"


        # Calculate returns and advantages for each experience
        for experience, shaped_reward in zip(experiences, reward_list):

            # Expand rewards to cover all generated tokens
            # Each block's reward applies to all tokens it generated
            if shaped_reward.dim() == 2:
                expanded_rewards = shaped_reward.repeat(1, self.args.generate_max_len)
            elif shaped_reward.dim() == 3:
                expanded_rewards = shaped_reward.view(shaped_reward.shape[0], -1)

            # Create full sequence reward tensor
            # Prompt tokens get zero reward, generated tokens get their block's reward
            full_sequence_rewards = torch.zeros(shaped_reward.shape[0], full_sequence_length - 1, device=expanded_rewards.device)

            # the generations should be weighted by their reward
            # [p0 p1 p2 p3 g0 g1 g2 g3]
            # [1  1  1  1  r0 r1 r2 r3]
            full_sequence_rewards[:, -expanded_rewards.shape[-1]:] = expanded_rewards

            # Compute advantages and returns based on the estimator type
            if self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                # REINFORCE variants: use rewards directly as returns
                if args.gamma != 1.0 and self.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = full_sequence_rewards
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unknown advantage_estimator: {self.advantage_estimator}")

            # Store total return for logging
            experience.info["return"] = full_sequence_rewards.sum(dim=-1)
            
            # remove unnecessary info
            experience.kl = None

        return experiences
    
    @torch.no_grad()
    def compute_baseline(self, diversity_rewards, gt_rewards, n_samples_per_prompt):
        """
        Compute baseline for variance reduction in REINFORCE-style algorithms.

        The baseline uses leave-one-out estimation for gt rewards and
        a special formula for diversity rewards to reduce variance.

        Args:
            diversity_rewards: diversity reward tensor (n_batches, bsz, n_blocks)
            gt_rewards: gt reward tensor
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            Combined baseline tensor
        """
        # diversity_rewards is (num_batches, batch_size/nsamp, nsamp, num_blocks)
        # gt_rewards is (num_batches, batch_size/nsamp, nsamp, num_blocks)
        original_diversity_rewards_shape = diversity_rewards.shape

        if len(diversity_rewards.shape) == 3:
            diversity_rewards = diversity_rewards.reshape(diversity_rewards.shape[0], -1, n_samples_per_prompt, diversity_rewards.shape[-1])
            gt_rewards = gt_rewards.reshape(gt_rewards.shape[0], -1, n_samples_per_prompt, gt_rewards.shape[-1])
        else:
            orig_n_batches, orig_bsz, orig_n_blocks, seq_len = diversity_rewards.shape
            diversity_rewards = diversity_rewards.reshape(diversity_rewards.shape[0], -1, n_samples_per_prompt, diversity_rewards.shape[-2], seq_len)
            gt_rewards = gt_rewards.reshape(gt_rewards.shape[0], -1, n_samples_per_prompt, gt_rewards.shape[-2], seq_len)

        # Baseline must match the same scalar reward definition used for `samples.rewards`:
        #   reward = alignment_rew_coef * gt_reward
        #          - diversity_rew_coef      * diversity_reward
        #
        # Because baseline is linear, we can combine component baselines with the same coefficients.
        if n_samples_per_prompt <= 1:
            return torch.zeros_like(diversity_rewards).reshape(original_diversity_rewards_shape)

        denom_loo = float(n_samples_per_prompt - 1)

        sum_gt_rewards = gt_rewards.sum(2, keepdim=True)
        gt_baseline = (sum_gt_rewards - gt_rewards) / denom_loo

        alignment_rew_coef = float(getattr(self.args, "alignment_rew_coef", 1.0))
        diversity_rew_coef = float(getattr(self.args, "diversity_rew_coef", 0.0))

        # Diversity baseline formula needs n_samples_per_prompt > 2 (it uses n-2).
        if diversity_rew_coef != 0.0 and n_samples_per_prompt > 2:
            denom_div = float(n_samples_per_prompt - 2)
            sum_diversity_rewards = diversity_rewards.sum(2, keepdim=True)
            diversity_baseline = sum_diversity_rewards / denom_div - (2.0 * diversity_rewards) / denom_div
        else:
            diversity_baseline = torch.zeros_like(diversity_rewards)

        baseline = (
            alignment_rew_coef * gt_baseline
            - diversity_rew_coef * diversity_baseline
        )
        return baseline.reshape(original_diversity_rewards_shape)

    @torch.no_grad()
    def make_ppls_experience_batch(self, rollout_samples):
        """
        Build inputs needed to compute perplexities using ground-truth completions.

        Returns a dict with per-rank accumulated NLL sums and token counts for
        full sequence tokens (matches compute_perplexity on pretrain-style loss).
        """
        # Prepare tensors for GT sequences and masks (evaluation naming to avoid confusion)
        build = self.make_ppl_experience(rollout_samples)
        eval_sequences_list       = build["eval_sequences"]        # list of [B, S]
        eval_attention_mask_list  = build["eval_attention_masks"]  # list of [B, S]
        eval_loss_masks_list      = build["eval_loss_masks"]       # list of [B, S-1]
        prompt_lens = build["prompt_len"]                     # list of int

        args = self.strategy.args
        eval_full_ppl_mode_list = [True] * len(eval_sequences_list)

        # Query actor for log probs on full-sequence mask (B, S-1) logprobs
        full_seq_log_probs = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=eval_sequences_list,
            action_mask=eval_loss_masks_list,
            attention_mask=eval_attention_mask_list,
            eval_full_ppl_mode=eval_full_ppl_mode_list,
            prompt_lens=prompt_lens,
        )

        # Sync to avoid GPU OOM when colocating models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(full_seq_log_probs)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Handle duplicated returns from ring/tp groups (keep one copy)
        duplicate_factor   = args.ring_attn_size * args.ds_tensor_parallel_size
        full_seq_log_probs_list = sum(ray.get(full_seq_log_probs)[::duplicate_factor], [])


        # Accumulate local sums in float64 for stability (on CPU to save GPU mem)
        device        = torch.device("cpu")
        full_ce_loss_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        full_ce_token_sum  = torch.tensor(0.0, dtype=torch.float64, device=device)
        for full_log_probs, full_loss_mask in zip(full_seq_log_probs_list, eval_loss_masks_list):

            # Negative log likelihood sums over selected tokens
            full_ce_loss_sum += (-full_log_probs * full_loss_mask).sum()
            full_ce_token_sum  += full_loss_mask.sum()

        return {
            "full_ce_loss_sum": full_ce_loss_sum,
            "full_ce_token_sum":  full_ce_token_sum,
            # "chunk_ppls": chunk_ppl_avg,
        }

    @torch.no_grad()
    def make_squared_loss_experience_batch(self, rollout_samples):
        """
        Build inputs needed to compute perplexities using ground-truth completions.

        Returns a dict with per-rank accumulated NLL sums and token counts for
        full sequence tokens (matches compute_perplexity on pretrain-style loss).
        """
        # Prepare tensors for GT sequences and masks (evaluation naming to avoid confusion)
        build = self.make_ppl_experience(rollout_samples)
        eval_sequences_list       = build["eval_sequences"]        # list of [B, S]
        eval_attention_mask_list  = build["eval_attention_masks"]  # list of [B, S]
        eval_loss_masks_list      = build["eval_loss_masks"]       # list of [B, S-1]
        prompt_lens = build["prompt_len"]                     # list of int

        args = self.strategy.args
        eval_full_ppl_mode_list = [True] * len(eval_sequences_list)

        # Query actor for log probs on full-sequence mask (B, S-1) logprobs
        full_squared_error = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=eval_sequences_list,
            action_mask=eval_loss_masks_list,
            attention_mask=eval_attention_mask_list,
            eval_full_ppl_mode=eval_full_ppl_mode_list,
            return_squared_loss=[True] * len(eval_sequences_list),
            prompt_lens=prompt_lens,
        )

        # Sync to avoid GPU OOM when colocating models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(full_squared_error)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Handle duplicated returns from ring/tp groups (keep one copy)
        duplicate_factor   = args.ring_attn_size * args.ds_tensor_parallel_size
        full_squared_error_list = sum(ray.get(full_squared_error)[::duplicate_factor], [])


        # Accumulate local sums in float64 for stability (on CPU to save GPU mem)
        device        = torch.device("cpu")
        mean_squared_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

        for full_squared_error in full_squared_error_list:
            # each batch returns one mean value for batch. all same number of tokens so this is ok
            mean_squared_loss += full_squared_error
        mean_squared_loss = mean_squared_loss / len(full_squared_error_list)
        return {
            "mse": mean_squared_loss,
            # "chunk_ppls": chunk_ppl_avg,
        }

    def make_ppl_experience(self, samples_list: List[Experience]):
        """
        Prepare per-microbatch tensors for ground-truth sequences and corresponding masks
        needed for full-sequence PPL computation.

        Returns a dict with lists of tensors: sequences [B,S], attention_masks [B,S],
        loss_masks [B,S-1].
        """
        start_time = time.time()

        # Calculate total number of samples for logging
        total_samples = sum([
            len(s.prompts) if isinstance(s.prompts, torch.Tensor) and s.prompts.dim() > 0
            else 1 for s in samples_list
        ])
        logger.info(f"🚀 Preparing PPL inputs with {total_samples} samples")

        eval_sequences_list = []
        eval_attention_mask_list = []
        eval_loss_masks_list = []
        prompt_len_list = []

        for sample in samples_list:
            # Use prompts field for perplexity evaluation (full original sequences)
            sequences = sample.prompts  # Using new field name

            # Create attention mask (all ones for valid tokens)
            attention_mask = torch.ones_like(sequences, dtype=torch.long)

            # Get batch and sequence dimensions
            batch_size, seq_length = sequences.shape

            # Convert to appropriate dtype
            eval_seq_batch = sequences.to(torch.long)
            eval_att_batch = attention_mask.to(torch.long)

            # Create loss mask for next-token prediction (length is seq_length - 1)
            eval_mask_batch = torch.ones(batch_size, seq_length - 1, dtype=torch.long)

 
            # Sanity: masks must be one shorter than sequences along last dim
            assert eval_mask_batch.shape[1] == eval_seq_batch.shape[1] - 1, \
                f"full mask shape {eval_mask_batch.shape} not S-1 for seq {eval_seq_batch.shape}"

            eval_sequences_list.append(eval_seq_batch)
            eval_attention_mask_list.append(eval_att_batch)
            eval_loss_masks_list.append(eval_mask_batch)
            prompt_len_list.append(sequences.shape[1])

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ PPL input preparation completed in {time_str}")

        return {
            "eval_sequences":        eval_sequences_list,
            "eval_attention_masks":  eval_attention_mask_list,
            "eval_loss_masks":       eval_loss_masks_list,
            "prompt_len": prompt_len_list,
        }

