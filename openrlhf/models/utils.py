from typing import Optional, Tuple, Union

import os
import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    # Pure task reward only (no KL shaping). Broadcast scalar per-sample reward
    # across all generated positions indicated by action_mask.
    #
    # Inputs:
    # - r: shape [B] or [B, 1], per-sample scalar reward (e.g., 1 - normalized_edit)
    # - action_mask: shape [B, T_resp], 1 on generated token positions
    #
    # Output:
    # - reward_per_token: shape [B, T_resp], same scalar reward on active positions

    assert action_mask is not None, "compute_reward requires action_mask"

    # Normalize r to [B, 1]
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=torch.float32, device=action_mask.device)
    if r.dim() == 1:
        r = r.unsqueeze(1)
    elif r.dim() == 2 and r.size(1) == 1:
        # already [B,1]
        pass
    else:
        # Flatten any extra dims to [B,1]
        r = r.view(r.size(0), -1)[:, :1]

    # Optional clipping on the scalar reward
    if reward_clip_range is not None:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    reward = r.to(dtype=torch.float32, device=action_mask.device) * action_mask.to(dtype=torch.float32, device=action_mask.device)
    return reward



def build_strided_attention_mask_and_positions(
        full_sequence_length: int,
        prompt_length: int,
        context_length: int,
        generation_step: int,
        max_generation_length: int,
        stride: int,
        num_blocks: int,
        device: torch.device,
        doc_ids: torch.Tensor,
        document_masking: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build attention mask and position IDs for strided parallel generation.

    This function creates a specialized attention mask that enables parallel token generation
    across multiple stride-offset blocks. Each block predicts tokens using a different
    context window from the original prompt.

    Args:
        full_sequence_length: Total length of full sequence (prompt + generated tokens)
        prompt_length: Length of the original prompt sequence
        context_length: Length of context used for each block
        generation_step: Number of generation steps completed so far
        max_generation_length: Total number of tokens to generate
        stride: Number of tokens offset between consecutive blocks
        device: Device to create tensors on
        num_blocks: Number of prediction blocks

    Returns:
        attention_mask: Shape [1, 1, full_sequence_length, full_sequence_length]
                       Values are 0 for allowed attention, -inf for masked positions
        position_ids: Shape [1, full_sequence_length]
                     Position embeddings for each token in the sequence
    """

    min_value = torch.finfo(dtype).min
    attention_mask = torch.full((doc_ids.shape[0],1,full_sequence_length, full_sequence_length), min_value, dtype=dtype, device=device)

    doc_ids = doc_ids.to(device)

    # We want to create a mask of shape (B, 1, S, S) where mask[b, 0, i, j] is true
    # if token i and token j in batch item b belong to the same document.
    # (B, S) → compare every pair of positions → (B, S, S) → add head dim → (B, 1, S, S)
    same_doc_mask = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
    same_doc_mask = same_doc_mask.unsqueeze(1)  # (B, 1, S, S)
    if bool(int(os.environ.get("OPENRLHF_DEBUG_MASKS", "0"))):
        print(f'same_doc_mask shape: {same_doc_mask.shape}')
    if not document_masking:
        same_doc_mask = same_doc_mask.fill_(True)

    # 1) Standard causal mask over the prompt (L x L)
    causal_mask = torch.tril(
        torch.ones((prompt_length, prompt_length), dtype=torch.bool, device=device)
    )

    # 2) Reshape to (1, 1, L, L) so it can broadcast over the batch dimension
    causal_mask = causal_mask.view(1, 1, prompt_length, prompt_length)

    # 3) Restrict same_doc_mask to the prompt region: (B, 1, L, L)
    prompt_same_doc_mask = same_doc_mask[:, :, :prompt_length, :prompt_length]

    # 4) Allowed positions = causal AND same-document
    prompt_allowed = causal_mask & prompt_same_doc_mask  # (B, 1, L, L)

    # 5) attention_mask is already filled with min_value, so just open the allowed entries
    attention_mask[:, :, :prompt_length, :prompt_length].masked_fill_(prompt_allowed, 0.0)

    # Part 2: Set up attention patterns for generated tokens.
    # For each generated token at full-sequence position (prompt_length + gen_step*num_blocks + block_idx),
    # open attention to: (1) its context window in the prompt, (2) itself, (3) earlier same-block tokens.
    for gen_step in range(generation_step):
        for block_idx in range(num_blocks):
            # Calculate position of this generated token in the full sequence
            generated_token_position = prompt_length + gen_step * num_blocks + block_idx

            # Allow this token to attend to its designated context window
            # Block k sees the first (k+1)*stride prompt tokens
            context_window_end = min(block_idx * stride + context_length, prompt_length - max_generation_length)

            # Get the document that the first ground truth token belongs to. This is the document associated with the generation at this given anchor
            cur_doc_ids = doc_ids[:,context_window_end+gen_step].unsqueeze(-1) # shape(B,1)

            # So far we have been taking all preceding tokens as context. we want to mask out those that are not from the same doc as cur_doc_ids
            # Get the document ids for all tokens preceding. (B, context_window_end)
            context_doc_ids = doc_ids[:, :context_window_end]  # Shape: (B, context_window_end)
            # Only attend to tokens with the same doc_id as cur_doc_ids. Get the indices
            context_same_doc_idx = context_doc_ids == cur_doc_ids  # Shape: (B, context_window_end)
            if not document_masking:
                context_same_doc_idx = context_same_doc_idx.fill_(True)
            batch_idx = torch.arange(doc_ids.shape[0], device=device).unsqueeze(-1).expand_as(context_same_doc_idx)

            row = attention_mask[:, 0, generated_token_position, :context_window_end]
            row[context_same_doc_idx] = 0.0

            # Allow token to attend to itself (necessary for proper computation)
            attention_mask[:, 0, generated_token_position, generated_token_position] = 0.0

            if gen_step > 0:
                # Prompt index that this generated token is anchored to
                cur_anchor_idx = context_window_end + gen_step               # scalar
                cur_doc = doc_ids[:, cur_anchor_idx]                         # (B,)

                for prev_s in range(gen_step):
                    # Prompt index for the previous generation step in the same block
                    prev_anchor_idx = context_window_end + prev_s            # scalar

                    # Full-sequence index for that previous generated token
                    prev_pos = (
                        prompt_length
                        + prev_s * num_blocks
                        + block_idx
                    )  # scalar in [prompt_length, full_sequence_length)

                    # Only allow attention if the docs match
                    same_doc_prev = (doc_ids[:, prev_anchor_idx] == cur_doc)  # (B,)
                    if not document_masking:
                        same_doc_prev = same_doc_prev.fill_(True)

                    # Open attention: (b, 0, generated_token_position, prev_pos) for those batches
                    attention_mask[same_doc_prev, 0, generated_token_position, prev_pos] = 0.0

    if document_masking:
        boundaries = torch.zeros_like(doc_ids, dtype=torch.bool)
        boundaries[:, 0] = True
        boundaries[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]

        B, S = doc_ids.shape

        # 1) Absolute positions 0..S-1
        global_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # (B, S)

        # 2) For each boundary, store its index; 0 elsewhere
        segment_start_pos = global_pos.masked_fill(~boundaries, 0)  # (B, S)

        # 3) Propagate the last seen doc-boundary index forward so every token knows its doc start position.
        segment_start_pos, _ = torch.cummax(segment_start_pos, dim=1)  # (B, S)

        # 4) Position within each doc = global index - start index of its doc
        pos_ids = global_pos - segment_start_pos  # (B, S)

        # Build position IDs for the full sequence
        position_ids = torch.empty((doc_ids.shape[0], full_sequence_length), dtype=torch.long, device=device)

        # Prompt tokens use standard sequential position IDs
        position_ids[:, :prompt_length] = pos_ids

        # Generated tokens use position IDs that continue from their context windows
        # Each block continues from the last position it can see in the prompt
        block_starting_idx = (torch.arange(0, num_blocks, device=device) * stride) + context_length
        block_starting_positions = position_ids[:, block_starting_idx] # Shape: (B, num_blocks)

        for gen_step in range(generation_step):
            # Calculate where this step's generated tokens start in the full sequence
            step_start_idx = prompt_length + gen_step * num_blocks  
            step_end_idx = step_start_idx + num_blocks

            # Each block's tokens get sequential position IDs starting from their context end
            position_ids[:, step_start_idx:step_end_idx] = block_starting_positions + gen_step

    else:
        B, _ = doc_ids.shape

        # Build position IDs for the full sequence
        position_ids = torch.empty((B, full_sequence_length), dtype=torch.long, device=device)

        # Prompt tokens use standard sequential position IDs
        position_ids[:, :prompt_length] = torch.arange(prompt_length, device=device)

        # Generated tokens use position IDs that continue from their context windows
        # Each block continues from the last position it can see in the prompt
        block_starting_positions = (torch.arange(0, num_blocks, device=device) * stride) + context_length

        for gen_step in range(generation_step):
            # Calculate where this step's generated tokens start in the full sequence
            step_start_idx = prompt_length + gen_step * num_blocks  
            step_end_idx = step_start_idx + num_blocks

            # Each block's tokens get sequential position IDs starting from their context end
            position_ids[:, step_start_idx:step_end_idx] = block_starting_positions + gen_step
    return attention_mask, position_ids




def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    prompt_len: Optional[int] = None
) -> torch.Tensor:
    """
    Compute log probabilities from logits and labels.

    Args:
        logits: Logits tensor [batch_size, seq_len, vocab_size]
        labels: Label tensor [batch_size, seq_len]
        temperature: Temperature for scaling. If != 1.0, scales logits.
        prompt_len: If provided, only apply temperature to tokens after this position.
                   This allows keeping prompt tokens at temperature=1.0 while scaling
                   generated tokens. Note: logits are typically [batch, seq_len-1, vocab]
                   from prepare_logprobs, so prompt_len should account for this.

    Returns:
        Log probabilities [batch_size, seq_len]
    """

    # Apply temperature scaling selectively if requested
    if temperature != 1.0:
        if prompt_len is not None and prompt_len > 0:
            # Only apply temperature to generated tokens (after prompt)
            # Note: Make a copy to avoid modifying the input
            logits = logits.clone()
            logits[:, prompt_len:] = logits[:, prompt_len:] / temperature
        else:
            # Apply temperature to all tokens
            logits = logits / temperature
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels

def compute_squared_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.log_softmax(logits, dim=-1).exp()
    labels_one_hot = F.one_hot(labels, num_classes=logits.size(-1))
    se = (probs - labels_one_hot.float())**2
    se = se.sum(dim=-1) #sum over vocab
    sequence_se = se.mean() # mean over batch and seqlen
    return sequence_se


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Masked mean along `dim`. If, for a given slice along `dim`, the mask sums to 0,
    we fall back to the unmasked mean for that slice.

    Args:
        tensor: [..., S, ...]
        mask:   same shape as tensor, broadcastable, or None
        dim:    dimension to reduce over
        keepdim: keep reduced dimension
    """
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)

    # ensure numeric mask
    mask = mask.to(tensor.dtype)

    # No dim: treat whole tensor as one slice
    if dim is None:
        denom = mask.sum()
        if denom == 0:
            return tensor.mean()
        return (tensor * mask).sum() / denom

    # With dim: do per-slice safe mean
    masked_sum = (tensor * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    # Unmasked mean for fallback
    mean_all = tensor.mean(dim=dim, keepdim=keepdim)

    # Avoid divide-by-zero; we'll overwrite those positions anyway
    safe_div = masked_sum / denom.clamp(min=1)

    # Where denom == 0, use unmasked mean; otherwise masked mean
    return torch.where(denom == 0, mean_all, safe_div)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def extract_and_reorder_rewards(experiences, reward_field_name: str, sample_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract rewards from experiences and reorder them based on sample indices.

    This function is used in RLHF training to extract specific reward types from
    Experience objects and reorder them according to the original sample ordering,
    which is important when using dynamic batching.

    Args:
        experiences: List of Experience objects containing reward data
        reward_field_name: Name of the reward field to extract (e.g., 'rewards', 'diversity_rewards', 'gt_rewards')
        sample_indices: Tensor containing the original ordering indices for reordering

    Returns:
        Stacked and reordered reward tensor with shape [num_experiences, reward_length]
    """
    stacked_rewards = torch.stack([
        getattr(experience, reward_field_name) for experience in experiences
    ], dim=0)

    # Reorder based on original sample indices
    sort_indices = torch.argsort(sample_indices)
    return stacked_rewards[sort_indices]
