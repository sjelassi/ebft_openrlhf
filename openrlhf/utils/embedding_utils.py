import torch
import torch.nn.functional as F
import numpy as np
import ray
from collections import Counter


def prepare_tensors_for_embedding(prompts_list, full_sequences_list, prompt_length, stride, num_blocks, n_samples_per_prompt, context_length, gen_len, return_inputs=False):
    """
    Returns a list of decoded strings for semantic reward calculation, across all rollout_batch_size * n_samples_per_prompt examples.
    """
    prompts_tensor = torch.stack(prompts_list)  # (rollout_batch_size // micro_rollout_batch_size, micro_rollout_batch_size, prompt_len)
    prompts_tensor = prompts_tensor.reshape(prompts_tensor.shape[0], prompts_tensor.shape[1] // n_samples_per_prompt, n_samples_per_prompt, prompts_tensor.shape[2])

    full_tensor = torch.stack(full_sequences_list)
    full_tensor = full_tensor.reshape(full_tensor.shape[0], full_tensor.shape[1] // n_samples_per_prompt, n_samples_per_prompt, full_tensor.shape[2])

    starting_idx = context_length
    gt_tensor = prompts_tensor[:,:,:,starting_idx:].unfold(3, gen_len, stride)

    gen_tensor = full_tensor[:,:,:,prompt_length:]
    gen_tensor = gen_tensor.reshape(gen_tensor.shape[0], gen_tensor.shape[1], gen_tensor.shape[2], gen_len, num_blocks)
    gen_tensor = gen_tensor.transpose(-1, -2)

    if return_inputs:
        ct_tensor = prompts_tensor[:,:,:,:-gen_len].unfold(3, context_length, stride)
        return gen_tensor, gt_tensor, ct_tensor
    return gen_tensor, gt_tensor

def temp_embed_one_hot(input_sequences, gt_sequences, vocab_size, dtype=torch.float32):
    # one_hot requires Long dtype with values in [0, vocab_size-1]
    input_sequences = input_sequences.to(torch.long, non_blocking=True)
    gt_sequences    = gt_sequences.to(torch.long, non_blocking=True)

    input_oh = F.one_hot(input_sequences, num_classes=vocab_size).to(dtype)
    gt_oh    = F.one_hot(gt_sequences,    num_classes=vocab_size).to(dtype)

    return input_oh, gt_oh

def decode_tensor(input_tensor, tokenizer):
    return tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

def prepare_tensors_for_reward_model(input_sequences, gt_sequences, tokenizer, ct_sequences=None):
    if ct_sequences is not None:
        # flatten (rollout_batch_size * n_samples_per_prompt // micro_rollout_batch_size, micro_rollout_batch_size // n_samples_per_prompt, n_samples_per_prompt, num_blocks, partial_ct_len + gen_len)
        return decode_tensor(input_sequences.reshape(-1, input_sequences.shape[-1]), tokenizer), decode_tensor(gt_sequences.reshape(-1, gt_sequences.shape[-1]), tokenizer), decode_tensor(ct_sequences.reshape(-1, ct_sequences.shape[-1]), tokenizer)
    # flatten (rollout_batch_size * n_samples_per_prompt // micro_rollout_batch_size, micro_rollout_batch_size // n_samples_per_prompt, n_samples_per_prompt, num_blocks, partial_ct_len + gen_len)
    return decode_tensor(input_sequences.reshape(-1, input_sequences.shape[-1]), tokenizer), decode_tensor(gt_sequences.reshape(-1, gt_sequences.shape[-1]), tokenizer)


def whiten_embeddings_batched(
    Phi: torch.Tensor,
    Phi_gt: torch.Tensor,
    whiten_tol: float = 1e-5,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Batch whitening across the *sample axis* (N).

    This function expects the sample axis to be dimension 2 and the embedding dimension
    to be the last dimension.

    Supported input shapes:
    - 5D: (d1, d2, N, d3, D)
    - 7D: (d1, d2, N, d3, d4, d5, D)

    For each fixed index in the non-(N,D) dimensions, we take X ∈ R^{N×D} from Phi
    and compute:
        W = (X X^T)^(-1/2)   (pseudo-inverse with tolerance)
        Xw    = W X
        Xw_gt = W X_gt

    So Xw has (approximately) orthonormal rows when rank(X)=N; when rank-deficient,
    this becomes a projection onto the row-space.
    """
    if Phi.shape != Phi_gt.shape:
        raise ValueError(f"Phi and Phi_gt must have the same shape, got {Phi.shape} vs {Phi_gt.shape}")
    if Phi.ndim not in (5, 7):
        raise ValueError(f"Expected Phi.ndim in {{5,7}} with sample axis at dim=2, got {Phi.shape}")

    # Permute to move sample axis N next-to-last, keep embedding dim last.
    # For nd=5:  [0,1,3,2,4]
    # For nd=7:  [0,1,3,4,5,2,6]
    nd = Phi.ndim
    perm = [0, 1] + list(range(3, nd - 1)) + [2, nd - 1]
    inv_perm = [0] * nd
    for i, p in enumerate(perm):
        inv_perm[p] = i

    Phi_perm = Phi.permute(*perm).contiguous()
    Phi_gt_perm = Phi_gt.permute(*perm).contiguous()

    # Flatten all "batch" dims into one for batched linear algebra.
    *batch_dims, N, D = Phi_perm.shape
    B = 1
    for x in batch_dims:
        B *= int(x)

    Phi_flat = Phi_perm.reshape(B, N, D).float()
    Phi_gt_flat = Phi_gt_perm.reshape(B, N, D).float()

    # Batched SVD on (B, N, D) where typically N << D.
    # Use robust SVD with fallback for ill-conditioned matrices.
    try:
        U, S, _ = torch.linalg.svd(Phi_flat, full_matrices=False)  # U: (B,N,N), S: (B,N)
    except torch._C._LinAlgError:
        # Fallback 1: Add small noise to break degeneracy and retry
        noise_scale = 1e-6 * Phi_flat.abs().mean()
        Phi_flat_noisy = Phi_flat + noise_scale * torch.randn_like(Phi_flat)
        try:
            U, S, _ = torch.linalg.svd(Phi_flat_noisy, full_matrices=False)
        except torch._C._LinAlgError:
            # Fallback 2: Return original embeddings without whitening
            if normalize:
                Phi_out = F.normalize(Phi, p=2, dim=-1)
                Phi_gt_out = F.normalize(Phi_gt, p=2, dim=-1)
                return Phi_out, Phi_gt_out
            return Phi, Phi_gt

    # Safe inverse: zero out tiny singular values (per-batch).
    Smax = S.max(dim=-1, keepdim=True).values
    inv_S = torch.where(S > whiten_tol * Smax, 1.0 / (S + 1e-12), torch.zeros_like(S))  # (B,N)

    # W = U diag(inv_S) U^T
    W = (U * inv_S.unsqueeze(-2)) @ U.transpose(-1, -2)  # (B,N,N)
    Xw = W @ Phi_flat
    Xw_gt = W @ Phi_gt_flat

    # Cast back and reshape/unpermute to original.
    Xw = Xw.to(dtype=Phi.dtype)
    Xw_gt = Xw_gt.to(dtype=Phi_gt.dtype)

    Phi_tilde = Xw.reshape(*batch_dims, N, D).permute(*inv_perm).contiguous()
    Phi_gt_tilde = Xw_gt.reshape(*batch_dims, N, D).permute(*inv_perm).contiguous()

    if normalize:
        Phi_tilde = F.normalize(Phi_tilde, p=2, dim=-1)
        Phi_gt_tilde = F.normalize(Phi_gt_tilde, p=2, dim=-1)

    return Phi_tilde, Phi_gt_tilde

@torch.no_grad()
def call_rm_model(input_sequences, gt_sequences, n_samples, num_blocks, rm_actors, args, training=True, eval_dataloader_len=None):
    all_sequences = input_sequences + gt_sequences
    all_sequences = [(i // args.micro_reward_batch_size, all_sequences[i:i + args.micro_reward_batch_size]) for i in range(0, len(all_sequences), args.micro_reward_batch_size)]
    if not rm_actors:
        raise RuntimeError("No actors available in reward_model_group.")

    inflight_refs = []  # List[Tuple[int, ObjectRef, List[int]]]
    for k, (bid, batch_2d) in enumerate(all_sequences):
        a = rm_actors[k % len(rm_actors)]
        # IMPORTANT: pass a single 2D batch, not a list of batches
        ref = a.forward.remote(
            input_sequences=batch_2d,
        )
        inflight_refs.append((bid, ref))
    
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
    ordered = [results_by_bid[i] for i in range(len(results_by_bid))] # list of length rollout_batch * (1+n_samples) * n_blocks // reward_micro_batch of tensors of size (reward_micro_batch, embd_dim)
    ordered_tensor = torch.cat(ordered, dim=0) # (rollout_batch * (1+n_samples) * n_blocks, embd_dim)
    gen_embeddings, gt_embeddings = ordered_tensor[:-len(gt_sequences),:], ordered_tensor[-len(gt_sequences):,:]  # (rollout_batch * n_samples * n_blocks, embd_dim) and (rollout_batch * 1 * n_blocks, embd_dim)

    # what shape do we want to end up with? take all the samples (rollout_batch*n_samples_pp). first split into batches of micro_rollout_batch size. each mrb can contain multiple prompts
    # then split each mrb into groups of size n_samples_pp. each item in this group is of size num_blocks, embed_dim. 
    # final shape is rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, embed dim

    if training:
        gen_embeddings_reshaping_shape = (args.rollout_batch_size * n_samples // args.micro_rollout_batch_size, args.micro_rollout_batch_size // n_samples, n_samples, num_blocks, gen_embeddings.shape[-1])
        gt_embeddings_reshaping_shape = (args.rollout_batch_size * n_samples // args.micro_rollout_batch_size, args.micro_rollout_batch_size // n_samples, n_samples, num_blocks, gt_embeddings.shape[-1])
    else:
        gen_embeddings_reshaping_shape = (eval_dataloader_len, 1, n_samples, num_blocks, gen_embeddings.shape[-1])
        gt_embeddings_reshaping_shape = (eval_dataloader_len, 1, n_samples, num_blocks, gt_embeddings.shape[-1])
    gen_embeddings = gen_embeddings.reshape(gen_embeddings_reshaping_shape)
    gt_embeddings = gt_embeddings.reshape(gt_embeddings_reshaping_shape)
    return gen_embeddings, gt_embeddings

@torch.no_grad()
def compute_ngram_similarity(seq_y, seq_t, n):
    """
    Compute normalized n-gram overlap between sequences.
    This corresponds to equation (46) in the document.
    """
    # Ensure sequences are lists of ints
    if hasattr(seq_y, 'tolist'):
        seq_y = seq_y.tolist()
    if hasattr(seq_t, 'tolist'):
        seq_t = seq_t.tolist()

    # Get n-grams for both sequences
    ngrams_y = Counter([tuple(seq_y[i:i+n]) 
                        for i in range(len(seq_y) - n + 1)])
    ngrams_t = Counter([tuple(seq_t[i:i+n]) 
                        for i in range(len(seq_t) - n + 1)])
    
    # Compute numerator: sum of products for common n-grams
    common_ngrams = set(ngrams_y.keys()) & set(ngrams_t.keys())

     
    numerator = sum(ngrams_y[g] * ngrams_t[g] for g in common_ngrams)
    # Compute denominators
    norm_y = np.sqrt(sum(count**2 for count in ngrams_y.values()))
    norm_t = np.sqrt(sum(count**2 for count in ngrams_t.values()))
    
    # Return normalized similarity
    if norm_y > 0 and norm_t > 0:
        return numerator / (norm_y * norm_t)
    return 0.0

@torch.no_grad()
def get_mean_ngram_similarities(seq_y, seq_t, bleu_max_n, mean_mode):
    similarities = []
    for i in range(bleu_max_n):
        similarity = compute_ngram_similarity(seq_y, seq_t, i+1)
        similarities.append(similarity)
    similarities = torch.tensor(similarities)
    if mean_mode == "geometric":
        similarities = torch.log(similarities+1e-6)
    similarity = similarities.mean(dim=0)
    if mean_mode == "geometric":
        similarity = torch.exp(similarity)
    return similarity


@torch.no_grad()
def get_alignment_rewards(gen_embedding, gt_embedding):
    # Alignment reward: cosine similarity so the actor optimizes directional
    # alignment in embedding space (not raw vector magnitude).
    gt_rewards_tensor = F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)
    return gt_rewards_tensor


@torch.no_grad()
def get_diversity_rewards(gen_embedding, per_token=False):
    if gen_embedding.shape[2] > 1:
        if per_token:
            #rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, embed dim
            reorg = gen_embedding.permute(0,1,3,2,4,5) # num micro batches, num groups per micro batch, num blocks, n samples pp, embed dim
            n_samples_per_prompt = gen_embedding.shape[2]
            gen_embedding_unsqueeze_2 = reorg.unsqueeze(3).repeat(1,1,1,n_samples_per_prompt,1,1,1)
            gen_embedding_unsqueeze_3 = reorg.unsqueeze(4).repeat(1,1,1,1,n_samples_per_prompt,1,1)
            full_sims = torch.sum(gen_embedding_unsqueeze_2 * gen_embedding_unsqueeze_3, dim=-1) # num micro batches, num groups per micro batch, num blocks, num samples per group, num_samples_per_group
            # must zero out sim with itself. First create 2d diagonal mask
            no_jvms = torch.eye(full_sims.shape[-2], device=full_sims.device, dtype=torch.bool)
            # reshape diagonal mask to correct shape. fill full sims along this diag with zeros
            sims = full_sims.masked_fill(no_jvms.view(1,1,1,full_sims.shape[-2],full_sims.shape[-2],1), 0.0)
            # average across samples to get diversity reward for each sample
            diversity_rewards = sims.sum(dim=-2) / (full_sims.shape[-2] - 1)
            # reshape into original format
            # num micro batches, num groups per micro batch, num samples per group, num blocks
            diversity_rewards_tensor = diversity_rewards.permute(0,1,3,2,4)
        else:
            #rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, num features, embed dim
            reorg = gen_embedding.permute(0,1,3,2,4) # num micro batches, num groups per micro batch, num blocks, n samples pp, num features, embed dim
            n_samples_per_prompt = gen_embedding.shape[2]
            gen_embedding_unsqueeze_2 = reorg.unsqueeze(3).repeat(1,1,1,n_samples_per_prompt,1,1)
            gen_embedding_unsqueeze_3 = reorg.unsqueeze(4).repeat(1,1,1,1,n_samples_per_prompt,1)
            full_sims = torch.sum(gen_embedding_unsqueeze_2 * gen_embedding_unsqueeze_3, dim=-1) # num micro batches, num groups per micro batch, num blocks, num samples per group, num_samples_per_group
            # must zero out sim with itself. First create 2d diagonal mask
            no_jvms = torch.eye(full_sims.shape[-1], device=full_sims.device, dtype=torch.bool)
            # reshape diagonal mask to correct shape. fill full sims along this diag with zeros
            sims = full_sims.masked_fill(no_jvms.view(1,1,1,full_sims.shape[-1],full_sims.shape[-1]), 0.0)
            # average across samples to get diversity reward for each sample
            diversity_rewards = sims.sum(dim=-1) / (full_sims.shape[-1] - 1)
            # reshape into original format
            # num micro batches, num groups per micro batch, num samples per group, num blocks
            diversity_rewards_tensor = diversity_rewards.permute(0,1,3,2)
    else:
        # num micro batches, num groups per micro batch, num samples per group, num blocks
        diversity_rewards_tensor = torch.zeros(gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2], gen_embedding.shape[3], device=gen_embedding.device)
    return diversity_rewards_tensor