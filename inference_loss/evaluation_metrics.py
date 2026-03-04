"""
Evaluation Metrics - Compute rewards and perplexity metrics.

Maintains the same reward computation as ebft_trainer.py evaluate() function.
"""

import math
import torch
from typing import Dict, Tuple

from openrlhf.utils.embedding_utils import (
    get_alignment_rewards,
    get_diversity_rewards,
)
import torch.nn.functional as F


class EvaluationMetrics:
    """
    Compute evaluation metrics without training infrastructure.

    Computes:
    - Alignment rewards (gt)
    - Diversity rewards
    - Combined rewards (effective, effective_alpha_1)
    - Pass@1 and Pass@k metrics
    - Perplexity and cross-entropy loss
    """

    def __init__(
        self,
        alignment_rew_coef: float = 1.0,
        diversity_rew_coef: float = 1.0,
        embed_method: str = "last_token",
        use_whitening: bool = False,
    ):
        """
        Args:
            alignment_rew_coef: Coefficient for alignment reward
            diversity_rew_coef: Coefficient for diversity penalty
            embed_method: Embedding method (affects per-token rewards)
            use_whitening: Whether to apply whitening on embeddings
        """
        self.alignment_rew_coef = alignment_rew_coef
        self.diversity_rew_coef = diversity_rew_coef
        self.embed_method = embed_method
        self.use_whitening = use_whitening

    def compute_rewards(
        self,
        gen_embedding: torch.Tensor,
        gt_embedding: torch.Tensor,
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all reward metrics.

        Args:
            gen_embedding: Generated embeddings
                - If embed_method != "token": (batch * n_samples, num_blocks, num_features, hidden)
                - If embed_method == "token": (batch * n_samples, num_blocks, seq_len, num_features, hidden)
                  where consecutive n_samples sequences correspond to the same prompt.
                  This organization is guaranteed by the actor's repeat_interleave expansion.
            gt_embedding: Ground truth embeddings (same shape and organization as gen_embedding)
            n_samples: Number of samples per prompt

        Returns:
            Dictionary with reward tensors
        """
        # Input embeddings now have gen_len dimension:
        # Shape: (batch * n_samples, num_blocks, gen_len, num_features, hidden)
        total_size = gen_embedding.size(0)
        num_prompts = total_size // n_samples
        num_blocks = gen_embedding.size(1)
        gen_len = gen_embedding.size(2)
        num_features = gen_embedding.size(3)
        hidden_size = gen_embedding.size(4)

        # Reshape to separate n_samples: (num_prompts, n_samples, num_blocks, gen_len, num_features, hidden)
        gen_embedding = gen_embedding.view(num_prompts, n_samples, num_blocks, gen_len, num_features, hidden_size)
        gt_embedding = gt_embedding.view(num_prompts, n_samples, num_blocks, gen_len, num_features, hidden_size)

        if self.embed_method == "mean_pooling":
            # Average over gen_len dimension
            gt_embedding = torch.mean(gt_embedding, dim=3, keepdim=True)  # (num_prompts, n_samples, num_blocks, 1, num_features, hidden)
            gen_embedding = torch.mean(gen_embedding, dim=3, keepdim=True)
        elif self.embed_method == "last_token":
            # Take last token (matching ebft_trainer.py line 2581)
            gt_embedding = gt_embedding[:, :, :, -1, :, :].unsqueeze(3)  # (num_prompts, n_samples, num_blocks, 1, num_features, hidden)
            gen_embedding = gen_embedding[:, :, :, -1, :, :].unsqueeze(3)
        elif self.embed_method == "concat":
            # Concatenate across gen_len dimension (matching ebft_trainer.py line 2584)
            gt_embedding = gt_embedding.transpose(3, 4).reshape(num_prompts, n_samples, num_blocks, 1, num_features, gen_len * hidden_size)
            gen_embedding = gen_embedding.transpose(3, 4).reshape(num_prompts, n_samples, num_blocks, 1, num_features, gen_len * hidden_size)
        elif self.embed_method == "token":
            # Keep token-level embeddings
            pass  # Shape: (num_prompts, n_samples, num_blocks, gen_len, num_features, hidden)
        else:
            raise ValueError(f"Unknown embed_method: {self.embed_method}")

        # Apply whitening if requested (on un-flattened embeddings)
        # Shape: (num_prompts, n_samples, num_blocks, seq_len, num_features, hidden)
        if self.use_whitening:
            from openrlhf.utils.embedding_utils import whiten_embeddings_batched

            # Whitening expects 7D: (num_micro_batches, num_groups, n_samples, num_blocks, seq_len, num_feat, hidden)
            # We have 6D: (num_prompts, n_samples, num_blocks, seq_len, num_feat, hidden)
            # Add micro-batch dimension at the beginning
            gen_embedding_for_whitening = gen_embedding.unsqueeze(0)
            gt_embedding_for_whitening = gt_embedding.unsqueeze(0)

            gen_embedding_w, gt_embedding_w = whiten_embeddings_batched(
                gen_embedding_for_whitening, gt_embedding_for_whitening, whiten_tol=1e-5, normalize=False
            )

            # Remove micro-batch dimension
            gen_embedding = gen_embedding_w.squeeze(0)
            gt_embedding = gt_embedding_w.squeeze(0)

        # Flatten the last two dimensions: (num_features, hidden) -> (num_features * hidden)
        gen_embedding = gen_embedding.reshape(
            gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2],
            gen_embedding.shape[3], gen_embedding.shape[4] * gen_embedding.shape[5]
        )
        gt_embedding = gt_embedding.reshape(
            gt_embedding.shape[0], gt_embedding.shape[1], gt_embedding.shape[2],
            gt_embedding.shape[3], gt_embedding.shape[4] * gt_embedding.shape[5]
        )

        if self.embed_method != "token":
            gt_embedding = gt_embedding.squeeze(3)  # (num_prompts, n_samples, num_blocks, features)
            gen_embedding = gen_embedding.squeeze(3)

        # When use_whitening=True: whitening decorrelates samples and normalizes variance
        # When use_whitening=False: raw embeddings are used directly for reward computation

        # Add dummy batch dimension for compatibility with reward functions
        # Expected shape: (num_micro_batches, num_groups, n_samples, num_blocks, features)
        gen_embedding_batched = gen_embedding.unsqueeze(0)
        gt_embedding_batched = gt_embedding.unsqueeze(0)

        # Compute alignment and diversity rewards
        gt_rewards_tensor = get_alignment_rewards(
            gen_embedding_batched, gt_embedding_batched
        )
        diversity_rewards_tensor = get_diversity_rewards(
            gen_embedding_batched, per_token=(self.embed_method == "token")
        )

        # Reshape rewards
        gt_rewards_tensor = gt_rewards_tensor.reshape(
            gt_rewards_tensor.shape[0], -1, gt_rewards_tensor.shape[-1]
        )
        diversity_rewards_tensor = diversity_rewards_tensor.reshape(
            diversity_rewards_tensor.shape[0], -1, diversity_rewards_tensor.shape[-1]
        )

        # Scale rewards
        gt_rewards_tensor *= 2
        diversity_rewards_tensor *= 2

        # Combine rewards
        rewards_tensor = (self.alignment_rew_coef * gt_rewards_tensor -
                         self.diversity_rew_coef * diversity_rewards_tensor)
        effective_rewards_tensor = (self.alignment_rew_coef * gt_rewards_tensor -
                                    self.diversity_rew_coef * diversity_rewards_tensor / 2)

        # Flatten rewards for pass@1 and pass@k computation
        rewards = torch.cat(
            [r.transpose(1, 0).reshape(-1) for r in rewards_tensor], dim=0
        ).reshape(-1, n_samples)

        effective_rewards = torch.cat(
            [r.transpose(1, 0).reshape(-1) for r in effective_rewards_tensor], dim=0
        ).reshape(-1, n_samples)

        diversity_rewards = torch.cat(
            [r.transpose(1, 0).reshape(-1) for r in diversity_rewards_tensor], dim=0
        ).reshape(-1, n_samples)

        gt_rewards = torch.cat(
            [r.transpose(1, 0).reshape(-1) for r in gt_rewards_tensor], dim=0
        ).reshape(-1, n_samples)

        return {
            "rewards": rewards,
            "effective_rewards": effective_rewards,
            "diversity_rewards": diversity_rewards,
            "gt_rewards": gt_rewards,
        }

    def compute_pass_metrics(
        self,
        reward_dict: Dict[str, torch.Tensor],
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Compute pass@1 and pass@k metrics.

        Args:
            reward_dict: Dictionary with reward tensors from compute_rewards()
            n_samples: Number of samples per prompt

        Returns:
            Dictionary with pass@1 and pass@k metrics
        """
        rewards = reward_dict["rewards"]
        effective_rewards = reward_dict["effective_rewards"]
        diversity_rewards = reward_dict["diversity_rewards"]
        gt_rewards = reward_dict["gt_rewards"]

        # Calculate pass@k and pass@1 metrics
        if n_samples > 1:
            passk = rewards.max(dim=1).values.mean().item()
            passk_effective = effective_rewards.max(dim=1).values.mean().item()
            passk_diversity = diversity_rewards.max(dim=1).values.mean().item()
            passk_gt = gt_rewards.max(dim=1).values.mean().item()
        else:
            passk = rewards.mean().item()
            passk_effective = effective_rewards.mean().item()
            passk_diversity = diversity_rewards.mean().item()
            passk_gt = gt_rewards.mean().item()

        pass1 = rewards.mean().item()
        pass1_effective = effective_rewards.mean().item()
        pass1_diversity = diversity_rewards.mean().item()
        pass1_gt = gt_rewards.mean().item()

        # Compute effective_alpha_1 (explicitly computed as gt - div/2)
        pass1_effective_alpha_1 = pass1_gt - pass1_diversity / 2.0
        passk_effective_alpha_1 = passk_gt - passk_diversity / 2.0

        return {
            "reward_pass1": pass1,
            "reward_pass1_gt": pass1_gt,
            "reward_pass1_diversity": pass1_diversity,
            "reward_pass1_effective": pass1_effective,
            "reward_pass1_effective_alpha_1": pass1_effective_alpha_1,
            "reward_passk": passk,
            "reward_passk_gt": passk_gt,
            "reward_passk_diversity": passk_diversity,
            "reward_passk_effective": passk_effective,
            "reward_passk_effective_alpha_1": passk_effective_alpha_1,
        }

    def compute_perplexity_from_logprobs(
        self,
        log_probs: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute cross-entropy loss and perplexity from log probabilities.

        Args:
            log_probs: Log probabilities for each token

        Returns:
            (ce_loss, perplexity)
        """
        ce_loss = -log_probs.mean().item()
        perplexity = math.exp(ce_loss)

        return ce_loss, perplexity

    def compute_all_metrics(
        self,
        gen_embedding: torch.Tensor,
        gt_embedding: torch.Tensor,
        log_probs: torch.Tensor,
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics at once.

        Args:
            gen_embedding: Generated embeddings
            gt_embedding: Ground truth embeddings
            log_probs: Log probabilities for perplexity
            n_samples: Number of samples per prompt

        Returns:
            Dictionary with all metrics
        """
        # Compute rewards
        reward_dict = self.compute_rewards(gen_embedding, gt_embedding, n_samples)

        # Compute pass metrics
        metrics = self.compute_pass_metrics(reward_dict, n_samples)

        # Compute perplexity
        ce_loss, perplexity = self.compute_perplexity_from_logprobs(log_probs)
        metrics.update({
            "full_ce_loss": ce_loss,
            "full_perplexity": perplexity,
        })

        return metrics
