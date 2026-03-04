from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import masked_mean

        

class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_ratio = log_probs - old_log_probs

        if self.policy_loss_type == "ppo":
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")
        
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, ratio.mean(), math.sqrt(ratio.var())


class EmbeddingLoss(nn.Module):
    """
    Embedding Loss for ED
    """

    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()

    def forward(self, gt_hidden_states: torch.Tensor, gen_hidden_states: torch.Tensor) -> torch.Tensor:
        # bsz/nspp, nspp, num_blocks, generate_max_len, hidden_size
        diff = gt_hidden_states - gen_hidden_states
        loss = torch.mean(torch.sum(torch.mean(diff, dim=1)**2, dim=-1))
        return -loss




class ClassifierLoss(nn.Module):
    """
    Classifier Accuracy for ED
    """

    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()

    def forward(self, gt_logits: torch.Tensor, gen_logits: torch.Tensor, sequence_selection: str = "first") -> torch.Tensor:
        s_real = gt_logits.squeeze(-1)   # logit for real
        s_fake = gen_logits.squeeze(-1)  # logit for fake

        if sequence_selection == "first":
            s_real = s_real[:, 0, ...].unsqueeze(1)
            s_fake = s_fake[:, 0, ...].unsqueeze(1)

            loss = (F.logsigmoid(s_real) - F.logsigmoid(s_fake)).mean()

        elif sequence_selection == "all":
            loss = (F.logsigmoid(s_real) - F.logsigmoid(s_fake)).mean()

        elif sequence_selection == "closest":
            s_real = s_real.transpose(0, 1)
            s_fake = s_fake.transpose(0, 1)
            transposed_shape = s_real.shape
            s_real = s_real.reshape(s_real.shape[0], -1)
            s_fake = s_fake.reshape(s_fake.shape[0], -1)

            equal_logits = s_real == s_fake
            best_indices = torch.argmax(s_fake - 1000 * equal_logits.float(), dim=0)

            idx = best_indices.unsqueeze(0)  # [1, 126]

            s_real = s_real.gather(0, idx)   # [1, 126]
            s_fake = s_fake.gather(0, idx)   # [1, 126]

            shape = list(transposed_shape)
            shape[0] = 1
            transposed_shape = torch.Size(shape)     # convert back

            s_real = s_real.reshape(transposed_shape)
            s_fake = s_fake.reshape(transposed_shape)
            s_real = s_real.transpose(0, 1)
            s_fake = s_fake.transpose(0, 1)

            loss = (F.logsigmoid(s_real) - F.logsigmoid(s_fake)).mean()

        elif sequence_selection == "only_different":
            different_logits = s_real != s_fake

            loss = F.logsigmoid(s_real).mean() - (F.logsigmoid(s_fake) * different_logits.float()).sum() / different_logits.float().sum()

        else:
            raise ValueError(f"Invalid sequence selection: {sequence_selection}")

        return -loss


class ClassifierAccuracy(nn.Module):
    """
    Classifier Accuracy for ED
    """

    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()

    def forward(self, gt_logits: torch.Tensor, gen_logits: torch.Tensor):
        # Probabilities
        s_real = F.sigmoid(gt_logits.squeeze(-1))   # real → should be 1
        s_fake = F.sigmoid(gen_logits.squeeze(-1))  # fake → should be 0
        # Predictions
        pred_real = (s_real > 0.5).long()   # 1 if we predict real correctly
        pred_fake = (s_fake > 0.5).long()   # 1 if model thinks fake=real (wrong!)

        acc_thresh0p5_real = (s_real == 0.5).float().mean()
        acc_thresh0p5_fake = (s_fake == 0.5).float().mean()
        acc_thresh0p5 = (acc_thresh0p5_real + acc_thresh0p5_fake) / 2

        # True labels
        labels_real = torch.ones_like(pred_real)        # real = 1
        labels_fake = torch.zeros_like(pred_fake)       # fake = 0

        # Concatenate samples
        preds  = torch.cat([pred_real, pred_fake], dim=0)
        labels = torch.cat([labels_real, labels_fake], dim=0)

        # Accuracy
        accuracy = (preds == labels).float().mean()

        # Confusion matrix terms
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()

        # Precision, recall
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        # F1 score
        f1_score  = 2 * precision * recall / (precision + recall + 1e-8)

        # Precision-Recall AUC (Average Precision) using raw probabilities
        # Flatten scores and build corresponding binary labels
        all_scores = torch.cat([s_real.reshape(-1), s_fake.reshape(-1)], dim=0)
        all_labels = torch.cat(
            [
                torch.ones_like(s_real, dtype=all_scores.dtype).reshape(-1),
                torch.zeros_like(s_fake, dtype=all_scores.dtype).reshape(-1),
            ],
            dim=0,
        )
        pos_total = all_labels.sum()
        if pos_total <= 0:
            pr_auc = all_scores.new_tensor(0.0)
        else:
            order = torch.argsort(all_scores, descending=True)
            sorted_labels = all_labels[order]
            tp_cum = torch.cumsum(sorted_labels, dim=0)
            fp_cum = torch.cumsum(1 - sorted_labels, dim=0)
            precision_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
            recall_curve = tp_cum / (pos_total + 1e-8)
            # Prepend start point (recall=0, precision=1)
            precision_curve = torch.cat([precision_curve.new_tensor([1.0]), precision_curve], dim=0)
            recall_curve = torch.cat([recall_curve.new_tensor([0.0]), recall_curve], dim=0)
            # Step-wise area under PR curve (Average Precision)
            pr_auc = torch.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:])

        # --------------------------------------------------------------------
        # ROC-AUC
        # --------------------------------------------------------------------
        P = int(all_labels.sum().item())
        N = len(all_labels) - P

        if P == 0 or N == 0:
            roc_auc = all_scores.new_tensor(0.0)
        else:
            order = torch.argsort(all_scores)
            sorted_labels = all_labels[order]

            ranks = torch.arange(1, len(all_labels) + 1, device=all_scores.device, dtype=all_scores.dtype)
            sum_pos_ranks = (ranks * sorted_labels).sum()

            roc_auc = (sum_pos_ranks - P * (P + 1) / 2) / (P * N)

        # Mean probability gap: average(s_real) - average(s_fake)
        mean_prob_gap = s_real.mean() - s_fake.mean()
        prob_fake = s_fake.mean()

        pred_pos_frac = (preds == 1).float().mean()

        return accuracy, precision, recall, f1_score, pr_auc, roc_auc, mean_prob_gap, prob_fake, pred_pos_frac, acc_thresh0p5


class EBFTPolicyLoss(nn.Module):
    """
    Policy loss for Energy-based Diffusion training.
    Supports PPO and GSPO policy gradient methods.
    """

    def __init__(
        self,
        policy_loss_type: str = "ppo",
    ) -> None:
        super().__init__()
        self.policy_loss_type = policy_loss_type

    def forward(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        qa_masks: Optional[torch.Tensor] = None,
        qa_masking: bool = False,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.

        Args:
            log_probs: Action log probabilities from current policy [B, S]
            advantages: Advantage estimates for each token [B, S]
            action_mask: Binary mask for generated tokens [B, S]

        Returns:
            tuple: RL loss scalars
        """
        # Compute log ratio (currently 0 as comparing to self)
        log_ratio = log_probs - log_probs.clone().detach()

        if self.policy_loss_type == "ppo":
            # Token-level probability ratio
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # Sequence-level ratio (GSPO: https://arxiv.org/pdf/2507.18071)
            seq_log_ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = seq_log_ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        if not qa_masking:
            qa_masks = torch.ones_like(qa_masks)

        surr_loss = -ratio * advantages
        rl_mask = action_mask & qa_masks

        rl_loss = masked_mean(surr_loss, rl_mask, dim=-1).mean()

        ce_mask = ~action_mask & qa_masks
        ce_loss = masked_mean(-log_probs, ce_mask, dim=-1).mean()

        return rl_loss, ce_loss



class CELoss(nn.Module):

    def __init__(
        self,
        policy_loss_type: str = "ppo",
    ) -> None:
        super().__init__()
        self.policy_loss_type = policy_loss_type

    def forward(
        self,
        log_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.

        Args:
            log_probs: Action log probabilities from current policy [B, S]
            advantages: Advantage estimates for each token [B, S]
            action_mask: Binary mask for generated tokens [B, S]

        Returns:
            tuple: RL loss scalars
        """
        # Compute log ratio (currently 0 as comparing to self)
        ce_mask = ~action_mask
        ce_loss = masked_mean(-log_probs, ce_mask, dim=-1).mean()

        return ce_loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
