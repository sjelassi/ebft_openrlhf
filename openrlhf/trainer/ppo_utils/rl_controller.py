import numpy as np


class AdaptiveRLController:
    """
    Adaptive RL controller that starts at 0, then linearly increases from rl_loss_warmup_start to rl_loss_warmup_start + horizon,
    and stays fixed at init_rl_coef afterwards.
    """

    def __init__(self, target_rl, rl_loss_warmup_start, horizon):
        self.target_rl = target_rl
        self.rl_loss_warmup_start = rl_loss_warmup_start
        self.horizon = horizon
        self.total_steps = 0
        self.value = 0

    def update(self, n_steps):
        self.total_steps = n_steps

        if self.total_steps < self.rl_loss_warmup_start:
            # Before rl_loss_warmup_start: value stays at 0
            self.value = 0
        elif self.total_steps < self.rl_loss_warmup_start + self.horizon:
            # Linear warmup from rl_loss_warmup_start to rl_loss_warmup_start + horizon
            steps_into_warmup = self.total_steps - self.rl_loss_warmup_start
            self.value = self.target_rl * (steps_into_warmup / self.horizon)
        else:
            # After warmup: stays fixed at init_rl_coef
            self.value = self.target_rl


class FixedRLController:
    """Fixed RL controller."""

    def __init__(self, rl_coef):
        self.value = rl_coef

    def update(self, n_steps):
        pass
