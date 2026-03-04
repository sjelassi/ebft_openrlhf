import numpy as np


class AdaptiveCEController:
    """
    Adaptive CE controller that linearly decreases the CE coefficient to 0 within horizon steps.
    """

    def __init__(self, ce_loss_coef, horizon):
        self.init_value = ce_loss_coef
        self.value = ce_loss_coef
        self.horizon = horizon
        self.total_steps = 0

    def update(self, n_steps):
        self.total_steps = n_steps
        # Linear decay to 0
        self.value = max(0, self.init_value * (1 - self.total_steps / self.horizon))


class FixedCEController:
    """Fixed CE controller."""

    def __init__(self, ce_coef):
        self.value = ce_coef

    def update(self, n_steps):
        pass


