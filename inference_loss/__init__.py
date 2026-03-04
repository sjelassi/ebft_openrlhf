"""
Inference Loss - Standalone evaluation module for computing metrics without Ray/training infrastructure.

This module provides simplified classes for:
- Strided generation and embedding extraction
- Reward computation (alignment, diversity)
- Perplexity computation
"""

from .strided_actor import StridedActorModel
from .strided_critic import StridedCriticModel
from .evaluation_metrics import EvaluationMetrics

__all__ = [
    "StridedActorModel",
    "StridedCriticModel",
    "EvaluationMetrics",
]
