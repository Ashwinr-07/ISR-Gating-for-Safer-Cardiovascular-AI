"""Shared utility functions."""

import math
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wilson_bounds(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    pm = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return ((center - pm) / denom, (center + pm) / denom)


def trimmed_mean(vals: list[float], trim_alpha: float = 0.2) -> float:
    """Symmetrically trimmed mean (removes top/bottom trim_alpha fraction)."""
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) == 0:
        return 0.0
    k = int(len(v) * trim_alpha)
    if 2 * k >= len(v):
        return float(np.mean(v))
    return float(np.mean(v[k : len(v) - k]))
