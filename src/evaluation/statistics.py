"""Paired statistical utilities for benchmark comparisons."""

import random
from typing import Dict, Iterable, List, Sequence, Tuple


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> List[float]:
    if len(a) != len(b):
        raise ValueError("Paired samples must have equal length.")
    return [float(x) - float(y) for x, y in zip(a, b)]


def bootstrap_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_bootstrap: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap CI for the mean difference a - b."""
    diffs = _paired_differences(a, b)
    if not diffs:
        raise ValueError("Cannot bootstrap empty paired samples.")
    rng = random.Random(seed)
    estimates = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo_idx = max(0, min(len(estimates) - 1, int((alpha / 2) * len(estimates))))
    hi_idx = max(0, min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)) - 1))
    observed = sum(diffs) / n
    return {
        "mean_difference": observed,
        "ci_low": estimates[lo_idx],
        "ci_high": estimates[hi_idx],
        "n": float(n),
    }


def paired_permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation p-value."""
    diffs = _paired_differences(a, b)
    if not diffs:
        raise ValueError("Cannot test empty paired samples.")
    rng = random.Random(seed)
    observed = abs(sum(diffs) / len(diffs))
    extreme = 0
    for _ in range(n_permutations):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        statistic = abs(sum(signed) / len(signed))
        if statistic >= observed:
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)
