from __future__ import annotations

import math
from typing import List, Tuple, Dict

from .core import MetadataExperiment, DocumentDomain


def metadata_gain(baseline_recall: float, metadata_recall: float) -> float:
    """Return the difference in recall."""
    return metadata_recall - baseline_recall


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient manually."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def correlation_complexity_gain(
    complexity_scores: List[float], gains: List[float]
) -> float:
    """Pearson r between complexity scores and gains."""
    return pearson_r(complexity_scores, gains)


def domain_summary(
    experiments: List[MetadataExperiment],
) -> Dict[str, Dict]:
    """Return per-domain mean gain, mean complexity, and count."""
    from collections import defaultdict
    import statistics

    buckets: Dict[str, list] = defaultdict(list)
    for e in experiments:
        buckets[e.domain.value].append(e)

    result: Dict[str, Dict] = {}
    for domain_val, exps in buckets.items():
        gains = [ex.gain for ex in exps]
        complexities = [
            statistics.mean(ex.complexity_scores) for ex in exps
        ]
        result[domain_val] = {
            "mean_gain": sum(gains) / len(gains),
            "mean_complexity": sum(complexities) / len(complexities),
            "n": len(exps),
        }
    return result


def structure_prediction_error(
    predicted_gains: List[float], actual_gains: List[float]
) -> float:
    """Mean Absolute Error between predicted and actual gains."""
    if not predicted_gains:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted_gains, actual_gains)) / len(
        predicted_gains
    )


def rank_domains_by_gain(
    experiments: List[MetadataExperiment],
) -> List[Tuple[str, float]]:
    """Return list of (domain, mean_gain) sorted descending by mean_gain."""
    summary = domain_summary(experiments)
    ranked = sorted(
        ((d, v["mean_gain"]) for d, v in summary.items()),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked
