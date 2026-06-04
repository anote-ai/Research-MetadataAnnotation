"""Evaluation utilities for the metadata annotation study."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metadataannotation.core import MetadataExperiment


def metadata_gain(baseline_recall: float, metadata_recall: float) -> float:
    """Absolute recall gain from adding metadata annotations."""
    return metadata_recall - baseline_recall


def correlation_complexity_gain(
    complexity_scores: list[float], gains: list[float]
) -> float:
    """Pearson correlation coefficient between complexity scores and gains."""
    n = len(complexity_scores)
    if n != len(gains):
        raise ValueError("complexity_scores and gains must have the same length.")
    if n < 2:  # noqa: PLR2004
        raise ValueError("At least 2 data points required.")

    mean_x = sum(complexity_scores) / n
    mean_y = sum(gains) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(complexity_scores, gains))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in complexity_scores))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in gains))
    if std_x == 0 or std_y == 0:
        return 0.0
    return num / (std_x * std_y)


def domain_summary(experiments: list["MetadataExperiment"]) -> dict:
    """Per-domain mean gain and mean complexity."""
    from collections import defaultdict

    domain_gains: dict = defaultdict(list)
    domain_complexity: dict = defaultdict(list)

    for exp in experiments:
        gain = metadata_gain(exp.baseline_recall, exp.metadata_recall)
        domain_gains[exp.domain.value].append(gain)
        domain_complexity[exp.domain.value].extend(exp.complexity_scores)

    summary: dict = {}
    for domain in domain_gains:
        gains = domain_gains[domain]
        complexities = domain_complexity[domain]
        summary[domain] = {
            "mean_gain": sum(gains) / len(gains),
            "mean_complexity": sum(complexities) / len(complexities) if complexities else 0.0,
        }
    return summary


def structure_prediction_error(
    predicted_gains: list[float], actual_gains: list[float]
) -> float:
    """Mean Absolute Error between predicted and actual gains."""
    if len(predicted_gains) != len(actual_gains):
        raise ValueError("Lists must have the same length.")
    if not predicted_gains:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted_gains, actual_gains)) / len(predicted_gains)
