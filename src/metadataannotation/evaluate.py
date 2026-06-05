from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .core import (
    DocumentDomain,
    MetadataAnnotation,
    MetadataExperiment,
    annotation_confidence_score,
    fleiss_kappa,
)


def metadata_gain(baseline_recall: float, metadata_recall: float) -> float:
    """Return the difference in recall."""
    return metadata_recall - baseline_recall


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient manually."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
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
    if len(predicted_gains) != len(actual_gains):
        raise ValueError("predicted_gains and actual_gains must have the same length.")
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


def fleiss_kappa_from_annotations(
    annotations: List[MetadataAnnotation],
    doc_ids: Optional[List[str]] = None,
    field: Optional[str] = None,
) -> float:
    """Compute Fleiss’ kappa from a list of MetadataAnnotation objects.

    Optionally restrict to specific document IDs or a specific field.
    Items with fewer than 2 annotations are excluded.
    """
    subset = annotations
    if field is not None:
        subset = [a for a in subset if a.field == field]
    if doc_ids is not None:
        doc_id_set = set(doc_ids)
        subset = [a for a in subset if a.doc_id in doc_id_set]

    # Group by (doc_id, field) to form rating items
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for a in subset:
        groups[(a.doc_id, a.field)].append(a.value)

    # Keep only groups with >= 2 annotations
    ratings = [labels for labels in groups.values() if len(labels) >= 2]
    if not ratings:
        return 0.0
    return fleiss_kappa(ratings)


def annotation_quality_score(
    annotations: List[MetadataAnnotation],
    doc_ids: Optional[List[str]] = None,
    field: Optional[str] = None,
) -> float:
    """Composite annotation quality score in [0, 1].

    Combines Fleiss kappa (inter-annotator agreement) with mean confidence.
    Both components contribute 50% each.
    """
    kappa = fleiss_kappa_from_annotations(annotations, doc_ids=doc_ids, field=field)
    # Normalise kappa from [-1, 1] to [0, 1]
    kappa_norm = (kappa + 1.0) / 2.0
    confidence = annotation_confidence_score(annotations, field=field)
    return 0.5 * kappa_norm + 0.5 * confidence
