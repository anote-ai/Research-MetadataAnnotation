from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class DocumentDomain(str, Enum):
    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"


@dataclass
class StructureComplexity:
    doc_id: str
    num_sections: int
    avg_section_length: float
    has_tables: bool
    has_figures: bool

    @property
    def complexity_score(self) -> float:
        score = (
            0.3 * min(self.num_sections / 20, 1.0)
            + 0.3 * min(self.avg_section_length / 500, 1.0)
            + 0.2 * float(self.has_tables)
            + 0.2 * float(self.has_figures)
        )
        return score


@dataclass
class MetadataAnnotation:
    doc_id: str
    field: str
    value: str
    confidence: float
    annotator: str

    def __post_init__(self) -> None:
        if self.annotator not in {"llm", "human"}:
            raise ValueError(f"annotator must be 'llm' or 'human', got {self.annotator!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class MetadataExperiment:
    domain: DocumentDomain
    baseline_recall: float
    metadata_recall: float
    complexity_scores: List[float]

    @property
    def gain(self) -> float:
        return self.metadata_recall - self.baseline_recall


class StructurePredictor:
    """Linear regression: gain ~ complexity, fitted with OLS via math/statistics."""

    def __init__(self) -> None:
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._fitted: bool = False

    def fit(self, experiments: List[MetadataExperiment]) -> None:
        if len(experiments) < 2:
            raise ValueError("Need at least 2 experiments to fit.")
        xs = [statistics.mean(e.complexity_scores) for e in experiments]
        ys = [e.gain for e in experiments]
        n = len(xs)
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        if ss_xx == 0:
            self._slope = 0.0
            self._intercept = mean_y
        else:
            self._slope = ss_xy / ss_xx
            self._intercept = mean_y - self._slope * mean_x
        self._fitted = True

    def predict_gain(self, complexity: float) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_gain().")
        return self._slope * complexity + self._intercept

    def r_squared(self, experiments: List[MetadataExperiment]) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before r_squared().")
        xs = [statistics.mean(e.complexity_scores) for e in experiments]
        ys = [e.gain for e in experiments]
        mean_y = statistics.mean(ys)
        ss_res = sum((y - self.predict_gain(x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        if ss_tot == 0:
            return 1.0
        return 1.0 - ss_res / ss_tot
