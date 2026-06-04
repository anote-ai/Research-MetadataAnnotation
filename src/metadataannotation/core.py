"""Core data structures and logic for the metadata annotation study."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocumentDomain(str, Enum):
    """Document domains used in the study."""

    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"


@dataclass
class StructureComplexity:
    """Structural features of a document."""

    doc_id: str
    num_sections: int
    avg_section_length: float
    has_tables: bool
    has_figures: bool
    complexity_score: float = 0.0

    def __post_init__(self) -> None:
        if self.complexity_score == 0.0:
            self.complexity_score = compute_complexity_score(
                self.num_sections,
                self.avg_section_length,
                self.has_tables,
                self.has_figures,
            )


@dataclass
class MetadataAnnotation:
    """A single metadata annotation on a document."""

    doc_id: str
    field: str
    value: str
    confidence: float
    annotator: str  # "llm" | "human"

    def __post_init__(self) -> None:
        if self.annotator not in {"llm", "human"}:
            raise ValueError(f"annotator must be 'llm' or 'human', got {self.annotator!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class MetadataExperiment:
    """Results of a single domain experiment."""

    domain: DocumentDomain
    baseline_recall: float
    metadata_recall: float
    complexity_scores: list[float] = field(default_factory=list)


def compute_complexity_score(
    num_sections: int,
    avg_section_length: float,
    has_tables: bool,
    has_figures: bool,
) -> float:
    """Compute a weighted complexity score for a document.

    Formula:
        score = 0.4 * norm_sections + 0.3 * norm_length + 0.2 * tables + 0.1 * figures

    where norm_sections = num_sections / 50 and norm_length = avg_section_length / 1000.
    The raw score is clamped to [0, 1].
    """
    norm_sections = min(num_sections / 50.0, 1.0)
    norm_length = min(avg_section_length / 1000.0, 1.0)
    score = (
        0.4 * norm_sections
        + 0.3 * norm_length
        + 0.2 * float(has_tables)
        + 0.1 * float(has_figures)
    )
    return round(min(max(score, 0.0), 1.0), 6)


class StructurePredictor:
    """Predict metadata gain from document structure complexity."""

    def __init__(self) -> None:
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._fitted: bool = False

    def fit(self, experiments: list[MetadataExperiment]) -> None:
        """Fit a simple linear model: gain ~ complexity_score."""
        from metadataannotation.evaluate import (
            metadata_gain,
            correlation_complexity_gain,
        )
        import math

        all_complexity: list[float] = []
        all_gains: list[float] = []
        for exp in experiments:
            gain = metadata_gain(exp.baseline_recall, exp.metadata_recall)
            for cs in exp.complexity_scores:
                all_complexity.append(cs)
                all_gains.append(gain)

        if len(all_complexity) < 2:  # noqa: PLR2004
            self._fitted = True
            return

        n = len(all_complexity)
        mean_x = sum(all_complexity) / n
        mean_y = sum(all_gains) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(all_complexity, all_gains))
        den = sum((x - mean_x) ** 2 for x in all_complexity)
        self._slope = num / den if den != 0 else 0.0
        self._intercept = mean_y - self._slope * mean_x
        self._fitted = True

    def predict_gain(self, complexity: float) -> float:
        """Predict metadata gain for a given complexity score."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_gain().")
        return self._slope * complexity + self._intercept
