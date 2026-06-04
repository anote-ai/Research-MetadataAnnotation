from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence


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


# ---------------------------------------------------------------------------
# Document metadata schemas
# ---------------------------------------------------------------------------


@dataclass
class FinancialFilingSchema:
    """Metadata fields for SEC / EDGAR-style financial filings."""

    doc_id: str
    filing_type: str          # e.g. 10-K, 10-Q, 8-K
    fiscal_year_end: str      # ISO date
    company_name: str
    cik: str                  # SEC CIK number
    revenue_usd: Optional[float] = None
    net_income_usd: Optional[float] = None
    auditor: Optional[str] = None
    material_weakness: bool = False
    segments: List[str] = field(default_factory=list)

    @property
    def schema_completeness(self) -> float:
        """Fraction of optional fields that are populated."""
        optional_fields = [
            self.revenue_usd,
            self.net_income_usd,
            self.auditor,
        ]
        populated = sum(1 for f in optional_fields if f is not None)
        return populated / len(optional_fields)


@dataclass
class LegalContractSchema:
    """Metadata fields for commercial contracts."""

    doc_id: str
    contract_type: str        # e.g. NDA, MSA, SOW
    effective_date: str
    governing_law: str
    parties: List[str] = field(default_factory=list)
    termination_clause: bool = False
    liability_cap_usd: Optional[float] = None
    arbitration_required: bool = False
    auto_renewal: bool = False

    @property
    def risk_score(self) -> float:
        """Heuristic risk score [0, 1] based on contract metadata."""
        score = 0.0
        if not self.termination_clause:
            score += 0.3
        if self.liability_cap_usd is None:
            score += 0.3
        if self.arbitration_required:
            score += 0.2
        if self.auto_renewal:
            score += 0.2
        return min(score, 1.0)


@dataclass
class MedicalRecordSchema:
    """Metadata fields for clinical / EHR documents."""

    doc_id: str
    record_type: str          # e.g. discharge_summary, lab_report, progress_note
    patient_id: str
    encounter_date: str
    icd10_codes: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    attending_physician: Optional[str] = None
    de_identified: bool = True

    @property
    def clinical_richness(self) -> float:
        """Proxy for how much structured clinical data is present."""
        score = (
            min(len(self.icd10_codes) / 5, 1.0) * 0.5
            + min(len(self.medications) / 5, 1.0) * 0.3
            + (0.2 if self.attending_physician is not None else 0.0)
        )
        return score


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------


def cohen_kappa(
    annotations_a: Sequence[str],
    annotations_b: Sequence[str],
) -> float:
    """Cohen's kappa for two annotators over the same items.

    Both sequences must be the same length.  Returns a value in [-1, 1];
    1 means perfect agreement, 0 means chance-level agreement.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Both annotation sequences must have the same length.")
    n = len(annotations_a)
    if n == 0:
        return 0.0

    categories = list(dict.fromkeys(list(annotations_a) + list(annotations_b)))
    # Observed agreement
    p_o = sum(a == b for a, b in zip(annotations_a, annotations_b)) / n

    # Expected agreement
    p_e = sum(
        (annotations_a.count(c) / n) * (annotations_b.count(c) / n)
        for c in categories
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(
    ratings: List[List[str]],
) -> float:
    """Fleiss’ kappa for multiple annotators over the same items.

    ``ratings`` is a list of items; each item is a list of category labels
    assigned by each annotator (equal length across items).

    Returns kappa in [-1, 1].
    """
    if not ratings:
        return 0.0
    n_items = len(ratings)
    n_annotators = len(ratings[0])
    if n_annotators < 2:
        raise ValueError("Need at least 2 annotators for Fleiss kappa.")

    # Collect all categories
    categories: List[str] = list(
        dict.fromkeys(label for item in ratings for label in item)
    )
    k = len(categories)
    if k <= 1:
        return 1.0  # All annotators always agree on the single category

    # Count matrix: n_items x k
    count_matrix: List[Dict[str, int]] = []
    for item in ratings:
        counts: Dict[str, int] = {c: 0 for c in categories}
        for label in item:
            counts[label] += 1
        count_matrix.append(counts)

    # P_i: extent of agreement for item i
    p_items: List[float] = []
    for counts in count_matrix:
        total_pairs = n_annotators * (n_annotators - 1)
        if total_pairs == 0:
            p_items.append(1.0)
            continue
        agree_pairs = sum(c * (c - 1) for c in counts.values())
        p_items.append(agree_pairs / total_pairs)
    p_bar = sum(p_items) / n_items

    # p_j: overall fraction of assignments to category j
    p_j: Dict[str, float] = {}
    for c in categories:
        total = sum(counts[c] for counts in count_matrix)
        p_j[c] = total / (n_items * n_annotators)

    # Expected agreement
    p_e = sum(p**2 for p in p_j.values())
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def annotation_confidence_score(
    annotations: List[MetadataAnnotation],
    field: Optional[str] = None,
) -> float:
    """Mean confidence across annotations, optionally filtered to a field."""
    subset = [
        a for a in annotations if (field is None or a.field == field)
    ]
    if not subset:
        return 0.0
    return sum(a.confidence for a in subset) / len(subset)


def inter_annotator_value_agreement(
    annotations: List[MetadataAnnotation],
    doc_id: str,
    field: str,
) -> float:
    """Fraction of annotators who agree on the majority value for a (doc, field)."""
    relevant = [
        a for a in annotations if a.doc_id == doc_id and a.field == field
    ]
    if len(relevant) < 2:
        return 1.0  # trivially agree
    value_counts: Dict[str, int] = {}
    for a in relevant:
        value_counts[a.value] = value_counts.get(a.value, 0) + 1
    majority_count = max(value_counts.values())
    return majority_count / len(relevant)


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
