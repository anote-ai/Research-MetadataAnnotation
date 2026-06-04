from __future__ import annotations

import pytest

from metadataannotation.core import (
    DocumentDomain,
    FinancialFilingSchema,
    LegalContractSchema,
    MedicalRecordSchema,
    MetadataAnnotation,
    MetadataExperiment,
    StructureComplexity,
    StructurePredictor,
    annotation_confidence_score,
    cohen_kappa,
    fleiss_kappa,
    inter_annotator_value_agreement,
)
from metadataannotation.data import (
    make_annotation,
    make_annotation_set,
    make_dataset,
    make_experiment,
    make_financial_filing,
    make_legal_contract,
    make_medical_record,
)


# ---------------------------------------------------------------------------
# StructureComplexity
# ---------------------------------------------------------------------------


def test_complexity_score_range() -> None:
    sc = StructureComplexity(
        doc_id="d1",
        num_sections=10,
        avg_section_length=250.0,
        has_tables=True,
        has_figures=False,
    )
    assert 0.0 <= sc.complexity_score <= 1.0


def test_complexity_score_max() -> None:
    sc = StructureComplexity(
        doc_id="d",
        num_sections=20,
        avg_section_length=500.0,
        has_tables=True,
        has_figures=True,
    )
    assert sc.complexity_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetadataAnnotation
# ---------------------------------------------------------------------------


def test_annotation_invalid_annotator() -> None:
    with pytest.raises(ValueError, match="annotator"):
        MetadataAnnotation(
            doc_id="d", field="f", value="v", confidence=0.8, annotator="robot"
        )


def test_annotation_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MetadataAnnotation(
            doc_id="d", field="f", value="v", confidence=1.5, annotator="human"
        )


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement() -> None:
    labels = ["A", "B", "A", "C", "B"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_cohen_kappa_single_category() -> None:
    a = ["A", "A", "A"]
    b = ["A", "A", "A"]
    # p_e == 1.0 -> returns 1.0
    assert cohen_kappa(a, b) == 1.0


def test_cohen_kappa_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["A", "B"], ["A"])


def test_cohen_kappa_empty() -> None:
    assert cohen_kappa([], []) == 0.0


def test_cohen_kappa_range() -> None:
    a = ["A", "B", "A", "B", "A"]
    b = ["B", "A", "B", "A", "B"]
    k = cohen_kappa(a, b)
    assert -1.0 <= k <= 1.0


# ---------------------------------------------------------------------------
# Fleiss kappa
# ---------------------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement() -> None:
    # All annotators always pick "A"
    ratings = [["A", "A", "A"] for _ in range(5)]
    k = fleiss_kappa(ratings)
    assert k == pytest.approx(1.0)


def test_fleiss_kappa_random_disagreement() -> None:
    # Completely random labels from two categories
    import random
    rng = random.Random(0)
    ratings = [[rng.choice(["A", "B"]) for _ in range(3)] for _ in range(20)]
    k = fleiss_kappa(ratings)
    assert -1.0 <= k <= 1.0


def test_fleiss_kappa_empty() -> None:
    assert fleiss_kappa([]) == 0.0


def test_fleiss_kappa_single_category() -> None:
    ratings = [["X", "X"] for _ in range(4)]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# annotation_confidence_score
# ---------------------------------------------------------------------------


def test_annotation_confidence_score_all() -> None:
    annotations = [make_annotation(confidence=c) for c in [0.8, 0.9, 1.0]]
    score = annotation_confidence_score(annotations)
    assert score == pytest.approx(0.9)


def test_annotation_confidence_score_filtered() -> None:
    a1 = make_annotation(field="type", confidence=0.8)
    a2 = make_annotation(field="date", confidence=0.6)
    score = annotation_confidence_score([a1, a2], field="type")
    assert score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# inter_annotator_value_agreement
# ---------------------------------------------------------------------------


def test_inter_annotator_agreement_unanimous() -> None:
    anns = make_annotation_set(values=["10-K", "10-K", "10-K"])
    score = inter_annotator_value_agreement(anns, "doc-001", "filing_type")
    assert score == pytest.approx(1.0)


def test_inter_annotator_agreement_split() -> None:
    anns = make_annotation_set(values=["10-K", "10-Q"])
    score = inter_annotator_value_agreement(anns, "doc-001", "filing_type")
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------


def test_financial_filing_completeness() -> None:
    filing = make_financial_filing(seed=7)
    assert 0.0 <= filing.schema_completeness <= 1.0


def test_legal_contract_risk_score() -> None:
    contract = make_legal_contract(seed=3)
    assert 0.0 <= contract.risk_score <= 1.0


def test_medical_record_clinical_richness() -> None:
    record = make_medical_record(seed=5)
    assert 0.0 <= record.clinical_richness <= 1.0


def test_medical_record_de_identified() -> None:
    record = make_medical_record()
    assert record.de_identified is True


# ---------------------------------------------------------------------------
# StructurePredictor
# ---------------------------------------------------------------------------


def test_predictor_fit_and_predict() -> None:
    exps = make_dataset(n_per_domain=5, seed=0)
    predictor = StructurePredictor()
    predictor.fit(exps)
    pred = predictor.predict_gain(0.5)
    assert isinstance(pred, float)


def test_predictor_r_squared_range() -> None:
    exps = make_dataset(n_per_domain=8)
    predictor = StructurePredictor()
    predictor.fit(exps)
    r2 = predictor.r_squared(exps)
    assert -1.0 <= r2 <= 1.0


def test_predictor_requires_fit() -> None:
    predictor = StructurePredictor()
    with pytest.raises(RuntimeError):
        predictor.predict_gain(0.5)
