"""Tests for metadataannotation.core."""

import pytest
from metadataannotation.core import (
    DocumentDomain,
    StructureComplexity,
    MetadataAnnotation,
    MetadataExperiment,
    compute_complexity_score,
    StructurePredictor,
)


def test_document_domain_values():
    assert DocumentDomain.FINANCE.value == "finance"
    assert DocumentDomain.LEGAL.value == "legal"
    assert DocumentDomain.MEDICAL.value == "medical"
    assert DocumentDomain.TECHNICAL.value == "technical"


def test_document_domain_enum_members():
    members = {d.value for d in DocumentDomain}
    assert members == {"finance", "legal", "medical", "technical"}


def test_compute_complexity_score_known():
    score = compute_complexity_score(25, 500.0, True, False)
    # norm_sections = 0.5, norm_length = 0.5, tables = 1, figures = 0
    expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.2 * 1.0 + 0.1 * 0.0
    assert abs(score - expected) < 1e-5


def test_compute_complexity_score_clamped():
    score = compute_complexity_score(100, 5000.0, True, True)
    assert score == 1.0


def test_compute_complexity_score_zero():
    score = compute_complexity_score(0, 0.0, False, False)
    assert score == 0.0


def test_structure_complexity_construction():
    sc = StructureComplexity(
        doc_id="doc1",
        num_sections=10,
        avg_section_length=200.0,
        has_tables=False,
        has_figures=True,
        complexity_score=0.5,
    )
    assert sc.doc_id == "doc1"
    assert sc.num_sections == 10
    assert sc.complexity_score == 0.5


def test_structure_complexity_auto_score():
    sc = StructureComplexity(
        doc_id="doc2",
        num_sections=25,
        avg_section_length=500.0,
        has_tables=True,
        has_figures=False,
    )
    expected = compute_complexity_score(25, 500.0, True, False)
    assert abs(sc.complexity_score - expected) < 1e-5


def test_metadata_annotation_valid():
    ann = MetadataAnnotation(
        doc_id="doc1", field="title", value="Annual Report", confidence=0.9, annotator="llm"
    )
    assert ann.annotator == "llm"
    assert ann.confidence == 0.9


def test_metadata_annotation_invalid_annotator():
    with pytest.raises(ValueError, match="annotator"):
        MetadataAnnotation(
            doc_id="doc1", field="title", value="x", confidence=0.5, annotator="robot"
        )


def test_metadata_annotation_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        MetadataAnnotation(
            doc_id="doc1", field="title", value="x", confidence=1.5, annotator="human"
        )


def test_metadata_experiment_construction():
    exp = MetadataExperiment(
        domain=DocumentDomain.FINANCE,
        baseline_recall=0.6,
        metadata_recall=0.75,
        complexity_scores=[0.3, 0.5, 0.7],
    )
    assert exp.domain == DocumentDomain.FINANCE
    assert len(exp.complexity_scores) == 3


def test_structure_predictor_fit_and_predict():
    exps = [
        MetadataExperiment(DocumentDomain.FINANCE, 0.5, 0.7, [0.2, 0.4, 0.6]),
        MetadataExperiment(DocumentDomain.LEGAL, 0.6, 0.8, [0.5, 0.7, 0.9]),
    ]
    predictor = StructurePredictor()
    predictor.fit(exps)
    gain = predictor.predict_gain(0.5)
    assert isinstance(gain, float)


def test_structure_predictor_not_fitted():
    predictor = StructurePredictor()
    with pytest.raises(RuntimeError, match="fit"):
        predictor.predict_gain(0.5)
