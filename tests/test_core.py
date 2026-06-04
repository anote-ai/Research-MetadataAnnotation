"""Tests for metadataannotation.core — 10 tests."""
from __future__ import annotations

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metadataannotation.core import (
    DocumentDomain,
    StructureComplexity,
    MetadataAnnotation,
    MetadataExperiment,
    StructurePredictor,
)
from metadataannotation.data import make_complexity


def test_document_domain_values():
    assert DocumentDomain.FINANCE == "finance"
    assert DocumentDomain.LEGAL == "legal"
    assert DocumentDomain.MEDICAL == "medical"
    assert DocumentDomain.TECHNICAL == "technical"


def test_complexity_score_range():
    sc = StructureComplexity("d1", 10, 250.0, True, False)
    assert 0.0 <= sc.complexity_score <= 1.0


def test_complexity_score_all_zeros():
    sc = StructureComplexity("d0", 0, 0.0, False, False)
    assert sc.complexity_score == 0.0


def test_complexity_score_all_max():
    sc = StructureComplexity("d_max", 20, 500.0, True, True)
    assert abs(sc.complexity_score - 1.0) < 1e-9


def test_metadata_annotation_valid():
    ann = MetadataAnnotation("d1", "author", "Alice", 0.9, "llm")
    assert ann.annotator == "llm"
    assert ann.confidence == 0.9


def test_metadata_annotation_invalid_annotator():
    with pytest.raises(ValueError):
        MetadataAnnotation("d1", "author", "Alice", 0.9, "robot")


def test_metadata_experiment_gain():
    exp = MetadataExperiment(DocumentDomain.FINANCE, 0.5, 0.7, [0.3, 0.6])
    assert abs(exp.gain - 0.2) < 1e-9


def test_structure_predictor_fit_predict():
    # Perfect linear case: gain = 0.5 * complexity
    experiments = [
        MetadataExperiment(DocumentDomain.FINANCE, 0.0, 0.0 + 0.5 * c, [c])
        for c in [0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    pred = StructurePredictor()
    pred.fit(experiments)
    predicted = pred.predict_gain(0.5)
    assert abs(predicted - 0.25) < 0.01


def test_structure_predictor_r_squared_perfect():
    experiments = [
        MetadataExperiment(DocumentDomain.LEGAL, 0.0, 0.1 * c, [c])
        for c in [0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    pred = StructurePredictor()
    pred.fit(experiments)
    r2 = pred.r_squared(experiments)
    assert r2 > 0.99


def test_make_complexity_defaults():
    sc = make_complexity()
    assert sc.doc_id == "d1"
    assert sc.num_sections == 10
    assert sc.avg_section_length == 250.0
    assert sc.has_tables is False
    assert sc.has_figures is False
