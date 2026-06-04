"""Tests for metadataannotation.evaluate."""

import math
import pytest
from metadataannotation.core import DocumentDomain, MetadataExperiment
from metadataannotation.evaluate import (
    metadata_gain,
    correlation_complexity_gain,
    domain_summary,
    structure_prediction_error,
)


def test_metadata_gain_positive():
    assert abs(metadata_gain(0.6, 0.75) - 0.15) < 1e-9


def test_metadata_gain_zero():
    assert metadata_gain(0.7, 0.7) == 0.0


def test_metadata_gain_negative():
    assert metadata_gain(0.8, 0.7) == pytest.approx(-0.1)


def test_correlation_complexity_gain_perfect():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    r = correlation_complexity_gain(xs, ys)
    assert abs(r - 1.0) < 1e-9


def test_correlation_complexity_gain_negative():
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 2.0, 1.0]
    r = correlation_complexity_gain(xs, ys)
    assert abs(r - (-1.0)) < 1e-9


def test_correlation_mismatch_length():
    with pytest.raises(ValueError):
        correlation_complexity_gain([1.0, 2.0], [1.0])


def test_domain_summary():
    exps = [
        MetadataExperiment(DocumentDomain.FINANCE, 0.5, 0.7, [0.3, 0.6]),
        MetadataExperiment(DocumentDomain.FINANCE, 0.6, 0.8, [0.4, 0.7]),
        MetadataExperiment(DocumentDomain.MEDICAL, 0.7, 0.75, [0.2]),
    ]
    summary = domain_summary(exps)
    assert "finance" in summary
    assert "medical" in summary
    assert abs(summary["finance"]["mean_gain"] - 0.2) < 1e-9
    assert abs(summary["medical"]["mean_gain"] - 0.05) < 1e-9


def test_structure_prediction_error_zero():
    assert structure_prediction_error([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0


def test_structure_prediction_error_known():
    mae = structure_prediction_error([0.0, 0.5, 1.0], [0.1, 0.4, 0.9])
    assert abs(mae - 0.1) < 1e-9


def test_structure_prediction_error_mismatch():
    with pytest.raises(ValueError):
        structure_prediction_error([0.1], [0.1, 0.2])
