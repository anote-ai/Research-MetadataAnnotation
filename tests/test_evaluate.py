"""Tests for metadataannotation.evaluate — 7 tests."""
from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metadataannotation.core import DocumentDomain, MetadataExperiment
from metadataannotation.evaluate import (
    metadata_gain,
    pearson_r,
    domain_summary,
    structure_prediction_error,
    rank_domains_by_gain,
)


def test_metadata_gain_basic():
    assert abs(metadata_gain(0.5, 0.7) - 0.2) < 1e-9


def test_pearson_r_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert abs(pearson_r(xs, ys) - 1.0) < 1e-9


def test_pearson_r_anti_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert abs(pearson_r(xs, ys) + 1.0) < 1e-9


def test_pearson_r_constant_y():
    xs = [1.0, 2.0, 3.0]
    ys = [5.0, 5.0, 5.0]
    assert pearson_r(xs, ys) == 0.0


def test_domain_summary_structure():
    experiments = [
        MetadataExperiment(DocumentDomain.FINANCE, 0.5, 0.7, [0.5]),
        MetadataExperiment(DocumentDomain.LEGAL, 0.6, 0.8, [0.4]),
    ]
    summary = domain_summary(experiments)
    assert "finance" in summary
    assert "legal" in summary
    assert "mean_gain" in summary["finance"]
    assert "mean_complexity" in summary["finance"]
    assert summary["finance"]["n"] == 1


def test_structure_prediction_error_perfect():
    preds = [0.1, 0.2, 0.3]
    actuals = [0.1, 0.2, 0.3]
    assert structure_prediction_error(preds, actuals) == 0.0


def test_rank_domains_by_gain_sorted_descending():
    experiments = [
        MetadataExperiment(DocumentDomain.FINANCE, 0.5, 0.6, [0.5]),  # gain=0.1
        MetadataExperiment(DocumentDomain.LEGAL, 0.4, 0.7, [0.4]),    # gain=0.3
        MetadataExperiment(DocumentDomain.MEDICAL, 0.5, 0.65, [0.5]), # gain=0.15
    ]
    ranked = rank_domains_by_gain(experiments)
    gains = [g for _, g in ranked]
    assert gains == sorted(gains, reverse=True)
    assert ranked[0][0] == "legal"
