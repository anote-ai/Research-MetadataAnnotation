"""Tests for metadataannotation.data — 4 tests."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metadataannotation.core import DocumentDomain, MetadataExperiment
from metadataannotation.data import make_experiment, make_dataset


def test_make_experiment_returns_metadata_experiment():
    exp = make_experiment()
    assert isinstance(exp, MetadataExperiment)


def test_dataset_length():
    dataset = make_dataset(n_per_domain=5)
    assert len(dataset) == 20  # 4 domains × 5


def test_complexity_scores_length():
    exp = make_experiment()
    assert len(exp.complexity_scores) == 5


def test_baseline_less_than_metadata_recall_mostly():
    dataset = make_dataset(n_per_domain=10)
    count_positive = sum(1 for e in dataset if e.metadata_recall > e.baseline_recall)
    # Should always hold since gain is drawn from (0.05, 0.25)
    assert count_positive == len(dataset)
