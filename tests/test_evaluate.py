from __future__ import annotations

import pytest

from metadataannotation.core import DocumentDomain
from metadataannotation.data import (
    make_annotation,
    make_annotation_set,
    make_dataset,
    make_experiment,
)
from metadataannotation.evaluate import (
    annotation_quality_score,
    correlation_complexity_gain,
    domain_summary,
    fleiss_kappa_from_annotations,
    metadata_gain,
    rank_domains_by_gain,
    structure_prediction_error,
)


def test_metadata_gain_positive() -> None:
    assert metadata_gain(0.5, 0.7) == pytest.approx(0.2)


def test_metadata_gain_negative() -> None:
    assert metadata_gain(0.8, 0.6) == pytest.approx(-0.2)


def test_correlation_complexity_gain_perfect() -> None:
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [x * 2 for x in xs]  # perfect linear
    r = correlation_complexity_gain(xs, ys)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_correlation_complexity_gain_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation_complexity_gain([0.1, 0.2], [0.1])


def test_domain_summary_keys() -> None:
    exps = make_dataset(n_per_domain=3)
    summary = domain_summary(exps)
    assert set(summary.keys()) == {d.value for d in DocumentDomain}


def test_domain_summary_counts() -> None:
    exps = make_dataset(n_per_domain=4)
    summary = domain_summary(exps)
    for domain_data in summary.values():
        assert domain_data["n"] == 4


def test_structure_prediction_error_perfect() -> None:
    preds = [0.1, 0.2, 0.3]
    actuals = [0.1, 0.2, 0.3]
    assert structure_prediction_error(preds, actuals) == pytest.approx(0.0)


def test_structure_prediction_error_empty() -> None:
    assert structure_prediction_error([], []) == 0.0


def test_structure_prediction_error_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        structure_prediction_error([0.1, 0.2], [0.1])


def test_rank_domains_by_gain_order() -> None:
    exps = make_dataset(n_per_domain=5)
    ranked = rank_domains_by_gain(exps)
    gains = [g for _, g in ranked]
    assert gains == sorted(gains, reverse=True)


# ---------------------------------------------------------------------------
# fleiss_kappa_from_annotations
# ---------------------------------------------------------------------------


def test_fleiss_kappa_from_annotations_perfect() -> None:
    anns = make_annotation_set(
        doc_id="doc-1",
        field="category",
        values=["A", "A", "A"],
    )
    k = fleiss_kappa_from_annotations(anns)
    assert k == pytest.approx(1.0)


def test_fleiss_kappa_from_annotations_filtered_field() -> None:
    anns = [
        make_annotation(doc_id="d1", field="type", value="X"),
        make_annotation(doc_id="d1", field="type", value="X"),
        make_annotation(doc_id="d1", field="date", value="2023"),
        make_annotation(doc_id="d1", field="date", value="2024"),
    ]
    k_type = fleiss_kappa_from_annotations(anns, field="type")
    k_date = fleiss_kappa_from_annotations(anns, field="date")
    assert k_type == pytest.approx(1.0)
    assert k_date < 1.0


def test_fleiss_kappa_from_annotations_empty() -> None:
    assert fleiss_kappa_from_annotations([]) == 0.0


# ---------------------------------------------------------------------------
# annotation_quality_score
# ---------------------------------------------------------------------------


def test_annotation_quality_score_range() -> None:
    anns = make_annotation_set(values=["A", "A", "B"])
    score = annotation_quality_score(anns)
    assert 0.0 <= score <= 1.0


def test_annotation_quality_score_perfect() -> None:
    anns = make_annotation_set(
        values=["10-K", "10-K", "10-K"],
        seed=0,
    )
    # Force confidence to 1.0 by overriding
    for a in anns:
        a.confidence = 1.0
    score = annotation_quality_score(anns)
    assert score == pytest.approx(1.0)
