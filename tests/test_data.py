from __future__ import annotations

from metadataannotation.core import DocumentDomain
from metadataannotation.data import (
    make_annotation,
    make_annotation_set,
    make_complexity,
    make_dataset,
    make_experiment,
    make_financial_filing,
    make_legal_contract,
    make_medical_record,
)


def test_make_complexity_defaults() -> None:
    sc = make_complexity()
    assert sc.doc_id == "d1"
    assert sc.num_sections == 10


def test_make_experiment_recall_bounds() -> None:
    exp = make_experiment()
    assert 0.0 <= exp.baseline_recall <= 1.0
    assert 0.0 <= exp.metadata_recall <= 1.0


def test_make_dataset_length() -> None:
    exps = make_dataset(n_per_domain=3)
    assert len(exps) == 3 * len(DocumentDomain)


def test_make_annotation_defaults() -> None:
    ann = make_annotation()
    assert ann.annotator == "human"
    assert 0.0 <= ann.confidence <= 1.0


def test_make_annotation_set_length() -> None:
    anns = make_annotation_set(values=["A", "B", "A"])
    assert len(anns) == 3


def test_financial_filing_fields() -> None:
    filing = make_financial_filing(filing_type="10-Q", seed=1)
    assert filing.filing_type == "10-Q"
    assert filing.cik.isdigit()


def test_legal_contract_parties() -> None:
    contract = make_legal_contract(seed=0)
    assert len(contract.parties) == 2


def test_medical_record_icd10_codes() -> None:
    record = make_medical_record(seed=2)
    assert isinstance(record.icd10_codes, list)
    assert len(record.icd10_codes) >= 1


def test_medical_record_patient_id_format() -> None:
    record = make_medical_record()
    assert record.patient_id.startswith("PT-")
