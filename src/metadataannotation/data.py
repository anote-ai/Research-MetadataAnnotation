from __future__ import annotations

import random
from typing import List, Optional

from .core import (
    DocumentDomain,
    FinancialFilingSchema,
    LegalContractSchema,
    MedicalRecordSchema,
    MetadataAnnotation,
    MetadataExperiment,
    StructureComplexity,
)


def make_complexity(
    doc_id: str = "d1",
    num_sections: int = 10,
    avg_section_length: float = 250.0,
    has_tables: bool = False,
    has_figures: bool = False,
) -> StructureComplexity:
    return StructureComplexity(
        doc_id=doc_id,
        num_sections=num_sections,
        avg_section_length=avg_section_length,
        has_tables=has_tables,
        has_figures=has_figures,
    )


def make_experiment(
    domain: DocumentDomain = DocumentDomain.FINANCE,
    seed: int = 42,
) -> MetadataExperiment:
    rng = random.Random(seed)
    baseline = rng.uniform(0.4, 0.7)
    gain = rng.uniform(0.05, 0.25)
    metadata_recall = min(baseline + gain, 1.0)
    complexity_scores = [rng.uniform(0.1, 0.9) for _ in range(5)]
    return MetadataExperiment(
        domain=domain,
        baseline_recall=baseline,
        metadata_recall=metadata_recall,
        complexity_scores=complexity_scores,
    )


def make_dataset(
    n_per_domain: int = 5,
    seed: int = 42,
) -> List[MetadataExperiment]:
    domains = list(DocumentDomain)
    experiments: List[MetadataExperiment] = []
    base_rng = random.Random(seed)
    for domain in domains:
        for _i in range(n_per_domain):
            exp_seed = base_rng.randint(0, 10_000)
            experiments.append(make_experiment(domain=domain, seed=exp_seed))
    return experiments


def make_annotation(
    doc_id: str = "doc-001",
    field: str = "category",
    value: str = "invoice",
    confidence: float = 0.9,
    annotator: str = "human",
) -> MetadataAnnotation:
    return MetadataAnnotation(
        doc_id=doc_id,
        field=field,
        value=value,
        confidence=confidence,
        annotator=annotator,
    )


def make_annotation_set(
    doc_id: str = "doc-001",
    field: str = "filing_type",
    values: Optional[List[str]] = None,
    annotator_types: Optional[List[str]] = None,
    seed: int = 0,
) -> List[MetadataAnnotation]:
    """Create multiple annotations for the same (doc, field) from different annotators."""
    rng = random.Random(seed)
    if values is None:
        values = ["10-K", "10-Q", "10-K"]
    if annotator_types is None:
        annotator_types = ["human"] * len(values)
    return [
        MetadataAnnotation(
            doc_id=doc_id,
            field=field,
            value=v,
            confidence=rng.uniform(0.7, 1.0),
            annotator=at,
        )
        for v, at in zip(values, annotator_types)
    ]


# ---------------------------------------------------------------------------
# Document schema factories
# ---------------------------------------------------------------------------


def make_financial_filing(
    doc_id: str = "filing-001",
    filing_type: str = "10-K",
    seed: int = 0,
) -> FinancialFilingSchema:
    rng = random.Random(seed)
    return FinancialFilingSchema(
        doc_id=doc_id,
        filing_type=filing_type,
        fiscal_year_end=f"{2020 + rng.randint(0, 4)}-12-31",
        company_name=rng.choice(["Acme Corp", "Globex Inc", "Initech LLC"]),
        cik=str(rng.randint(100_000, 999_999)),
        revenue_usd=rng.uniform(1e6, 1e9),
        net_income_usd=rng.uniform(-1e7, 1e8),
        auditor=rng.choice(["Deloitte", "PwC", "KPMG", "EY"]),
        material_weakness=rng.random() < 0.1,
        segments=rng.sample(["North America", "EMEA", "APAC", "LatAm"], k=rng.randint(1, 3)),
    )


def make_legal_contract(
    doc_id: str = "contract-001",
    contract_type: str = "MSA",
    seed: int = 0,
) -> LegalContractSchema:
    rng = random.Random(seed)
    return LegalContractSchema(
        doc_id=doc_id,
        contract_type=contract_type,
        effective_date=f"{2021 + rng.randint(0, 3)}-0{rng.randint(1, 9)}-01",
        governing_law=rng.choice(["New York", "Delaware", "California", "English Law"]),
        parties=["Party A", "Party B"],
        termination_clause=rng.random() > 0.3,
        liability_cap_usd=rng.choice([None, 1e5, 5e5, 1e6]),
        arbitration_required=rng.random() < 0.4,
        auto_renewal=rng.random() < 0.3,
    )


def make_medical_record(
    doc_id: str = "record-001",
    record_type: str = "discharge_summary",
    seed: int = 0,
) -> MedicalRecordSchema:
    rng = random.Random(seed)
    icd10_pool = ["I10", "E11.9", "J18.9", "K92.1", "M54.5", "F32.1"]
    med_pool = ["metformin", "lisinopril", "atorvastatin", "amoxicillin", "ibuprofen"]
    return MedicalRecordSchema(
        doc_id=doc_id,
        record_type=record_type,
        patient_id=f"PT-{rng.randint(10000, 99999)}",
        encounter_date=f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        icd10_codes=rng.sample(icd10_pool, k=rng.randint(1, 4)),
        medications=rng.sample(med_pool, k=rng.randint(0, 3)),
        attending_physician=rng.choice(["Dr. Smith", "Dr. Patel", None]),
        de_identified=True,
    )
