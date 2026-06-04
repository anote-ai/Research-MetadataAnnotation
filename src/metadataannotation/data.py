from __future__ import annotations

import random
from typing import List

from .core import (
    DocumentDomain,
    StructureComplexity,
    MetadataExperiment,
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
        for i in range(n_per_domain):
            exp_seed = base_rng.randint(0, 10_000)
            experiments.append(make_experiment(domain=domain, seed=exp_seed))
    return experiments
