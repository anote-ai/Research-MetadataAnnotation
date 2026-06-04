# research-metadataannotation

**Does Document Structure Predict Metadata-Annotation Value in RAG?**

## Research Hypothesis

LLM-generated metadata annotations provide greater retrieval recall improvements for documents
with heterogeneous, complex structure (e.g., SEC 10-K filings) than for documents with uniform
structure (e.g., clinical notes). Concretely, **structure complexity is a reliable predictor of
metadata annotation gain** across domains.

## Motivation

Metadata annotation in RAG pipelines is expensive. Not all document corpora benefit equally.
This study quantifies *when* metadata annotation is worth the cost by correlating structural
features with retrieval improvement.

## Experimental Design

| Domain    | Structure Type        | Avg Complexity | Baseline Recall | +Metadata Recall |
|-----------|-----------------------|---------------|-----------------|------------------|
| Finance   | Heterogeneous (10-K)  | ~0.72         | TBD             | TBD              |
| Legal     | Semi-structured       | ~0.58         | TBD             | TBD              |
| Medical   | Uniform (SOAP notes)  | ~0.31         | TBD             | TBD              |
| Technical | Mixed (manuals)       | ~0.55         | TBD             | TBD              |

## Complexity Score

The complexity score is a weighted combination of structural features:

```
score = 0.4 * (num_sections / 50)
      + 0.3 * (avg_section_length / 1000)
      + 0.2 * has_tables
      + 0.1 * has_figures
```

Score is clamped to [0, 1].

## Results Template

| Domain    | Pearson r (complexity vs. gain) | p-value | Conclusion              |
|-----------|---------------------------------|---------|-------------------------|
| Finance   | TBD                             | TBD     | TBD                     |
| Legal     | TBD                             | TBD     | TBD                     |
| Medical   | TBD                             | TBD     | TBD                     |
| Technical | TBD                             | TBD     | TBD                     |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from metadataannotation.core import (
    DocumentDomain, StructureComplexity, MetadataExperiment,
    compute_complexity_score, StructurePredictor,
)
from metadataannotation.evaluate import correlation_complexity_gain, domain_summary

# Compute complexity for a document
score = compute_complexity_score(num_sections=30, avg_section_length=600.0,
                                  has_tables=True, has_figures=False)

# Create an experiment record
exp = MetadataExperiment(
    domain=DocumentDomain.FINANCE,
    baseline_recall=0.62,
    metadata_recall=0.79,
    complexity_scores=[0.45, 0.61, 0.72],
)

# Fit predictor
predictor = StructurePredictor()
predictor.fit([exp])
predicted_gain = predictor.predict_gain(0.65)
```

## Citation

```bibtex
@misc{anote2024metadataannotation,
  title  = {Does Document Structure Predict Metadata-Annotation Value in RAG?},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-metadataannotation},
}
```
