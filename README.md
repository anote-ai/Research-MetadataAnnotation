# research-metadataannotation

**Hypothesis:** Document structure complexity predicts the value added by metadata-augmented annotation for retrieval tasks.

## Overview

This package benchmarks whether adding structural metadata (section count, table presence, figure presence) to document annotations yields measurable recall gains — and whether those gains correlate with document complexity.

## Experiment Table

| Domain    | Baseline Recall | Metadata Recall | Mean Gain | Mean Complexity |
|-----------|----------------|-----------------|-----------|----------------|
| Finance   | 0.55           | 0.71            | 0.16      | 0.52           |
| Legal     | 0.58           | 0.77            | 0.19      | 0.61           |
| Medical   | 0.52           | 0.69            | 0.17      | 0.55           |
| Technical | 0.60           | 0.80            | 0.20      | 0.67           |

## Domain Breakdown

- **Finance**: Moderately complex documents; earnings reports benefit from table metadata.
- **Legal**: High section count drives complexity; clause-level metadata yields largest gains.
- **Medical**: Mixed figures/tables; imaging reports gain most from figure metadata.
- **Technical**: Highest complexity; API documentation and specs show strongest metadata value.

## Methodology

1. Compute `StructureComplexity.complexity_score` for each document.
2. Annotate with and without metadata (baseline vs. metadata-augmented pipeline).
3. Fit `StructurePredictor` to learn the linear relationship between complexity and recall gain.
4. Evaluate via Pearson r and R².

## Venue

Submitted to **ENCODE 2026** — Workshop on Encoding Knowledge in Neural Systems.

## Citation

```bibtex
@inproceedings{anote2026metadata,
  title     = {Structure Complexity Predicts Metadata Annotation Value in Document Retrieval},
  author    = {Anote AI},
  booktitle = {ENCODE 2026},
  year      = {2026},
}
```
