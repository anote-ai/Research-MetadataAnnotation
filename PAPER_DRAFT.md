# Structure Complexity and Metadata Annotation Value: A Status Report and Draft Skeleton

**Status:** Early draft / skeleton. Sections marked **(projected, pending full
experiment run)** are aspirational and have NOT been measured; sections
marked **(measured)** were computed by actually running the code in this
repository (`experiments/exp0_baseline.py`) and are reproducible by
re-running that script.

---

## Abstract (draft)

Enterprise document retrieval systems increasingly augment raw text with
structured metadata (section counts, table/figure presence, schema
fields) before indexing or prompting an LLM. We ask whether a document's
**structural complexity** predicts the **retrieval recall gain** obtained
from such metadata augmentation, across four domains: finance, legal,
medical, and technical documents. We introduce a lightweight complexity
score and a linear predictor relating complexity to gain, along with
standard inter-annotator agreement tooling (Cohen's and Fleiss' kappa)
for validating metadata annotation quality. **(projected)** We expect
that domains with higher structural complexity (legal, technical) show
larger metadata gains than lower-complexity domains (finance). **(measured)**
On the synthetic dataset currently shipped with this codebase, no such
relationship is observed (Pearson r ≈ 0.00–0.03 in magnitude across
sample sizes), because the synthetic generator does not encode a
complexity-gain relationship — this result reflects the data, not the
method, and the central hypothesis remains untested on real documents.

## 1. Introduction

Manual metadata enrichment is a well-known component of data governance
and retrieval-augmented generation pipelines, but it is unclear which
documents benefit most from it. We hypothesize that documents with more
structural complexity (more sections, longer sections, presence of
tables/figures) see larger gains from metadata augmentation because
metadata provides cheap signal where raw-text retrieval struggles most.

(See companion `DESIGN_DOC.md` in this repository for a broader research
agenda — including a metadata-hallucination-rate benchmark across
database/document schema types — that has not yet been implemented in
code. This paper draft covers only the structure-complexity-vs-gain
strand of work that the current `src/metadataannotation` package actually
implements.)

## 2. Method

### 2.1 Complexity score

For a document `d` with `num_sections`, `avg_section_length`,
`has_tables`, `has_figures`:

```
complexity(d) = 0.3 * min(num_sections / 20, 1)
              + 0.3 * min(avg_section_length / 500, 1)
              + 0.2 * has_tables
              + 0.2 * has_figures
```

This is implemented in `src/metadataannotation/core.py::StructureComplexity`.
**(projected)** The weights (0.3/0.3/0.2/0.2) are currently hand-chosen
and have not been validated against human judgments of "complexity" —
a calibration study against human ratings is future work.

### 2.2 Metadata gain

`gain(e) = metadata_recall(e) - baseline_recall(e)` for an experiment
`e` (implemented in `MetadataExperiment.gain`).

### 2.3 Predictor

A single-variable OLS regression of `gain` on mean `complexity_score`
across a document's sections (`StructurePredictor`), fit in closed form
via `statistics.mean` and sum-of-squares — no external ML dependency
required for this part, despite scikit-learn being listed as a
dependency.

### 2.4 Inter-annotator agreement

Cohen's kappa (two annotators) and Fleiss' kappa (≥2 annotators) are
implemented from scratch in `core.py` and used to validate the
reliability of any human or LLM-vs-human metadata labeling underlying
the recall numbers. **(projected)** These have not yet been applied to
any actual annotation campaign — there is no real annotated dataset in
this repository yet.

## 3. Data

**(measured)** The dataset currently used by `scripts/run_demo.py` and
`experiments/exp0_baseline.py` is fully synthetic
(`src/metadataannotation/data.py::make_dataset`). For each of 4 domains
(finance, legal, medical, technical) it draws, independently and
uniformly at random:

- `baseline_recall ~ U(0.4, 0.7)`
- `gain ~ U(0.05, 0.25)`, with `metadata_recall = min(baseline + gain, 1)`
- 5 `complexity_scores ~ U(0.1, 0.9)` per "document"

Because `gain` and `complexity_scores` are drawn independently, **the
generator does not encode any true relationship between complexity and
gain by construction.** Any correlation observed at small sample sizes
is sampling noise, not signal.

**(projected, pending full experiment run)** Real data collection would
require: (a) a corpus of real finance/legal/medical/technical documents,
(b) a real retrieval system run with and without structured metadata
indexing, (c) real recall@k measurements against human-labeled relevant
passages for a query set. None of this exists in the repo yet.

## 4. Results

### 4.1 Baseline correlation check (measured)

Running `experiments/exp0_baseline.py` (seed=42) against the synthetic
generator:

| Run | n (total) | Pearson r (complexity vs. gain) | R² |
|---|---|---|---|
| Small (5/domain, matches original demo) | 20 | -0.167 | 0.028 |
| Large (200/domain) | 800 | 0.005 | 0.00002 |

Per-domain mean gain and mean complexity (large run, n=200/domain):

| Domain | Mean gain | Mean complexity |
|---|---|---|
| Finance | 0.155 | 0.506 |
| Legal | 0.147 | 0.511 |
| Medical | 0.151 | 0.485 |
| Technical | 0.152 | 0.493 |

These values are reproducible by running `python experiments/exp0_baseline.py`
and inspecting `results/exp0_baseline.json`.

**Interpretation:** At both sample sizes the correlation is statistically
indistinguishable from zero, and the small negative r at n=20 flips sign
and shrinks toward zero at n=800 — consistent with pure sampling noise
around a true correlation of 0, exactly as expected given how the data
generator works (see §3). This also means the experiment table in the
current `README.md` (showing complexity and recall rising together
across domains) is **not** reproduced by re-running the shipped code on
the documented default settings, and should be revised or clearly
labeled as a target/illustrative table rather than a measured result.

### 4.2 MHR, schema adherence, MetaHal detector results

**(projected, pending full experiment run)** Not implemented. See
`DESIGN_DOC.md` Experiments 1–5 for the full specification. No code in
this repository currently calls any LLM API, computes a hallucination
rate, or trains a detector.

## 5. Discussion / Limitations

- The core measurement infrastructure (complexity scoring, regression
  fitting, agreement statistics) is implemented and unit-tested, and can
  be pointed at real data once it exists.
- The headline hypothesis — complexity predicts metadata gain — is
  currently **untested on real data** and, on the available synthetic
  data, shows no effect by construction.
- The broader hallucination-rate research agenda in `DESIGN_DOC.md`
  (MHR, MetaHal, schema adherence, context-window ablation) has no
  corresponding implementation in `src/` yet; this draft does not make
  any claims about that work.

## 6. Next Steps

1. Replace synthetic data with a small real pilot corpus (~50 docs across
   the 4 domains) and a real retrieval baseline (e.g., BM25 or a simple
   embedding retriever) run with/without metadata augmentation.
2. Re-run `experiments/exp0_baseline.py`-style analysis on real recall
   numbers and report whatever correlation is actually found.
3. If the complexity-gain link is one of the listed `DESIGN_DOC.md`
   experiments instead, scope a minimal MHR pilot (Experiment 0 in that
   doc: ~200 column-description annotations from one LLM) as the next
   implementation milestone.

## Target Venue

ENCODE 2026 (per `README.md`) for the structure-complexity strand;
VLDB 2027 / EMNLP 2026 / CIDR 2027 (per `DESIGN_DOC.md`) for the
hallucination-benchmark strand, contingent on the above work being done.
