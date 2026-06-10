# MetadataAnnotation — Research Design Document

## Goal

Build and evaluate a benchmark for LLM-based metadata annotation quality for enterprise data catalogs — measuring whether AI-generated column descriptions are accurate, useful, and free of hallucinations — and derive practical deployment guidelines for data teams.

## Objective

1. Construct a benchmark of 500+ real-world-style catalog entries (tables, columns, datasets) with human-authored gold descriptions and LLM-generated candidates
2. Evaluate 5+ LLMs on description quality, factual accuracy, and hallucination rate
3. Produce the "Metadata Annotation Reliability Index" — a per-schema-type deployment recommendation tool

## Background / Motivation

Every major data platform (Databricks, Snowflake, dbt) is pushing AI-generated metadata descriptions as a product feature. But there are no public benchmarks measuring whether these descriptions are actually accurate. The hallucination risk is real: a wrong description of a financial table column could cause analysts to misuse data, leading to bad business decisions.

## Experimental Design

### Baseline Experiment

**Collect human-authored descriptions for 100 publicly-available database tables. Evaluate 3 LLMs (GPT-4o, Claude, Gemini) with ROUGE and BERTScore vs. human descriptions.**

- Metric: ROUGE-L, BERTScore, human preference rate
- Purpose: establish baseline quality and confirm human preference correlation with automatic metrics
- Expected result: ROUGE-L ≈ 0.25–0.35; BERTScore ≈ 0.85–0.90; human preference for LLM descriptions ≈ 55–65%

### Test Experiment 1: Hallucination Rate by Schema Type

For 5 schema types (financial, medical, e-commerce, IoT sensor, HR), have domain experts rate LLM-generated descriptions as: accurate, misleading, or hallucinated.

**Expected result:** hallucination rate varies dramatically — financial tables ~8%; IoT sensor tables ~35%. Schema name opacity is the primary predictor of hallucination risk.

### Test Experiment 2: Context Window Effects

Compare LLM description quality with: (1) column name only, (2) + table name, (3) + 3 sample values, (4) + full schema context. Measure how each context level reduces hallucination rate.

**Expected result:** adding sample values (option 3) reduces hallucination by ~40% vs. column name only; full schema context provides only marginal additional improvement

### Test Experiment 3: Hallucination Detection

Train a lightweight hallucination detector predicting whether a (column_name, description) pair is likely hallucinated. Measure precision and recall vs. domain expert labels.

**Expected result:** 80%+ precision at 70%+ recall — sufficient to flag high-risk descriptions for human review

## Expected Results

1. A benchmark of 500+ catalog entries across 5 schema types
2. Hallucination rate characterization by schema type and context level
3. **Key finding:** "IoT and sensor schema metadata has 4x the hallucination rate of financial schemas"
4. Context window recommendation: "Adding 3 sample values reduces hallucination by 40%"
5. A deployable hallucination detection classifier

## Why This Matters / Why People Would Care

- **Data engineers:** deploying AI catalog annotation now; knowing hallucination rate by schema type tells them where human review is needed
- **Data platform vendors** (Databricks, Snowflake, Atlan): their AI catalog features have unknown hallucination rates; this benchmark characterizes the risk
- **ML researchers:** metadata annotation is high-impact but underexplored
- **Data governance:** inaccurate metadata is a compliance risk; quantifying it is the first step to managing it

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | Catalog entry construction (500 entries, domain expert gold labels) |
| 3 | LLM annotation + domain expert hallucination ratings |
| 4 | Context window experiments + hallucination detector training |
| 5 | Analysis + reliability index derivation |
| 6 | Submission to VLDB 2027 or EMNLP 2026 |

## Related Issues

- Design doc GitHub issue: #20
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
