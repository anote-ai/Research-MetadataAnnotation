# Research Design Document: Metadata Annotation

## Vision Statement

Establish that LLM-generated metadata annotation is a **viable, scalable alternative** to manual schema enrichment for enterprise data catalogs — but only when hallucination risk is understood and managed — and deliver **MetaAnnotate-Bench**: the first benchmark for measuring metadata annotation quality, hallucination rates, and schema adherence across database and document types.

---

## Problem Statement & Novelty

Enterprise data governance requires annotating thousands of tables, columns, and documents with metadata: descriptions, data types, PII labels, business glossary terms, lineage tags, and quality indicators. Manual annotation is slow and expensive; LLM-based automation is increasingly used but has an undercharacterized failure mode — **metadata hallucination**: generating plausible-sounding but factually incorrect metadata that propagates silently through data catalogs.

Key gaps in existing work:
1. No systematic study of **hallucination rates by metadata type** (description vs. PII label vs. business term are very different risks).
2. No characterization of **context window effects** on annotation quality (short column name vs. sample data vs. full schema).
3. No benchmark for comparing metadata annotation systems.
4. No **hallucination detector** tailored to metadata annotation.

### Novel Contributions

| Contribution | Description |
|---|---|
| **MetaAnnotate-Bench** | 5,000 annotation tasks across 6 metadata types, 4 schema types, with gold labels |
| **MHR metric** | Metadata Hallucination Rate: fraction of annotations with factually incorrect claims |
| **Schema adherence score** | Fraction of annotations conforming to target schema (format, vocabulary, constraints) |
| **Context window ablation** | First systematic study of how context richness affects metadata annotation quality |
| **MetaHal detector** | Fine-tuned hallucination detector specific to metadata annotation domain |

### MHR Definition

```
MHR = hallucinated_annotations / total_annotations

Hallucination criteria (metadata-specific):
  - Description: contains factually false claims about column content
  - PII label: incorrect PII category (e.g., calls SSN "public identifier")
  - Data type: incorrect inference from sample values
  - Business term: maps to wrong glossary entry
  - Lineage tag: invents non-existent upstream tables
```

---

## Research Objectives

1. Quantify **MHR by metadata type**: which annotation types are safe to automate vs. require human review?
2. Characterize the **context window effect**: how much does annotation quality improve when sample data is provided vs. column name only?
3. Build a **MetaHal detector** that identifies hallucinated metadata with >85% precision.
4. Show that LLM annotation with MetaHal filtering achieves >90% of human annotation quality at <20% of the cost.
5. Identify **schema types** where LLM annotation reliably fails (e.g., custom business ontologies).

---

## Dataset Construction

### Metadata Type Coverage

| Metadata Type | Count | Description | Hallucination Risk |
|---|---|---|---|
| Column description | 1,000 | Natural language description of column | Medium |
| PII label | 1,000 | PII category (SSN, email, name, etc.) | High |
| Data type inference | 500 | Logical data type from samples | Low |
| Business glossary term | 1,000 | Maps column to enterprise glossary | High |
| Data quality indicator | 750 | Completeness, accuracy, freshness flags | Medium |
| Lineage tag | 750 | Upstream/downstream table references | Very High |

### Schema Type Coverage (4 types)
- **Relational database**: SQL schemas (INFORMATION_SCHEMA format)
- **Data warehouse**: dimensional models (star schema, fact/dimension)
- **Document store**: JSON/BSON schema definitions
- **Unstructured documents**: PDF/Word business documents

### Context Richness Levels

```
Level 0: Column name only ("customer_id")
Level 1: Column name + data type ("customer_id: INTEGER")
Level 2: Level 1 + 5 sample values ([1001, 1002, 1003, NULL, 1005])
Level 3: Level 2 + table schema (all columns + types)
Level 4: Level 3 + table description + business context
```

---

## Systems Under Evaluation

| System | Model | Notes |
|---|---|---|
| GPT-4o | OpenAI | Frontier baseline |
| Claude Sonnet 4 | Anthropic | Our primary |
| Gemini 1.5 Pro | Google | Long-context |
| Llama 3.1 70B | Meta | Open-source |
| GPT-3.5-turbo | OpenAI | Cost baseline |
| Rule-based system | Custom | Legacy baseline |
| Human annotators | — | Gold standard |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Run GPT-4o on column description annotation (Level 2 context) for 200 relational database columns. Compute accuracy and MHR.

**Expected result**: Accuracy ≈ 0.84, MHR ≈ 0.12. Establishes that column description is one of the safer annotation types — but still has non-trivial hallucination rate.

---

### Experiment 1: MHR by Metadata Type
**Hypothesis**: Lineage tags have MHR > 0.45 for all models; PII labels have MHR > 0.25; column descriptions have MHR < 0.15.

**Protocol**:
1. Run all LLM systems on all 5,000 annotation tasks (Level 3 context).
2. Two human raters score each annotation for hallucination (κ > 0.80 required).
3. Compute MHR per (model × metadata type × schema type) cell.

**Expected results**:

| Metadata Type | GPT-4o MHR | Claude MHR | Human MHR |
|---|---|---|---|
| Column description | 0.11 | 0.09 | 0.03 |
| PII label | 0.24 | 0.21 | 0.04 |
| Data type inference | 0.06 | 0.05 | 0.02 |
| Business glossary | 0.31 | 0.28 | 0.06 |
| Data quality indicator | 0.18 | 0.16 | 0.05 |
| Lineage tag | 0.51 | 0.48 | 0.08 |

- Key finding: lineage tags should never be auto-annotated without human review; column descriptions and data types are safe above MHR threshold of 0.15.

---

### Experiment 2: Context Window Ablation
**Hypothesis**: MHR decreases monotonically from Level 0 to Level 4 context; the largest single improvement comes from adding sample values (Level 1 → Level 2).

**Protocol**:
1. Run GPT-4o and Claude on all annotation types across all 5 context levels.
2. Compute MHR and annotation quality score per level.
3. Measure marginal improvement from each context level addition.

**Expected results**:

| Context Level | GPT-4o MHR (avg) | Quality Score |
|---|---|---|
| Level 0 (name only) | 0.42 | 0.51 |
| Level 1 (name + type) | 0.33 | 0.61 |
| Level 2 (+ samples) | 0.21 | 0.74 |
| Level 3 (+ schema) | 0.15 | 0.81 |
| Level 4 (+ context) | 0.11 | 0.86 |

- Largest single jump: Level 1 → Level 2 (adding sample values reduces MHR by 12 pp)
- Practical implication: always include sample values in metadata annotation prompts

---

### Experiment 3: Schema Adherence
**Hypothesis**: LLMs exhibit >25% schema violation rate on custom enterprise business glossaries, because they cannot accurately map to vocabularies not in training data.

**Protocol**:
1. Provide each system with enterprise glossary (200 business terms) as context.
2. Run business glossary annotation on 1,000 columns.
3. Measure: (a) correct term assignment, (b) term hallucination (invents terms not in glossary), (c) closest-match errors.

**Expected results**:
- Correct assignment rate: GPT-4o ≈ 0.58, Claude ≈ 0.61
- Term hallucination (invents non-existent terms): ≈ 28% of errors
- Closest-match errors (wrong but adjacent term): ≈ 45% of errors
- Key finding: retrieval-augmented prompting (inject relevant glossary terms) reduces term hallucination from 28% to 9%

---

### Experiment 4: MetaHal Detector
**Hypothesis**: A fine-tuned hallucination detector achieves >85% precision and >80% recall on metadata hallucination detection, enabling cost-effective human review targeting.

**Protocol**:
1. Use Experiment 1 annotations as training data (n=3,000; 500 hallucinated per metadata type).
2. Fine-tune DeBERTa-v3 as binary hallucination classifier.
3. Evaluate on held-out 1,000 examples.
4. Measure: at what review rate does MetaHal filtering achieve 95% annotation quality?

**Expected results**:
- MetaHal precision: 0.87, recall: 0.83, F1: 0.85
- At 15% human review rate (MetaHal-flagged annotations): 94% of human annotation quality
- At 5% review rate: 87% of human annotation quality
- Cost model: 15% review rate → 85% cost savings vs. fully human annotation at near-human quality

```python
# MetaHal-filtered annotation pipeline
def annotate_with_quality_control(schema_items, llm_annotator, metahal_detector, review_budget=0.15):
    annotations = llm_annotator.annotate_batch(schema_items)
    hallucination_scores = metahal_detector.score_batch(annotations)
    review_threshold = np.percentile(hallucination_scores, (1 - review_budget) * 100)
    for ann, score in zip(annotations, hallucination_scores):
        ann.needs_review = score > review_threshold
    return annotations
```

---

### Experiment 5: Custom Ontology vs. Standard Schema
**Hypothesis**: LLM annotation quality on standard schemas (ISO data types, GDPR PII categories) is ≥15 pp higher than on custom enterprise ontologies, and this gap cannot be closed by prompting alone.

**Protocol**:
1. Compare annotation quality: (a) standard schemas (ISO, GDPR), (b) custom enterprise ontologies (synthetic + real-world).
2. Test: zero-shot, few-shot (5 examples), RAG-augmented (inject relevant ontology definitions).
3. Compute quality gap between standard and custom for each prompting method.

**Expected results**:
- Standard schema annotation accuracy: 0.83 (GPT-4o)
- Custom ontology accuracy (zero-shot): 0.61 (gap: 22 pp)
- Custom ontology accuracy (RAG-augmented): 0.74 (gap closes to 9 pp)
- Key finding: RAG-augmented prompting with ontology injection is necessary for custom schemas; zero-shot is insufficient

---

## Expected Results Summary

| Finding | Result |
|---|---|
| Lineage tags MHR | >0.45 for all models — human review required |
| Sample values impact | Adding samples reduces MHR by ~12 pp (largest single context lever) |
| MetaHal detector | 85% F1; enables 85% cost savings at 94% quality |
| Custom ontology gap | 22 pp quality gap vs. standard; RAG closes to 9 pp |
| Business glossary term hallucination | 28% of errors; RAG-augmented reduces to 9% |

**Primary claim**: MHR varies from 0.06 (data type inference) to 0.51 (lineage tags) — a 8× range that determines whether auto-annotation is safe. MetaAnnotate-Bench provides the first principled framework for making this determination.

---

## Why This Matters

**For researchers**: MetaAnnotate-Bench fills the gap in understanding LLM-generated metadata quality — critical as data catalog automation accelerates.

**For practitioners**: MHR thresholds and the MetaHal detector enable data governance teams to automate annotation safely, with principled human review budgeting.

**For Anote products**: Directly applicable to Anote's data annotation platform — MetaHal can be integrated as a quality gate.

**Market**: Enterprise data catalog market is $1.6B; Collibra, Alation, and Microsoft Purview all need reliable automated annotation.

---

## Implementation Plan

```
research-metadataannotation/
├── data/
│   ├── schemas/         # 5,000 annotation tasks
│   ├── gold_labels/     # Human-annotated gold
│   └── glossaries/      # Enterprise ontologies
├── annotation/
│   ├── prompt_templates/ # Per-metadata-type prompts
│   └── rag_augmented.py  # Ontology-injected prompting
├── metrics/
│   ├── mhr.py
│   └── schema_adherence.py
├── metahal/
│   ├── train_detector.py
│   └── metahal_model/
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_mhr_by_type.py
│   ├── exp2_context_ablation.py
│   ├── exp3_schema_adherence.py
│   ├── exp4_metahal.py
│   └── exp5_custom_ontology.py
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Dataset construction | 6 weeks | 5,000 labeled annotation tasks |
| LLM annotation collection | 2 weeks | All model annotations |
| Human hallucination labeling | 4 weeks | MHR ground truth |
| MetaHal training | 3 weeks | Fine-tuned detector |
| Experiments | 4 weeks | All results |
| Paper writing | 4 weeks | VLDB 2027 or EMNLP 2026 |

**Target venues**: VLDB 2027, EMNLP 2026, or CIDR 2027

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Custom enterprise ontology access | High | Create synthetic ontologies + validate with domain experts |
| Human annotation cost for hallucination | High | Phased: label most ambiguous cases only |
| MetaHal generalization to new domains | Medium | Validate on held-out schema types |
| LLM log-probability availability | Medium | Use open-source models with logprob access |

---

## Related Issues

- Product integration: Anote data annotation platform
- Reproducibility package
- Related work audit: DataBench, SchemaAnnot, ColumnNet
- Ethics: PII labeling errors and regulatory risk
