# Can giving an AI extra "metadata" about a document make it better at finding answers?

*A plain-language look at the metadata-annotation research project.*

## The question

When you ask an AI system to find information in a pile of documents
(financial filings, legal contracts, medical records, technical specs),
it usually just reads the raw text. This project asks a simple question:
**does telling the AI a bit about a document's structure first — how many
sections it has, whether it contains tables or figures — help it retrieve
the right answer more often?**

The intuition is that a 200-page legal contract with 20 sections and
embedded tables is a fundamentally different retrieval problem than a
2-page memo. If we can measure that difference ("structural complexity")
and show it predicts how much "metadata-augmented" retrieval helps, we'd
have a practical rule of thumb: *invest in metadata enrichment for your
most structurally complex documents first.*

## What's been built so far

We built a small, well-tested Python toolkit (`metadataannotation`) with:

- A **complexity score** for a document, combining section count, average
  section length, and whether it has tables/figures.
- Data models for three realistic document types: SEC financial filings,
  legal contracts, and medical records (with synthetic but
  schema-realistic fields like CIK numbers, ICD-10 codes, governing law).
- Standard **inter-annotator agreement** statistics (Cohen's kappa,
  Fleiss' kappa) for measuring how much human or LLM annotators agree.
- A simple **linear predictor** that fits "recall gain from metadata" as
  a function of "document complexity," so you can ask: *if I plug in a
  document this complex, how much retrieval gain should I expect?*

All of this is implemented, tested (`pytest` suite passing), and runnable
end-to-end via `python scripts/run_demo.py` or `python experiments/exp0_baseline.py`.

## What we found when we actually ran it

This is the important, honest part. The current dataset used to test the
complexity-predicts-gain hypothesis is **synthetic**: baseline recall,
metadata recall, and complexity scores are each drawn from independent
random distributions rather than from real documents run through a real
retrieval pipeline.

When we ran the existing prediction code against this synthetic data:

- At small scale (5 examples/domain, matching the original demo): Pearson
  r between complexity and recall gain = **-0.17**, R² = **0.03**.
- At larger scale (200 examples/domain, to rule out small-sample noise):
  Pearson r = **0.005**, R² ≈ **0** — essentially zero.

In other words: **with the current synthetic data generator, there is no
measurable relationship between document complexity and metadata gain**,
because the generator doesn't encode one. The README's experiment table
(showing complexity rising in lockstep with recall gain across Finance,
Legal, Medical, Technical) does not match what the shipped code actually
produces when run — it reads as an *illustrative target table*, not a
verified result. This is a gap worth closing before any external claims
are made about this hypothesis.

## Why this still matters

None of this means the underlying idea is bad — it means the empirical
evidence isn't there yet. The pipeline (complexity scoring, predictor
fitting, evaluation metrics, agreement statistics) is sound and reusable.
What's missing is real input: actual documents, an actual retrieval
system run with and without metadata, and recall numbers computed from
real queries against real ground truth — not `random.uniform()` draws.

## What's next

1. Replace the synthetic data generator with real documents (even a small
   pilot set of ~50 real filings/contracts/records) and a real retrieval
   baseline.
2. Re-run the complexity-vs-gain analysis on that real data and report
   whatever correlation actually emerges — positive, negative, or null.
3. If a real signal exists, scale up the document set and add the
   hallucination-rate and schema-adherence studies described in the
   project's design document, which are not yet implemented in code.

This is an early-stage, infrastructure-complete project: the measurement
tools work, but the experiment hasn't yet been run on real data.
