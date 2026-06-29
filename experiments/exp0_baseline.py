#!/usr/bin/env python3
"""Experiment 0: Baseline run of the structure-complexity -> recall-gain pipeline.

This script runs the ACTUAL code in src/metadataannotation against its
synthetic data generator (data.make_dataset) and writes REAL computed
statistics (Pearson r, R^2, per-domain means, predicted gains) to
results/exp0_baseline.json.

IMPORTANT CONTEXT FOR READERS:
The numbers this script produces are real outputs of running the code in
this repository -- they are not fabricated. However, the underlying
*data* is synthetic: `metadataannotation.data.make_experiment` draws
`baseline_recall`, `gain`, and `complexity_scores` from independent
`random.uniform(...)` calls. There is no encoded causal or correlational
relationship between document complexity and recall gain in the
generator. Consequently this experiment currently measures the
correlation/regression machinery's correctness on synthetic data, NOT
a real finding about document complexity and metadata-annotation value
(which would require real documents, a real retrieval pipeline, and
real annotation results -- none of which exist yet in this repo).

See the "Known Gaps" section of BLOG.md and PAPER_DRAFT.md for details.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metadataannotation.core import StructurePredictor
from metadataannotation.data import make_dataset
from metadataannotation.evaluate import (
    correlation_complexity_gain,
    domain_summary,
    rank_domains_by_gain,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run(n_per_domain: int, seed: int) -> dict:
    dataset = make_dataset(n_per_domain=n_per_domain, seed=seed)

    predictor = StructurePredictor()
    predictor.fit(dataset)

    complexities = [statistics.mean(e.complexity_scores) for e in dataset]
    gains = [e.gain for e in dataset]
    corr = correlation_complexity_gain(complexities, gains)
    r2 = predictor.r_squared(dataset)

    summary = domain_summary(dataset)
    ranked = rank_domains_by_gain(dataset)

    predicted_gains = {
        f"{c:.1f}": predictor.predict_gain(c) for c in (0.2, 0.4, 0.6, 0.8, 1.0)
    }

    return {
        "n_per_domain": n_per_domain,
        "seed": seed,
        "n_total": len(dataset),
        "pearson_r_complexity_vs_gain": corr,
        "r_squared": r2,
        "domain_summary": summary,
        "domains_ranked_by_mean_gain": ranked,
        "predicted_gain_by_complexity_level": predicted_gains,
        "note": (
            "Data is synthetic (independent random draws); a Pearson r "
            "near 0 here is EXPECTED and indicates the generator does not "
            "encode a complexity-gain relationship, not a code bug."
        ),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Small run, matching scripts/run_demo.py defaults, for direct comparison.
    small = run(n_per_domain=5, seed=42)
    # Larger run to check whether the small-n correlation is noise.
    large = run(n_per_domain=200, seed=42)

    output = {"small_n_run": small, "large_n_run": large}

    out_path = os.path.join(RESULTS_DIR, "exp0_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Small run (n_per_domain=5):  r={small['pearson_r_complexity_vs_gain']:.4f}  R2={small['r_squared']:.4f}")
    print(f"Large run (n_per_domain=200): r={large['pearson_r_complexity_vs_gain']:.4f}  R2={large['r_squared']:.4f}")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
