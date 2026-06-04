#!/usr/bin/env python3
"""Demo script for metadataannotation package."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metadataannotation.data import make_dataset
from metadataannotation.core import StructurePredictor
from metadataannotation.evaluate import (
    correlation_complexity_gain,
    domain_summary,
    rank_domains_by_gain,
)


def main() -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    dataset = make_dataset(n_per_domain=5, seed=42)

    # Fit predictor
    predictor = StructurePredictor()
    predictor.fit(dataset)

    # Compute correlation
    import statistics
    complexities = [statistics.mean(e.complexity_scores) for e in dataset]
    gains = [e.gain for e in dataset]
    corr = correlation_complexity_gain(complexities, gains)
    r2 = predictor.r_squared(dataset)

    print(f"\nPearson r (complexity vs gain): {corr:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Domain summary
    summary = domain_summary(dataset)
    ranked = rank_domains_by_gain(dataset)

    print("\nDomain Summary:")
    if console:
        table = Table(title="Domain Summary")
        table.add_column("Domain")
        table.add_column("Mean Gain", justify="right")
        table.add_column("Mean Complexity", justify="right")
        table.add_column("N", justify="right")
        for domain, _ in ranked:
            info = summary[domain]
            table.add_row(
                domain,
                f"{info['mean_gain']:.4f}",
                f"{info['mean_complexity']:.4f}",
                str(info['n']),
            )
        console.print(table)
    else:
        for domain, mg in ranked:
            info = summary[domain]
            print(f"  {domain}: gain={info['mean_gain']:.4f}, complexity={info['mean_complexity']:.4f}, n={info['n']}")

    # Predictions at various complexity levels
    print("\nPredicted gain at complexity levels:")
    for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pg = predictor.predict_gain(c)
        print(f"  complexity={c:.1f} -> predicted gain={pg:.4f}")


if __name__ == "__main__":
    main()
