#!/usr/bin/env python3
"""
VectorHotspot — Layer 4: Prediction Uncertainty Analysis

Reads pre-computed prediction interval CSVs from outputs/explainability/ and produces:
  1. Interval width statistics (mean, median, p25, p75, p95) per disease/horizon
  2. Coverage metrics: what fraction of true values fall within 90% and 95% intervals
  3. Method documentation: infers and documents the uncertainty method used
  4. Combined summary CSV across all diseases and horizons

Input files (must already exist):
  outputs/explainability/prediction_intervals_{disease}_lead{k}.csv
    - Columns: prediction, actual, lower_90, upper_90, lower_95, upper_95

Output files (new, written to outputs/explainability/):
  outputs/explainability/uncertainty_summary_{disease}_lead{k}.csv
  outputs/explainability/uncertainty_summary_all.csv
  outputs/explainability/uncertainty_method_notes.txt

Usage:
  py -3 src/explainability/uncertainty_analysis.py

Notes:
  - Does NOT retrain models or recompute intervals — only reads existing outputs
  - Does NOT modify any existing file
  - Interval structure (90% + 95%) is consistent with quantile regression output
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPL_DIR = PROJECT_ROOT / "outputs" / "explainability"
OUT_DIR = EXPL_DIR

DISEASES = ["dengue", "malaria"]
HORIZONS = [1, 2, 3, 4]
CHUNK_SIZE = 50_000


def analyze_intervals(disease: str, lead: int) -> dict:
    """
    Read prediction_intervals_{disease}_lead{lead}.csv and compute:
      - Interval widths (upper - lower) for 90% and 95% intervals
      - Empirical coverage: fraction of actual values inside the interval
      - Basic prediction accuracy stats

    Returns a dict of summary statistics.
    """
    pi_path = EXPL_DIR / f"prediction_intervals_{disease}_lead{lead}.csv"
    if not pi_path.exists():
        raise FileNotFoundError(f"Prediction interval file not found: {pi_path}")

    print(f"  Reading {pi_path.name} ...", end=" ", flush=True)
    t0 = time.time()

    # Accumulators for chunked reading
    width_90_list = []
    width_95_list = []
    covered_90 = 0
    covered_95 = 0
    pred_errors = []
    n_rows = 0

    for chunk in pd.read_csv(pi_path, chunksize=CHUNK_SIZE):
        n = len(chunk)
        n_rows += n

        # Interval widths
        w90 = (chunk["upper_90"] - chunk["lower_90"]).values
        w95 = (chunk["upper_95"] - chunk["lower_95"]).values
        width_90_list.append(w90)
        width_95_list.append(w95)

        # Empirical coverage: is actual inside the interval?
        actual = chunk["actual"].values
        covered_90 += np.sum(
            (actual >= chunk["lower_90"].values) & (actual <= chunk["upper_90"].values)
        )
        covered_95 += np.sum(
            (actual >= chunk["lower_95"].values) & (actual <= chunk["upper_95"].values)
        )

        # Prediction error (MAE)
        pred_errors.append(np.abs(chunk["prediction"].values - actual))

    all_w90 = np.concatenate(width_90_list)
    all_w95 = np.concatenate(width_95_list)
    all_errors = np.concatenate(pred_errors)

    summary = {
        "disease": disease,
        "horizon": f"t+{lead}",
        "n_rows": n_rows,
        # Coverage (empirical vs nominal)
        "coverage_90pct": round(covered_90 / n_rows, 4),
        "coverage_95pct": round(covered_95 / n_rows, 4),
        # Interval width — 90%
        "width_90_mean": round(float(np.mean(all_w90)), 6),
        "width_90_median": round(float(np.median(all_w90)), 6),
        "width_90_p25": round(float(np.percentile(all_w90, 25)), 6),
        "width_90_p75": round(float(np.percentile(all_w90, 75)), 6),
        "width_90_p95": round(float(np.percentile(all_w90, 95)), 6),
        # Interval width — 95%
        "width_95_mean": round(float(np.mean(all_w95)), 6),
        "width_95_median": round(float(np.median(all_w95)), 6),
        # Prediction accuracy
        "mae": round(float(np.mean(all_errors)), 6),
    }

    elapsed = time.time() - t0
    print(f"done ({n_rows:,} rows, {elapsed:.1f}s)", flush=True)
    return summary


def infer_and_document_method(sample_row: dict) -> str:
    """
    Infer the uncertainty method from the interval structure.

    The presence of both 'lower_90'/'upper_90' and 'lower_95'/'upper_95' columns
    (two separate confidence levels) is characteristic of quantile regression
    (where each quantile is a separate model output) or conformal prediction.

    The structure is NOT consistent with bootstrap (which would typically produce
    a single percentile-based interval) or simple ±σ Gaussian intervals.

    Since both 90% and 95% levels are pre-computed and stored separately,
    and given that LightGBM/XGBoost natively support quantile regression objectives,
    the most likely method is:

        QUANTILE REGRESSION
        - Lower bound: q=0.05 (for 90% interval), q=0.025 (for 95%)
        - Upper bound: q=0.95 (for 90% interval), q=0.975 (for 95%)
        - Trained as separate LightGBM models with objective='quantile'
    """
    return (
        "INFERRED UNCERTAINTY METHOD: Quantile Regression\n"
        "\n"
        "Evidence:\n"
        "  - Two distinct confidence levels stored (90% and 95%)\n"
        "  - Each has separate lower and upper bounds (lower_90/upper_90, lower_95/upper_95)\n"
        "  - This structure is consistent with quantile regression (two separate quantile pairs)\n"
        "  - Lower bounds are frequently 0.0 (consistent with quantile pinball loss on\n"
        "    zero-heavy count data where q=0.05 collapses to 0)\n"
        "  - LightGBM and XGBoost both support objective='quantile' natively\n"
        "\n"
        "Likely training setup:\n"
        "  - 90% interval: LightGBM trained at alpha=0.05 (lower) and alpha=0.95 (upper)\n"
        "  - 95% interval: LightGBM trained at alpha=0.025 (lower) and alpha=0.975 (upper)\n"
        "\n"
        "Limitation:\n"
        "  - The exact training script for prediction intervals was not found in src/.\n"
        "  - This inference is based on the output file structure, not source code inspection.\n"
        "  - Empirical coverage metrics computed in uncertainty_summary_all.csv confirm\n"
        "    whether the intervals achieve their nominal coverage.\n"
    )


def print_summary(s: dict):
    """Print a formatted summary row."""
    print(f"\n  {s['disease'].title()} {s['horizon']}:")
    print(f"    Rows          : {s['n_rows']:,}")
    print(f"    Coverage 90%  : {s['coverage_90pct']:.1%}  (nominal: 90%)")
    print(f"    Coverage 95%  : {s['coverage_95pct']:.1%}  (nominal: 95%)")
    print(f"    Width 90% mean: {s['width_90_mean']:.6f}  median: {s['width_90_median']:.6f}")
    print(f"    Width 95% mean: {s['width_95_mean']:.6f}  median: {s['width_95_median']:.6f}")
    print(f"    MAE (point)   : {s['mae']:.6f}")


def run_uncertainty_analysis():
    """Main entry point."""
    print("=" * 65)
    print(" VECTORHOTSPOT — PREDICTION UNCERTAINTY ANALYSIS")
    print("=" * 65)
    print(f"Reading from : {EXPL_DIR}")
    print(f"Writing to   : {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for disease in DISEASES:
        print(f"\n{'─'*65}")
        print(f" Disease: {disease.upper()}")
        print(f"{'─'*65}")

        for lead in HORIZONS:
            summary = analyze_intervals(disease, lead)
            print_summary(summary)
            all_summaries.append(summary)

            # Save per-disease/horizon CSV
            df_s = pd.DataFrame([summary])
            out_path = OUT_DIR / f"uncertainty_summary_{disease}_lead{lead}.csv"
            df_s.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

    # Combined summary
    df_all = pd.DataFrame(all_summaries)
    combined_path = OUT_DIR / "uncertainty_summary_all.csv"
    df_all.to_csv(combined_path, index=False)

    # Method documentation
    method_notes = infer_and_document_method({})
    notes_path = OUT_DIR / "uncertainty_method_notes.txt"
    notes_path.write_text(method_notes, encoding="utf-8")

    print(f"\n{'='*65}")
    print(f" Combined uncertainty summary : {combined_path.name}")
    print(f" Method documentation         : {notes_path.name}")
    print()
    print(" COVERAGE SUMMARY (all diseases × horizons):")
    print(f" {'Disease':<8} {'Horizon':<6} {'Cov90%':>8} {'Cov95%':>8} {'Width90 Median':>16}")
    print(f" {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*16}")
    for s in all_summaries:
        print(
            f" {s['disease']:<8} {s['horizon']:<6}"
            f" {s['coverage_90pct']:>8.1%}"
            f" {s['coverage_95pct']:>8.1%}"
            f" {s['width_90_median']:>16.6f}"
        )
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_uncertainty_analysis()
