#!/usr/bin/env python3
"""
VectorHotspot — Layer 4: Global SHAP Feature Importance Analysis

Reads pre-computed SHAP value CSVs from outputs/explainability/ and produces:
  1. Global feature importance table: mean |SHAP| per feature, per disease, per horizon
  2. Top-N feature ranking report (printed to console and saved as CSV)

Input files (must already exist):
  outputs/explainability/shap_values_{disease}_lead{k}.csv
    - Columns: one column per model feature, each row = per-sample SHAP values

Output files (new, written to outputs/explainability/):
  outputs/explainability/global_shap_importance_{disease}_lead{k}.csv
    - Columns: feature, mean_abs_shap, rank

Usage:
  py -3 src/explainability/shap_analysis.py

Notes:
  - Reads SHAP CSVs in chunks to handle ~50MB files efficiently
  - Does NOT retrain models or recompute SHAP — only reads existing outputs
  - Does NOT modify any existing file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHAP_DIR = PROJECT_ROOT / "outputs" / "explainability"
OUT_DIR = SHAP_DIR  # write summaries alongside existing SHAP files

DISEASES = ["dengue", "malaria"]
HORIZONS = [1, 2, 3, 4]
TOP_N = 15  # number of top features to highlight in the report


def compute_global_shap_importance(disease: str, lead: int) -> pd.DataFrame:
    """
    Read shap_values_{disease}_lead{lead}.csv and compute mean |SHAP| per feature.

    Returns a DataFrame sorted by mean_abs_shap descending with columns:
      feature, mean_abs_shap, rank
    """
    shap_path = SHAP_DIR / f"shap_values_{disease}_lead{lead}.csv"
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP file not found: {shap_path}")

    print(f"  Reading {shap_path.name}  ...", end=" ", flush=True)
    t0 = time.time()

    # Read in chunks to keep memory manageable (~50MB file, 45 feature columns)
    chunk_size = 50_000
    accum = None  # will hold running sum of |SHAP| and count

    for chunk in pd.read_csv(shap_path, chunksize=chunk_size):
        abs_chunk = chunk.abs()
        if accum is None:
            accum = abs_chunk.sum()
            n_rows = len(chunk)
        else:
            accum = accum + abs_chunk.sum()
            n_rows += len(chunk)

    mean_abs = accum / n_rows
    df_imp = (
        mean_abs
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["rank"] = df_imp.index + 1

    print(f"done ({n_rows:,} rows, {time.time() - t0:.1f}s)", flush=True)
    return df_imp


def print_top_features(disease: str, lead: int, df_imp: pd.DataFrame):
    """Print a readable top-N feature table."""
    top = df_imp.head(TOP_N)
    print(f"\n  Top {TOP_N} features — {disease.title()} t+{lead}:")
    print(f"  {'Rank':>4}  {'Feature':<35}  {'Mean |SHAP|':>12}")
    print(f"  {'----':>4}  {'-'*35}  {'-'*12}")
    for _, row in top.iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<35}  {row['mean_abs_shap']:>12.6f}")


def run_shap_analysis():
    """Main entry point — processes all diseases × horizons."""
    print("=" * 65)
    print(" VECTORHOTSPOT — GLOBAL SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 65)
    print(f"Reading from : {SHAP_DIR}")
    print(f"Writing to   : {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for disease in DISEASES:
        print(f"\n{'─'*65}")
        print(f" Disease: {disease.upper()}")
        print(f"{'─'*65}")

        for lead in HORIZONS:
            df_imp = compute_global_shap_importance(disease, lead)
            print_top_features(disease, lead, df_imp)

            # Add disease/horizon metadata columns
            df_imp.insert(0, "horizon", f"t+{lead}")
            df_imp.insert(0, "disease", disease)

            # Save per-disease/horizon importance CSV
            out_path = OUT_DIR / f"global_shap_importance_{disease}_lead{lead}.csv"
            df_imp.to_csv(out_path, index=False)
            print(f"\n  Saved: {out_path.name}")

            all_summaries.append(df_imp)

    # Combined summary across all diseases × horizons (useful for cross-comparison)
    df_combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = OUT_DIR / "global_shap_importance_all.csv"
    df_combined.to_csv(combined_path, index=False)
    print(f"\n{'='*65}")
    print(f" Combined importance table saved: {combined_path.name}")
    print(f" Total feature-importance rows  : {len(df_combined):,}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_shap_analysis()
