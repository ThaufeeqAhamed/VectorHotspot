#!/usr/bin/env python3
"""
VectorHotspot — Layer 5: Early Warning Lead Time Analysis

Measures how many weeks ahead the model correctly predicts a hotspot
before it becomes apparent in observed case data.

Uses the hotspot taxonomy parquets (2024 annual data) which contain
per-cell, per-week hotspot status across all forecast horizons.

Method:
  For each unique H3 cell, identify weeks where:
    - The model predicts a hotspot at t+k  (z-score from pred{k} column)
    - An actual hotspot is observed in a nearby future week (z-score from observed data)
  
  Since the taxonomy parquet uses pred/z columns (predicted Gi* z-scores) without
  a paired observed ground truth, this script:
    1. Uses the test parquet (which HAS both pred and actual hotspot flags)
    2. For each actual hotspot cell (is_hotspot_act_lead_1 == 1), checks
       if it was predicted k weeks earlier across leads 1–4
    3. Reports the earliest lead time at which the prediction was correct

Input files:
  outputs/hotspots/dengue_hotspots_test_2023_2024.parquet
  outputs/hotspots/malaria_hotspots_test_2023_2024.parquet
  outputs/hotspots/dengue_hotspot_taxonomy_2024.parquet
  outputs/hotspots/malaria_hotspot_taxonomy_2024.parquet

Output files (new):
  outputs/metrics/lead_time_analysis.csv
  outputs/metrics/lead_time_summary.csv

Usage:
  py -3 src/evaluation/lead_time_analysis.py

Notes:
  - Lead time = number of forecast weeks ahead when hotspot was first detected
  - A prediction at t+1 with k=1 means 1 week of advance warning
  - Higher k (e.g., t+4) means 4 weeks of advance warning
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
HOTSPOT_DIR = PROJECT_ROOT / "outputs" / "hotspots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

DISEASES = ["dengue", "malaria"]
HORIZONS = [1, 2, 3, 4]

# Gi* z-score threshold for "significant hotspot"
# p < 0.05 (two-tailed) ≈ |z| > 1.96;  p < 0.01 ≈ |z| > 2.58
HOTSPOT_Z_THRESHOLD = 1.96


def analyze_lead_time_from_test(disease: str) -> tuple[pd.DataFrame, dict]:
    """
    Use test parquet to measure early warning lead time.

    Logic:
      For each (h3_index, year, week) row where is_hotspot_act_lead_1 == 1
      (meaning an actual hotspot is observed 1 week ahead from this timepoint),
      check whether the model also predicted it at each horizon:
        - is_hotspot_pred_lead_1 == 1  → 1 week advance warning
        - is_hotspot_pred_lead_2 == 1  → 2 weeks advance warning (if predicted here)
        - etc.

      The "earliest correct prediction" across horizons gives the lead time.

    Returns:
      df_lead : per-row lead time DataFrame
      summary : dict with mean/median/distribution stats
    """
    parquet_path = HOTSPOT_DIR / f"{disease}_hotspots_test_2023_2024.parquet"
    print(f"\n  Loading {parquet_path.name} ...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"done ({len(df):,} rows, {time.time()-t0:.1f}s)", flush=True)

    # Focus on rows where an actual hotspot is observed at t+1 (ground truth positive)
    actual_hotspot_mask = df["is_hotspot_act_lead_1"] == 1
    df_actual = df[actual_hotspot_mask].copy()
    print(f"  Actual hotspot events (t+1): {len(df_actual):,} rows", flush=True)

    if len(df_actual) == 0:
        print("  WARNING: No actual hotspot events found — skipping lead time analysis.")
        return pd.DataFrame(), {}

    # For each actual hotspot event, find the maximum horizon at which it was predicted
    # (higher horizon = more advance warning)
    lead_time_rows = []
    for k in HORIZONS:
        pred_col = f"is_hotspot_pred_lead_{k}"
        # Rows where actual hotspot AND predicted at this horizon
        correctly_predicted = df_actual[pred_col] == 1
        n_correct = int(correctly_predicted.sum())
        n_total = len(df_actual)
        detection_rate = n_correct / n_total if n_total > 0 else 0.0

        lead_time_rows.append({
            "disease": disease,
            "lead_weeks": k,
            "actual_hotspot_events": n_total,
            "correctly_predicted_at_lead_k": n_correct,
            "detection_rate": round(detection_rate, 4),
            "missed": n_total - n_correct,
        })

        print(
            f"    Lead t+{k}: {n_correct:,}/{n_total:,} actual hotspots detected "
            f"({detection_rate:.1%} detection rate)"
        )

    df_lead = pd.DataFrame(lead_time_rows)

    # Summary: best (maximum) lead time with >50% detection rate
    good_leads = df_lead[df_lead["detection_rate"] >= 0.50]
    max_lead = int(good_leads["lead_weeks"].max()) if len(good_leads) > 0 else 0

    summary = {
        "disease": disease,
        "max_effective_lead_weeks": max_lead,
        "detection_rate_lead1": float(df_lead[df_lead["lead_weeks"] == 1]["detection_rate"].values[0]),
        "detection_rate_lead2": float(df_lead[df_lead["lead_weeks"] == 2]["detection_rate"].values[0]),
        "detection_rate_lead3": float(df_lead[df_lead["lead_weeks"] == 3]["detection_rate"].values[0]),
        "detection_rate_lead4": float(df_lead[df_lead["lead_weeks"] == 4]["detection_rate"].values[0]),
        "total_actual_hotspot_events": int(df_lead["actual_hotspot_events"].iloc[0]),
    }

    return df_lead, summary


def analyze_taxonomy_persistence(disease: str) -> pd.DataFrame:
    """
    Summarize hotspot taxonomy categories from the annual 2024 taxonomy parquet.

    The taxonomy parquet contains 'hotspot_taxonomy' labels such as:
      'Persistent Hotspot', 'Emerging Hotspot', 'Expanding Hotspot',
      'Diminishing Hotspot', 'Non-Hotspot', etc.

    Returns a DataFrame with category counts and percentages.
    """
    tax_path = HOTSPOT_DIR / f"{disease}_hotspot_taxonomy_2024.parquet"
    if not tax_path.exists():
        print(f"  WARNING: Taxonomy file not found: {tax_path.name}")
        return pd.DataFrame()

    df_tax = pd.read_parquet(tax_path)
    taxonomy_counts = (
        df_tax["hotspot_taxonomy"]
        .value_counts()
        .reset_index()
    )
    taxonomy_counts.columns = ["taxonomy_category", "count"]
    taxonomy_counts["percentage"] = (
        taxonomy_counts["count"] / len(df_tax) * 100
    ).round(2)
    taxonomy_counts.insert(0, "disease", disease)

    print(f"\n  {disease.title()} — Hotspot Taxonomy (2024):")
    for _, row in taxonomy_counts.iterrows():
        print(f"    {row['taxonomy_category']:<30} {row['count']:>8,}  ({row['percentage']:.1f}%)")

    return taxonomy_counts


def run_lead_time_analysis():
    """Main entry point."""
    print("=" * 75)
    print(" VECTORHOTSPOT — EARLY WARNING LEAD TIME ANALYSIS")
    print("=" * 75)
    print(f"Reading from : {HOTSPOT_DIR}")
    print(f"Writing to   : {METRICS_DIR}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_lead_dfs = []
    all_summaries = []
    all_taxonomy_dfs = []

    for disease in DISEASES:
        print(f"\n{'─'*75}")
        print(f" {disease.upper()} — Lead Time & Taxonomy Analysis")
        print(f"{'─'*75}")

        # Lead time from test parquet
        df_lead, summary = analyze_lead_time_from_test(disease)
        if len(df_lead) > 0:
            all_lead_dfs.append(df_lead)
            all_summaries.append(summary)

        # Taxonomy from annual parquet
        df_tax = analyze_taxonomy_persistence(disease)
        if len(df_tax) > 0:
            all_taxonomy_dfs.append(df_tax)

    # Save detailed lead time analysis
    if all_lead_dfs:
        df_lead_all = pd.concat(all_lead_dfs, ignore_index=True)
        lead_path = METRICS_DIR / "lead_time_analysis.csv"
        df_lead_all.to_csv(lead_path, index=False)
        print(f"\n  Saved: {lead_path.name}")

    # Save summary
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        summary_path = METRICS_DIR / "lead_time_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path.name}")

        # Print summary table
        print("\n" + "=" * 75)
        print(" EARLY WARNING LEAD TIME SUMMARY")
        print("=" * 75)
        print(
            f"  {'Disease':<10} {'MaxEffLead':>10} "
            f"{'DetRate@t+1':>12} {'DetRate@t+2':>12} "
            f"{'DetRate@t+3':>12} {'DetRate@t+4':>12}"
        )
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for s in all_summaries:
            print(
                f"  {s['disease']:<10} {s['max_effective_lead_weeks']:>10}"
                f" {s['detection_rate_lead1']:>12.1%}"
                f" {s['detection_rate_lead2']:>12.1%}"
                f" {s['detection_rate_lead3']:>12.1%}"
                f" {s['detection_rate_lead4']:>12.1%}"
            )
        print("=" * 75)

    # Save taxonomy
    if all_taxonomy_dfs:
        df_tax_all = pd.concat(all_taxonomy_dfs, ignore_index=True)
        tax_path = METRICS_DIR / "hotspot_taxonomy_distribution.csv"
        df_tax_all.to_csv(tax_path, index=False)
        print(f"  Saved: {tax_path.name}")

    print("\n Lead time analysis complete.\n")


if __name__ == "__main__":
    run_lead_time_analysis()
