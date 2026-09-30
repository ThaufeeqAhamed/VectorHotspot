#!/usr/bin/env python3
"""
VectorHotspot — Layer 5: Hotspot Validation

Compares predicted hotspots against actual (observed) hotspots in the 2023–2024
test dataset for both Dengue and Malaria, across all 4 forecast horizons.

Metrics computed per disease × horizon:
  - Precision   : of all predicted hotspot cells, how many were truly hotspots?
  - Recall      : of all true hotspot cells, how many did we correctly predict?
  - F1 Score    : harmonic mean of Precision and Recall
  - Spatial IoU : |predicted ∩ actual| / |predicted ∪ actual| (in cell count terms)
  - Support     : total actual hotspot cells (prevalence context)

Input files (must already exist):
  outputs/hotspots/dengue_hotspots_test_2023_2024.parquet
  outputs/hotspots/malaria_hotspots_test_2023_2024.parquet

  Relevant columns per horizon k:
    is_hotspot_pred_lead_{k}   : 1/0 — model-predicted hotspot
    is_hotspot_act_lead_{k}    : 1/0 — observed actual hotspot (ground truth)

Output files (new):
  outputs/metrics/hotspot_validation_metrics.csv

Usage:
  py -3 src/evaluation/hotspot_validator.py

Notes:
  - Evaluates separately for:
      * All test rows (seen + spatial holdout)
      * Seen districts only   (is_spatial_holdout == False)
      * Spatial holdout only  (is_spatial_holdout == True)
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


def compute_hotspot_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Compute Precision, Recall, F1, and Spatial IoU for binary hotspot labels.

    Parameters
    ----------
    y_true : array-like of int (0/1) — observed hotspot labels
    y_pred : array-like of int (0/1) — predicted hotspot labels

    Returns
    -------
    dict with keys: precision, recall, f1, spatial_iou, support, predicted_positives, tp, fp, fn
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    # tn = int(np.sum(~y_pred & ~y_true))  # not needed but available

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Spatial IoU = |P ∩ O| / |P ∪ O|  (in terms of H3 cell count)
    intersection = tp
    union = tp + fp + fn
    spatial_iou = intersection / union if union > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "spatial_iou": round(spatial_iou, 4),
        "support": int(np.sum(y_true)),                # actual hotspot count
        "predicted_positives": int(np.sum(y_pred)),    # predicted hotspot count
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def validate_disease(disease: str) -> list[dict]:
    """
    Load the test parquet for a disease and compute all validation metrics.

    Returns a list of metric dicts (one per horizon × split combination).
    """
    parquet_path = HOTSPOT_DIR / f"{disease}_hotspots_test_2023_2024.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {parquet_path}")

    print(f"\n  Loading {parquet_path.name} ...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"done ({len(df):,} rows, {time.time()-t0:.1f}s)", flush=True)

    # Define evaluation splits
    splits = {
        "All (Seen + Holdout)": df,
        "Seen Districts": df[~df["is_spatial_holdout"]],
        "Spatial Holdout": df[df["is_spatial_holdout"]],
    }

    records = []
    for lead in HORIZONS:
        pred_col = f"is_hotspot_pred_lead_{lead}"
        act_col = f"is_hotspot_act_lead_{lead}"

        for split_name, split_df in splits.items():
            y_pred = split_df[pred_col].values
            y_true = split_df[act_col].values

            metrics = compute_hotspot_metrics(y_true, y_pred)
            record = {
                "disease": disease,
                "horizon": f"t+{lead}",
                "split": split_name,
                "n_rows": len(split_df),
                **metrics,
            }
            records.append(record)

            print(
                f"    {disease.title()} t+{lead} [{split_name:22s}]  "
                f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  IoU={metrics['spatial_iou']:.3f}  "
                f"Support={metrics['support']:,}"
            )

    return records


def print_summary_table(df_metrics: pd.DataFrame):
    """Print a formatted summary table for Seen Districts split."""
    print("\n" + "=" * 75)
    print(" HOTSPOT VALIDATION SUMMARY — Seen Districts Split")
    print("=" * 75)
    seen = df_metrics[df_metrics["split"] == "Seen Districts"].copy()
    seen = seen.sort_values(["disease", "horizon"])
    print(
        f"  {'Disease':<8} {'Horizon':<6} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>6} {'IoU':>8} {'Support':>10}"
    )
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*10}")
    for _, row in seen.iterrows():
        print(
            f"  {row['disease']:<8} {row['horizon']:<6}"
            f" {row['precision']:>10.3f} {row['recall']:>8.3f}"
            f" {row['f1']:>6.3f} {row['spatial_iou']:>8.3f}"
            f" {row['support']:>10,}"
        )
    print("=" * 75)


def run_hotspot_validation():
    """Main entry point."""
    print("=" * 75)
    print(" VECTORHOTSPOT — HOTSPOT VALIDATION (Precision / Recall / F1 / IoU)")
    print("=" * 75)
    print(f"Reading from : {HOTSPOT_DIR}")
    print(f"Writing to   : {METRICS_DIR}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for disease in DISEASES:
        print(f"\n{'─'*75}")
        print(f" {disease.upper()} — All horizons × All splits")
        print(f"{'─'*75}")
        records = validate_disease(disease)
        all_records.extend(records)

    df_metrics = pd.DataFrame(all_records)

    # Reorder columns for readability
    col_order = [
        "disease", "horizon", "split", "n_rows",
        "precision", "recall", "f1", "spatial_iou",
        "support", "predicted_positives", "tp", "fp", "fn",
    ]
    df_metrics = df_metrics[col_order]

    out_path = METRICS_DIR / "hotspot_validation_metrics.csv"
    df_metrics.to_csv(out_path, index=False)

    print_summary_table(df_metrics)

    print(f"\n Saved: {out_path}")
    print(f" Rows : {len(df_metrics)} ({len(DISEASES)} diseases × {len(HORIZONS)} horizons × 3 splits)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_hotspot_validation()
