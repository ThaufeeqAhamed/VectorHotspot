"""
Phase 13 - Task 1: Alert Generation Engine.

Three-tier alert system based on predicted hotspot intensity and case growth:
  - Level 1 (Yellow/Watch): Predicted case growth >20%, Gi* z > 1.65
  - Level 2 (Orange/Warning): Emerging hotspot (Gi* ≥ 1.96), lead ≥ 2 weeks
  - Level 3 (Red/High Alert): Intensifying/Persistent hotspot (Gi* ≥ 2.58)

Outputs:
  - alerts_dengue_latest.json (alert status per hex for latest week)
  - alerts_malaria_latest.json
  - alert_summary.json (counts per level per state)
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HOT_DIR = ROOT / "outputs" / "hotspots"
OUT_DIR = ROOT / "outputs" / "alerts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_case_growth(row):
    """Compute % case growth from current to lead-1."""
    current = row["cases_lag_1"]
    predicted = row["pred_lead_1"]
    if current < 1e-6:
        return 0.0
    return ((predicted - current) / current) * 100


def assign_alert_level(row):
    """
    Assign alert level based on rules:
      Level 3 (Red): Gi* pred ≥ 2.58 (p < 0.01) AND (Persistent OR Intensifying)
      Level 2 (Orange): Gi* pred ≥ 1.96 (p < 0.05) AND Emerging
      Level 1 (Yellow): Case growth > 20% OR Gi* pred > 1.65
      Level 0 (None): No alert
    """
    z = row["gi_zscore_pred_lead_1"]
    growth = row["case_growth_pct"]
    taxonomy = row.get("hotspot_taxonomy", "")

    # Level 3: High alert (Persistent/Intensifying with high significance)
    if z >= 2.58 and taxonomy in ["Persistent", "Intensifying"]:
        return 3

    # Level 2: Warning (Emerging hotspot with p < 0.05)
    if z >= 1.96 and taxonomy == "Emerging":
        return 2

    # Level 1: Watch (moderate growth or weak significance)
    if growth > 20 or z > 1.65:
        return 1

    return 0


def generate_alerts(disease):
    """Generate alerts for latest week in test set."""
    print(f"\n[{disease.title()}] Loading hotspot predictions...")
    df = pd.read_parquet(HOT_DIR / f"{disease}_hotspots_test_2023_2024.parquet")

    # Filter to latest week
    latest_year = df["year"].max()
    latest_week = df[df["year"] == latest_year]["week"].max()
    latest = df[(df["year"] == latest_year) & (df["week"] == latest_week)].copy()

    print(f"  Latest week: Year {latest_year}, Week {latest_week}")
    print(f"  Hexagons: {len(latest):,}")

    # Compute case growth
    latest["case_growth_pct"] = latest.apply(compute_case_growth, axis=1)

    # Assign alert levels
    latest["alert_level"] = latest.apply(assign_alert_level, axis=1)

    # Alert summary
    alert_counts = latest["alert_level"].value_counts().sort_index()
    print(f"\n  Alert distribution:")
    for level, count in alert_counts.items():
        label = ["None", "Yellow (Watch)", "Orange (Warning)", "Red (High Alert)"][level]
        print(f"    Level {level} ({label}): {count:,}")

    # Export alerts
    alerts_out = latest[["h3_index", "state", "district", "year", "week",
                         "cases_lag_1", "pred_lead_1", "case_growth_pct",
                         "gi_zscore_pred_lead_1", "is_hotspot_pred_lead_1",
                         "hotspot_taxonomy", "alert_level"]].copy()

    alerts_out.to_json(OUT_DIR / f"alerts_{disease}_latest.json", orient="records", indent=2)

    # Per-state summary
    state_summary = latest.groupby("state")["alert_level"].value_counts().unstack(fill_value=0)
    state_summary.columns = [f"level_{int(c)}" for c in state_summary.columns]
    state_summary = state_summary.reset_index()
    state_summary.to_csv(OUT_DIR / f"alerts_{disease}_by_state.csv", index=False)

    return {
        "disease": disease,
        "latest_year": int(latest_year),
        "latest_week": int(latest_week),
        "total_hexagons": int(len(latest)),
        "level_0": int(alert_counts.get(0, 0)),
        "level_1_yellow": int(alert_counts.get(1, 0)),
        "level_2_orange": int(alert_counts.get(2, 0)),
        "level_3_red": int(alert_counts.get(3, 0)),
    }


def main():
    print("=== Phase 13 - Task 1: Alert Generation Engine ===\n")
    print("Generating 3-tier alerts for latest week (2023-2024 test set)...\n")

    summary = {}
    for disease in ["dengue", "malaria"]:
        summary[disease] = generate_alerts(disease)

    with open(OUT_DIR / "alert_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Alerts generated -> {OUT_DIR}")
    print("=== Task 1 Complete ===")


if __name__ == "__main__":
    main()
