"""
Phase 11 - Task 2: National Spatial Holdout Validation.

108 districts across 28 states were withheld from training entirely
(is_spatial_holdout=True). This script evaluates whether the model
generalises to geography it has never seen.

Method:
  - Aggregate hex-level predictions to district-week level.
  - Per week: Spearman rank correlation between predicted Gi* z-score
    and actual cases across all holdout districts.
  - Per state: mean Spearman rho across weeks.
  - Hotspot detection rate: for district-weeks where actual cases > median,
    what fraction have at least one predicted hotspot hex?

Outputs:
  outputs/evaluation/spatial_holdout_validation.csv   (district-week level)
  outputs/evaluation/spatial_holdout_by_state.csv     (per-state summary)
  outputs/evaluation/spatial_holdout_summary.json     (top-level metrics)
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT    = Path(__file__).resolve().parent.parent.parent
HOT_DIR = ROOT / "outputs" / "hotspots"
OUT_DIR = ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_holdout(disease):
    path = HOT_DIR / f"{disease}_hotspots_test_2023_2024.parquet"
    df = pd.read_parquet(path)
    return df[df["is_spatial_holdout"] == True].copy()


def aggregate_to_district(df):
    """Aggregate hex-level rows to (state, district, year, week)."""
    return df.groupby(["state", "district", "year", "week"]).agg(
        pred_risk_lead1   = ("gi_zscore_pred_lead_1", "mean"),
        pred_risk_lead2   = ("gi_zscore_pred_lead_2", "mean"),
        pred_risk_lead3   = ("gi_zscore_pred_lead_3", "mean"),
        pred_risk_lead4   = ("gi_zscore_pred_lead_4", "mean"),
        actual_cases      = ("target_lead_1", "sum"),
        n_hotspot_pred    = ("is_hotspot_pred_lead_1", "sum"),
        n_hotspot_act     = ("is_hotspot_act_lead_1",  "sum"),
        n_hexes           = ("h3_index", "count"),
    ).reset_index()


def compute_spearman_per_week(dist_df, risk_col="pred_risk_lead1"):
    """Spearman rho across districts for each (year, week)."""
    rows = []
    for (year, week), grp in dist_df.groupby(["year", "week"]):
        if len(grp) < 5:
            continue
        rho, pval = spearmanr(grp[risk_col], grp["actual_cases"])
        rows.append({
            "year": year, "week": week,
            "n_districts": len(grp),
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pval), 6),
            "significant": bool(pval < 0.05),
        })
    return pd.DataFrame(rows)


def compute_by_state(dist_df, risk_col="pred_risk_lead1"):
    """Per-state Spearman rho (pool all weeks within the state)."""
    rows = []
    for state, grp in dist_df.groupby("state"):
        if len(grp) < 10:
            continue
        rho, pval = spearmanr(grp[risk_col], grp["actual_cases"])
        rows.append({
            "state": state,
            "n_district_weeks": len(grp),
            "n_districts": grp["district"].nunique(),
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pval), 6),
            "significant": bool(pval < 0.05),
        })
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


def hotspot_detection_rate(dist_df):
    """Fraction of high-actual-case district-weeks flagged as predicted hotspot."""
    median_cases = dist_df["actual_cases"].median()
    high = dist_df[dist_df["actual_cases"] > median_cases]
    if len(high) == 0:
        return float("nan")
    return float((high["n_hotspot_pred"] > 0).mean())


def run_disease(disease):
    print(f"\n[{disease.title()}] Loading holdout predictions...")
    df = load_holdout(disease)
    print(f"  Holdout rows: {len(df):,}  |  districts: {df['district'].nunique()}  |  states: {df['state'].nunique()}")

    dist_df = aggregate_to_district(df)
    print(f"  District-week rows: {len(dist_df):,}")

    spearman_weekly = compute_spearman_per_week(dist_df)
    by_state        = compute_by_state(dist_df)
    det_rate        = hotspot_detection_rate(dist_df)

    mean_rho = spearman_weekly["spearman_rho"].mean()
    pct_sig  = spearman_weekly["significant"].mean() * 100

    print(f"  Mean Spearman rho (across weeks) : {mean_rho:.4f}")
    print(f"  % weeks significant (p<0.05)     : {pct_sig:.1f}%")
    print(f"  Hotspot detection rate           : {det_rate:.4f}")
    print(f"\n  Top 5 states by Spearman rho:")
    print(by_state.head(5)[["state", "spearman_rho", "n_districts"]].to_string(index=False))

    dist_df["disease"] = disease
    spearman_weekly["disease"] = disease
    by_state["disease"] = disease

    return dist_df, spearman_weekly, by_state, {
        "n_holdout_districts": int(df["district"].nunique()),
        "n_holdout_states": int(df["state"].nunique()),
        "n_weeks_evaluated": int(len(spearman_weekly)),
        "mean_spearman_rho": round(float(mean_rho), 4),
        "pct_weeks_significant": round(float(pct_sig), 1),
        "hotspot_detection_rate": round(float(det_rate), 4),
    }


def main():
    print("=== Phase 11 - Task 2: National Spatial Holdout Validation ===")
    print("108 districts across 28 states held out from training entirely.\n")

    all_dist, all_weekly, all_state = [], [], []
    summary = {}

    for disease in ["dengue", "malaria"]:
        dist_df, weekly, by_state, stats = run_disease(disease)
        all_dist.append(dist_df)
        all_weekly.append(weekly)
        all_state.append(by_state)
        summary[disease] = stats

    pd.concat(all_dist,   ignore_index=True).to_csv(OUT_DIR / "spatial_holdout_validation.csv",  index=False)
    pd.concat(all_weekly, ignore_index=True).to_csv(OUT_DIR / "spatial_holdout_weekly.csv",       index=False)
    pd.concat(all_state,  ignore_index=True).to_csv(OUT_DIR / "spatial_holdout_by_state.csv",     index=False)

    with open(OUT_DIR / "spatial_holdout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] District-week data -> {OUT_DIR / 'spatial_holdout_validation.csv'}")
    print(f"[OK] Per-state summary  -> {OUT_DIR / 'spatial_holdout_by_state.csv'}")
    print(f"[OK] Summary JSON       -> {OUT_DIR / 'spatial_holdout_summary.json'}")
    print("\n=== Spatial Holdout Validation Complete ===")


if __name__ == "__main__":
    main()
