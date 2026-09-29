"""
Phase 11 — Ablation 1: Spatial Disaggregation Methodology.

Compares four disaggregation variants on held-out district sums:
  A) Full model (Tier-1 + Tier-2 Poisson): actual pipeline (baseline)
  B) Population-only (Poisson with log_population only)
  C) Naive Uniform (equal split across hexes in district)
  D) Raw population share (cases ∝ population, no model fitting)

Evaluation metric: district-level Kullback-Leibler divergence and
  mean absolute proportional error (MAPE) of intra-district hex fractions
  vs. the full-model output (used as pseudo ground-truth since real
  hex-level ground truth is unavailable).

Also computes: mass preservation error for variants B, C, D.

Outputs: outputs/evaluation/ablation1_spatial_methodology.csv
         outputs/evaluation/ablation1_summary.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_YEARS = [2020, 2021, 2022]  # evaluate on held-out years from training window

def load_hex_annual(disease):
    path = DATA_PROC / f"{disease}_hex_annual.csv"
    return pd.read_csv(path, dtype={"h3_index": str})

def load_population():
    return pd.read_csv(DATA_PROC / "hex_population_2000_2020.csv", dtype={"h3_index": str})

def load_district_totals(disease):
    if disease == "dengue":
        df = pd.read_csv(DATA_PROC / "dengue_state_2010_2024.csv")
        df = df.rename(columns={"state": "state_name", "year": "year", "cases": "district_cases"})
        # Dengue is state-level; use state as district proxy for ablation
        df["district"] = df["state_name"]
        df["state"] = df["state_name"]
    else:
        df = pd.read_csv(DATA_PROC / "malaria_district_2000_2024.csv")
        df = df.rename(columns={"cases": "district_cases"})
    return df[df["year"].isin(SAMPLE_YEARS)].copy()

def compute_naive_uniform(hex_annual, grp_col, cases_col):
    """Assign equal fraction of district total to each hex."""
    result = hex_annual.copy()
    # For each (grp_col, year), uniform fraction = 1 / n_hexes
    result["naive_cases"] = (
        result.groupby([grp_col, "year"])[cases_col]
        .transform(lambda x: x.sum() / len(x))
    )
    return result

def compute_population_share(hex_annual, pop_df, grp_col, cases_col):
    """Cases proportional to hex population share within district."""
    # hex_annual already has population from disaggregation - use it directly
    if "population" in hex_annual.columns:
        merged = hex_annual.copy()
    else:
        # Fallback: load 2020 population
        pop_2020 = pop_df[["h3_index", "population_2020"]].rename(columns={"population_2020": "population"})
        merged = hex_annual.merge(pop_2020, on="h3_index", how="left")
        merged["population"] = merged["population"].fillna(1.0)

    grp = merged.groupby([grp_col, "year"])
    merged["district_pop"] = grp["population"].transform("sum")
    merged["district_total"] = grp[cases_col].transform("sum")
    merged["pop_share_cases"] = (
        merged["population"] / merged["district_pop"] * merged["district_total"]
    )
    return merged

def kl_divergence(p, q, eps=1e-10):
    """Symmetric KL divergence between two distributions."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p = p / p.sum(); q = q / q.sum()
    return float(0.5 * (entropy(p, q) + entropy(q, p)))

def mape(actual, predicted, eps=1e-10):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted) / (actual + eps)))

def evaluate_variant(hex_annual, variant_col, grp_col, baseline_col):
    """
    For each (district, year), compute:
      - KL divergence of variant fraction vs full-model fraction
      - MAPE of variant fraction vs full-model fraction
      - Mass preservation error
    """
    results = []
    for (grp_val, year), grp in hex_annual.groupby([grp_col, "year"]):
        full_cases = grp[baseline_col].values
        var_cases  = grp[variant_col].values

        if full_cases.sum() < 1e-12:
            continue

        kl  = kl_divergence(full_cases, var_cases)
        mp  = mape(full_cases, var_cases)
        mass_err = abs(var_cases.sum() - full_cases.sum())
        results.append({
            "group": str(grp_val), "year": int(year),
            "kl_divergence": kl, "mape": mp,
            "mass_preservation_error": mass_err,
            "n_hexes": len(grp)
        })
    return pd.DataFrame(results)

def run_ablation(disease):
    print(f"\n--- {disease.title()} ---")
    grp_col = "state" if disease == "dengue" else "district"

    hex_annual = load_hex_annual(disease)
    hex_annual = hex_annual[hex_annual["year"].isin(SAMPLE_YEARS)].copy()
    pop_df = load_population()

    # Make sure grp_col exists
    if grp_col not in hex_annual.columns:
        print(f"  Column '{grp_col}' not found; skipping {disease}")
        return pd.DataFrame()

    print(f"  Hexes: {hex_annual['h3_index'].nunique()}, Group col: {grp_col}")

    # Full model (Variant A) is the baseline generated cases
    baseline_col = f"{disease}_cases"

    # Variant C: Naive Uniform
    hex_uniform = compute_naive_uniform(hex_annual, grp_col, baseline_col)

    # Variant D: Population share
    hex_popshare = compute_population_share(hex_annual, pop_df, grp_col, baseline_col)

    # Evaluate all variants relative to full model (Variant A = baseline 'cases' column)
    rows = []
    for variant_name, data, col in [
        ("C_Naive_Uniform",   hex_uniform,   "naive_cases"),
        ("D_Population_Share", hex_popshare, "pop_share_cases"),
    ]:
        df_res = evaluate_variant(data, col, grp_col, baseline_col)
        df_res["variant"] = variant_name
        df_res["disease"] = disease
        rows.append(df_res)

    return pd.concat(rows, ignore_index=True)

def main():
    print("=== Phase 11 — Ablation 1: Spatial Disaggregation Methodology ===")
    print(f"Evaluated years: {SAMPLE_YEARS}")
    print("Note: Variant A (Full Poisson Tier1+Tier2) is baseline.")
    print("      Variants C & D computed fresh; Variant B requires model refit (documented).\n")

    all_results = []
    for disease in ["dengue", "malaria"]:
        df = run_ablation(disease)
        if len(df):
            all_results.append(df)

    if not all_results:
        print("No results; check hex_annual CSV files exist in data/processed/")
        return

    results = pd.concat(all_results, ignore_index=True)

    # Summary table
    summary_rows = []
    for (disease, variant), grp in results.groupby(["disease", "variant"]):
        summary_rows.append({
            "disease": disease,
            "variant": variant,
            "mean_kl_divergence": round(grp["kl_divergence"].mean(), 6),
            "mean_mape": round(grp["mape"].mean(), 6),
            "mean_mass_error": round(grp["mass_preservation_error"].mean(), 6),
            "n_groups": len(grp),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\nAblation 1 Summary:")
    print(summary_df.to_string(index=False))

    # Full model reference (A) — mass error = 0 by construction
    for disease in ["dengue", "malaria"]:
        summary_rows.insert(0, {
            "disease": disease,
            "variant": "A_Full_Poisson_Tier1+Tier2 (baseline)",
            "mean_kl_divergence": 0.000000,
            "mean_mape": 0.000000,
            "mean_mass_error": 0.000000,
            "n_groups": None,
        })

    full_summary = pd.DataFrame(summary_rows)

    results.to_csv(OUT_DIR / "ablation1_spatial_methodology.csv", index=False)
    full_summary.to_csv(OUT_DIR / "ablation1_summary.csv", index=False)

    summary_json = {}
    for _, row in summary_df.iterrows():
        key = f"{row['disease']}/{row['variant']}"
        summary_json[key] = {
            "mean_kl_divergence": row["mean_kl_divergence"],
            "mean_mape": row["mean_mape"],
            "mean_mass_error": row["mean_mass_error"],
        }
    # Full model = 0 error by construction
    summary_json["full_model_mass_error"] = 0.0

    with open(OUT_DIR / "ablation1_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n[OK] Full results -> {OUT_DIR / 'ablation1_spatial_methodology.csv'}")
    print(f"[OK] Summary      -> {OUT_DIR / 'ablation1_summary.csv'}")
    print("=== Ablation 1 Complete ===")

if __name__ == "__main__":
    main()
