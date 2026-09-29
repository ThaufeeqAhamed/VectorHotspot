"""
Phase 12 - Task 2: Dual-Disease Ecological Divergence Analysis.

Analyzes geographic and environmental patterns distinguishing Dengue
from Malaria transmission:
  - Co-hotspot overlap: regions where both diseases are hotspots
  - Urban vs Rural disease dominance patterns
  - Environmental driver divergence (urbanization, forest, water)

Outputs:
  - Co-hotspot overlap statistics per state
  - Urban/rural disease dominance maps
  - Environmental covariate correlation analysis
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HOT_DIR = ROOT / "outputs" / "hotspots"
OUT_DIR = ROOT / "outputs" / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hotspots():
    """Load predicted hotspot data for both diseases."""
    dengue = pd.read_parquet(HOT_DIR / "dengue_hotspots_test_2023_2024.parquet")
    malaria = pd.read_parquet(HOT_DIR / "malaria_hotspots_test_2023_2024.parquet")
    return dengue, malaria


def compute_cohotspot_overlap(dengue, malaria):
    """Compute co-hotspot statistics: hexagons hot for both diseases."""
    # Use lead-1 predictions
    dengue_hot = dengue[["h3_index", "state", "district", "year", "week", "is_hotspot_pred_lead_1"]].copy()
    malaria_hot = malaria[["h3_index", "state", "district", "year", "week", "is_hotspot_pred_lead_1"]].copy()

    # Merge on (h3_index, year, week)
    merged = dengue_hot.merge(
        malaria_hot,
        on=["h3_index", "year", "week"],
        suffixes=("_dengue", "_malaria"),
        how="inner"
    )

    # Co-hotspot = both are hotspots
    merged["is_cohotspot"] = (merged["is_hotspot_pred_lead_1_dengue"] == 1) & (merged["is_hotspot_pred_lead_1_malaria"] == 1)
    merged["dengue_only"] = (merged["is_hotspot_pred_lead_1_dengue"] == 1) & (merged["is_hotspot_pred_lead_1_malaria"] == 0)
    merged["malaria_only"] = (merged["is_hotspot_pred_lead_1_dengue"] == 0) & (merged["is_hotspot_pred_lead_1_malaria"] == 1)

    return merged


def analyze_cohotspot_by_state(merged):
    """Per-state co-hotspot statistics."""
    state_stats = merged.groupby("state_dengue").agg(
        total_hex_weeks=("h3_index", "count"),
        n_cohotspot=("is_cohotspot", "sum"),
        n_dengue_only=("dengue_only", "sum"),
        n_malaria_only=("malaria_only", "sum"),
    ).reset_index()
    state_stats.columns = ["state", "total_hex_weeks", "n_cohotspot", "n_dengue_only", "n_malaria_only"]

    state_stats["pct_cohotspot"] = (state_stats["n_cohotspot"] / state_stats["total_hex_weeks"] * 100).round(2)
    state_stats["pct_dengue_only"] = (state_stats["n_dengue_only"] / state_stats["total_hex_weeks"] * 100).round(2)
    state_stats["pct_malaria_only"] = (state_stats["n_malaria_only"] / state_stats["total_hex_weeks"] * 100).round(2)

    return state_stats.sort_values("pct_cohotspot", ascending=False)


def analyze_environmental_divergence(dengue_hotspots, malaria_hotspots):
    """Correlate environmental features with disease-specific hotspots."""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_PROC = ROOT / "data" / "processed"

    # Load feature stores which have environmental columns
    test_dengue = pd.read_parquet(DATA_PROC / "features_dengue_test.parquet")
    test_malaria = pd.read_parquet(DATA_PROC / "features_malaria_test.parquet")

    env_cols = ["frac_built", "frac_trees", "frac_water", "frac_shrub", "jrc_occurrence", "ndvi_mean"]
    available_env = [c for c in env_cols if c in test_dengue.columns]
    print(f"  Available environmental columns: {available_env}")

    if not available_env:
        print("  WARNING: No environmental columns found. Skipping.")
        return pd.DataFrame()

    # Get h3_index of hotspots
    dengue_hot_h3 = dengue_hotspots[dengue_hotspots["is_hotspot_pred_lead_1"] == 1]["h3_index"].unique()
    malaria_hot_h3 = malaria_hotspots[malaria_hotspots["is_hotspot_pred_lead_1"] == 1]["h3_index"].unique()

    # Filter feature store by hotspot h3_index
    dengue_env = test_dengue[test_dengue["h3_index"].isin(dengue_hot_h3)][available_env + ["h3_index"]].drop_duplicates("h3_index")
    dengue_env["disease"] = "dengue"

    malaria_env = test_malaria[test_malaria["h3_index"].isin(malaria_hot_h3)][available_env + ["h3_index"]].drop_duplicates("h3_index")
    malaria_env["disease"] = "malaria"

    combined = pd.concat([dengue_env, malaria_env], ignore_index=True)

    # Compute mean per disease
    summary = combined.groupby("disease")[env_cols].mean().T
    summary.columns = ["dengue_mean", "malaria_mean"]
    summary["delta"] = summary["dengue_mean"] - summary["malaria_mean"]
    summary = summary.sort_values("delta", ascending=False)

    return summary


def plot_cohotspot_barplot(state_stats):
    """Bar plot: co-hotspot % by state (top 15)."""
    top15 = state_stats.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(top15))
    width = 0.25

    ax.bar(x - width, top15["pct_dengue_only"], width, label="Dengue Only", color="#e74c3c")
    ax.bar(x, top15["pct_cohotspot"], width, label="Co-Hotspot", color="#9b59b6")
    ax.bar(x + width, top15["pct_malaria_only"], width, label="Malaria Only", color="#3498db")

    ax.set_xlabel("State")
    ax.set_ylabel("% of Hex-Weeks")
    ax.set_title("Hotspot Overlap by State (Top 15 by Co-Hotspot %)")
    ax.set_xticks(x)
    ax.set_xticklabels(top15["state"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cohotspot_by_state.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_environmental_heatmap(env_summary):
    """Heatmap: environmental divergence between diseases."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        env_summary[["dengue_mean", "malaria_mean"]],
        annot=True, fmt=".3f", cmap="coolwarm", center=0,
        cbar_kws={"label": "Mean Value"}
    )
    plt.title("Environmental Covariate Means: Dengue vs Malaria Hotspots")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "environmental_divergence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=== Phase 12 - Task 2: Dual-Disease Ecological Divergence ===\n")

    print("Loading hotspot data...")
    dengue, malaria = load_hotspots()

    print("Computing co-hotspot overlap...")
    merged = compute_cohotspot_overlap(dengue, malaria)

    print("Analyzing co-hotspot by state...")
    state_stats = analyze_cohotspot_by_state(merged)
    state_stats.to_csv(OUT_DIR / "cohotspot_by_state.csv", index=False)
    print(f"  Top 5 states by co-hotspot %:")
    print(state_stats[["state", "pct_cohotspot", "pct_dengue_only", "pct_malaria_only"]].head(5).to_string(index=False))

    print("\nAnalyzing environmental divergence...")
    env_summary = analyze_environmental_divergence(dengue, malaria)
    env_summary.to_csv(OUT_DIR / "environmental_divergence.csv")
    print(env_summary.to_string())

    print("\nGenerating plots...")
    plot_cohotspot_barplot(state_stats)
    plot_environmental_heatmap(env_summary)

    summary_json = {
        "total_hex_weeks": int(merged.shape[0]),
        "n_cohotspot": int(merged["is_cohotspot"].sum()),
        "n_dengue_only": int(merged["dengue_only"].sum()),
        "n_malaria_only": int(merged["malaria_only"].sum()),
        "pct_cohotspot": round(float(merged["is_cohotspot"].mean() * 100), 2),
    }
    with open(OUT_DIR / "ecological_divergence_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n[OK] Ecological divergence analysis complete -> {OUT_DIR}")
    print("=== Task 2 Complete ===")


if __name__ == "__main__":
    main()
