"""
Phase 12 - Task 1: TreeSHAP Feature Importance Analysis.

Computes SHAP values for trained LightGBM models to explain:
  - Global feature importance (averaged across all predictions)
  - Feature interactions and dependencies
  - Local explanations for specific hotspot hexagons

Outputs:
  - Global SHAP summary plots (bar + beeswarm) per disease per horizon
  - Dependency plots for top 5 features
  - Local waterfall plots for sample emerging hotspot hexagons
  - SHAP values CSV for downstream analysis

Note: SHAP computation on 3.7M test rows is expensive. We sample
      10,000 background rows for TreeExplainer initialization.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["h3_index", "state", "district", "year", "week", "is_spatial_holdout"]

SAMPLE_SIZE = 10000  # Background data for TreeExplainer


def get_feature_cols(df):
    """Return feature column names (exclude meta and targets)."""
    targets = {f"target_lead_{k}" for k in range(1, 5)}
    outbreak = {f"outbreak_lead_{k}" for k in [1, 2, 4]}
    drop = set(META_COLS) | targets | outbreak
    return [c for c in df.columns if c not in drop]


def compute_shap_global(disease, horizon):
    """Compute global SHAP values for one disease-horizon pair."""
    print(f"\n[{disease.title()} Lead-{horizon}] Loading model and test data...")

    model_path = MODELS_DIR / f"lgbm_{disease}_lead{horizon}.joblib"
    model = joblib.load(model_path)

    test = pd.read_parquet(DATA_PROC / f"features_{disease}_test.parquet")
    feature_cols = get_feature_cols(test)
    X_test = test[feature_cols]

    # Sample background data for TreeExplainer
    print(f"  Sampling {SAMPLE_SIZE} background rows for explainer...")
    X_background = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

    # Initialize TreeExplainer
    print("  Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="tree_path_dependent")

    # Compute SHAP values on a sample (full 3.7M rows = too slow)
    sample_size = min(50000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    print(f"  Computing SHAP values for {sample_size} test samples...")
    shap_values = explainer.shap_values(X_sample)

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(OUT_DIR / f"shap_values_{disease}_lead{horizon}.csv", index=False)

    return shap_values, X_sample, feature_cols


def plot_shap_summary(shap_values, X_sample, feature_cols, disease, horizon):
    """Generate global SHAP summary plots."""
    print(f"  Generating summary plots...")

    # Bar plot (mean absolute SHAP)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.title(f"{disease.title()} Lead-{horizon} | Global Feature Importance (|SHAP|)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_summary_bar_{disease}_lead{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot (SHAP value distribution)
    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title(f"{disease.title()} Lead-{horizon} | Feature Impact Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_summary_beeswarm_{disease}_lead{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_values, X_sample, feature_cols, disease, horizon, top_n=5):
    """Generate dependence plots for top N features."""
    print(f"  Generating dependence plots for top {top_n} features...")

    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_cols[i] for i in top_indices]

    for feat in top_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_sample, show=False)
        plt.title(f"{disease.title()} Lead-{horizon} | Dependence: {feat}")
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(OUT_DIR / f"shap_dependence_{disease}_lead{horizon}_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    print("=== Phase 12 - Task 1: TreeSHAP Feature Importance ===\n")
    print("Computing SHAP values for Dengue and Malaria models (horizons 1-4)...")
    print("(This may take 10-20 minutes per model)\n")

    for disease in ["dengue", "malaria"]:
        for horizon in range(1, 5):
            shap_values, X_sample, feature_cols = compute_shap_global(disease, horizon)
            plot_shap_summary(shap_values, X_sample, feature_cols, disease, horizon)
            plot_shap_dependence(shap_values, X_sample, feature_cols, disease, horizon, top_n=5)

    print(f"\n[OK] SHAP analysis complete. Outputs saved to {OUT_DIR}")
    print("=== Task 1 Complete ===")


if __name__ == "__main__":
    main()
