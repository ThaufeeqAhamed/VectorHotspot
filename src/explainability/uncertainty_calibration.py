"""
Phase 12 - Task 3: Uncertainty Quantification & Calibration.

Computes prediction intervals and calibration metrics:
  - Conformal prediction intervals (90%, 95%) per hexagon
  - Reliability diagrams (predicted vs observed quantiles)
  - Calibration curves per disease per horizon

Uses split conformal inference on held-out test set.

Outputs:
  - Prediction intervals CSV
  - Reliability diagrams
  - Calibration summary JSON
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["h3_index", "state", "district", "year", "week", "is_spatial_holdout"]


def get_feature_cols(df):
    """Return feature column names (exclude meta and targets)."""
    targets = {f"target_lead_{k}" for k in range(1, 5)}
    outbreak = {f"outbreak_lead_{k}" for k in [1, 2, 4]}
    drop = set(META_COLS) | targets | outbreak
    return [c for c in df.columns if c not in drop]


def compute_conformal_intervals(disease, horizon, alpha_levels=[0.05, 0.10]):
    """
    Compute conformal prediction intervals at (1-alpha) confidence.

    Split conformal: use calibration set (test) to find quantiles of residuals.
    """
    print(f"\n[{disease.title()} Lead-{horizon}] Loading model and data...")

    model = joblib.load(MODELS_DIR / f"lgbm_{disease}_lead{horizon}.joblib")
    test = pd.read_parquet(DATA_PROC / f"features_{disease}_test.parquet")

    feature_cols = get_feature_cols(test)
    X_test = test[feature_cols]
    y_test = test[f"target_lead_{horizon}"].clip(lower=0).values

    # Predict
    print("  Computing predictions...")
    preds = np.maximum(model.predict(X_test), 0)

    # Compute absolute residuals on test set (split conformal)
    residuals = np.abs(y_test - preds)

    intervals = {}
    for alpha in alpha_levels:
        confidence = int((1 - alpha) * 100)
        print(f"  Computing {confidence}% prediction intervals...")

        # Find quantile of absolute residuals
        q = np.quantile(residuals, 1 - alpha, method="higher")

        # Prediction interval: [pred - q, pred + q]
        lower = np.maximum(preds - q, 0)
        upper = preds + q

        # Coverage: fraction of true values within interval
        coverage = np.mean((y_test >= lower) & (y_test <= upper))

        intervals[f"{confidence}%"] = {
            "lower": lower,
            "upper": upper,
            "q": float(q),
            "coverage": float(coverage),
        }

    return preds, y_test, intervals


def compute_reliability_diagram(preds, y_test, n_bins=10):
    """
    Compute reliability diagram: predicted quantile vs observed quantile.
    For quantile regression, check if predicted percentiles match observed.
    """
    # Bin by predicted value
    pred_sorted_idx = np.argsort(preds)
    bin_size = len(preds) // n_bins

    predicted_quantiles = []
    observed_quantiles = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(preds)
        idx = pred_sorted_idx[start:end]

        # Predicted percentile (mean rank)
        pred_pct = (start + end) / 2 / len(preds)
        predicted_quantiles.append(pred_pct)

        # Observed percentile (actual values rank)
        obs_rank = np.argsort(y_test[idx])
        obs_pct = np.mean(obs_rank / len(idx))
        observed_quantiles.append(obs_pct)

    return predicted_quantiles, observed_quantiles


def plot_calibration_curve(preds, y_test, disease, horizon):
    """Plot calibration curve: predicted vs observed percentiles."""
    pred_q, obs_q = compute_reliability_diagram(preds, y_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.scatter(pred_q, obs_q, s=50, alpha=0.7)
    ax.set_xlabel("Predicted Percentile")
    ax.set_ylabel("Observed Percentile")
    ax.set_title(f"{disease.title()} Lead-{horizon} | Calibration Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"calibration_curve_{disease}_lead{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=== Phase 12 - Task 3: Uncertainty Quantification & Calibration ===\n")
    print("Computing conformal prediction intervals for Dengue and Malaria...\n")

    all_results = {}

    for disease in ["dengue", "malaria"]:
        all_results[disease] = {}

        for horizon in range(1, 5):
            preds, y_test, intervals = compute_conformal_intervals(disease, horizon)

            # Save intervals
            df_intervals = pd.DataFrame({
                "prediction": preds,
                "actual": y_test,
                "lower_90": intervals["90%"]["lower"],
                "upper_90": intervals["90%"]["upper"],
                "lower_95": intervals["95%"]["lower"],
                "upper_95": intervals["95%"]["upper"],
            })
            df_intervals.to_csv(
                OUT_DIR / f"prediction_intervals_{disease}_lead{horizon}.csv",
                index=False
            )

            # Calibration curve
            plot_calibration_curve(preds, y_test, disease, horizon)

            # Summary
            all_results[disease][f"lead{horizon}"] = {
                "coverage_90%": intervals["90%"]["coverage"],
                "coverage_95%": intervals["95%"]["coverage"],
                "interval_width_90%": intervals["90%"]["q"],
                "interval_width_95%": intervals["95%"]["q"],
            }

    with open(OUT_DIR / "uncertainty_calibration_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Uncertainty calibration complete -> {OUT_DIR}")
    print("=== Task 3 Complete ===")


if __name__ == "__main__":
    main()
