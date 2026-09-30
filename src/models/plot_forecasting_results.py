#!/usr/bin/env python3
"""
Phase 9 Visualization: Forecasting Model Performance & Benchmarks
VectorHotspot Project

Generates publication-grade figures evaluating:
1. Multi-horizon forecasting accuracy comparison (LightGBM vs XGBoost vs Baselines).
2. Predicted vs Observed future case calibration (1:1 line).
3. Spatial generalization comparison (Seen vs 15% Unseen Held-out Districts).
4. Time-series forecast trajectory during transmission peaks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

METRICS_CSV = OUTPUTS_TABLES_DIR / "model_evaluation_metrics.csv"
FIGURE_OUT = OUTPUTS_FIGURES_DIR / "forecasting_models_evaluation.png"


def main():
    print("Generating Phase 9 Forecasting Model evaluation figures...")
    if not METRICS_CSV.exists():
        print(f"Metrics CSV {METRICS_CSV.name} not found! Run train_forecasting_models.py first.")
        return

    df_metrics = pd.read_csv(METRICS_CSV)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel A: Multi-Horizon R2 Comparison (Dengue & Malaria)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    df_seen = df_metrics[df_metrics['split'] == 'Test (Seen Districts)']

    models_palette = {
        'LightGBM Regressor (Proposed)': '#2b83ba',
        'XGBoost Regressor': '#abdda4',
        'Baseline 1: Naive Persistence': '#fdae61',
        'Baseline 2: Historical Seasonal': '#d7191c'
    }

    horizons = ['t+1', 't+2', 't+3', 't+4']
    for model_name, col in models_palette.items():
        sub = df_seen[(df_seen['disease'] == 'dengue') & (df_seen['model'] == model_name)]
        if len(sub) > 0:
            ax1.plot(horizons, sub['r2'], marker='o', lw=2.4, color=col, label=f'Dengue: {model_name}')

    ax1.set_title('(A) Dengue Multi-Horizon Accuracy Decay (R² across t+1 to t+4)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Forecast Horizon (Weeks Ahead)', fontsize=11)
    ax1.set_ylabel('Test Set R² Score', fontsize=11)
    ax1.set_ylim(-0.05, 1.0)
    ax1.legend(loc='lower left', frameon=True, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel B: Predicted vs Observed Scatter / Calibration (Horizon t+2)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    pred_path = PROCESSED_DATA_DIR / "forecast_dengue_predictions.parquet"
    if pred_path.exists():
        df_preds = pd.read_parquet(pred_path)
        sample_pts = df_preds.sample(min(len(df_preds), 15000), random_state=42)

        y_act = sample_pts['actual_lead_2'].values
        y_prd = sample_pts['pred_lgbm_lead_2'].values

        ax2.scatter(y_act, y_prd, alpha=0.35, color='#2b83ba', s=16, edgecolors='none', label='Predictions (t+2 weeks)')
        max_val = max(np.percentile(y_act, 99.5), np.percentile(y_prd, 99.5))
        ax2.plot([0, max_val], [0, max_val], color='#d7191c', linestyle='--', lw=2.0, label='1:1 Perfect Calibration')

        ax2.set_xlim(-0.01, max_val * 1.05)
        ax2.set_ylim(-0.01, max_val * 1.05)
        ax2.set_title('(B) Dengue Forecast Calibration at Horizon t+2 (2-Weeks Ahead)', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Observed Cases at Week t+2', fontsize=11)
        ax2.set_ylabel('LightGBM Predicted Cases at Week t+2', fontsize=11)
        ax2.legend(loc='upper left', frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel C: Spatial Generalization Benchmark (Seen vs Unseen Holdout Districts)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    df_lgbm = df_metrics[df_metrics['model'] == 'LightGBM Regressor (Proposed)']

    x = np.arange(len(horizons))
    width = 0.35

    dengue_seen = df_lgbm[(df_lgbm['disease'] == 'dengue') & (df_lgbm['split'] == 'Test (Seen Districts)')]['r2'].values
    dengue_unseen = df_lgbm[(df_lgbm['disease'] == 'dengue') & (df_lgbm['split'] == 'Test (Spatial Holdout)')]['r2'].values

    if len(dengue_seen) == 4 and len(dengue_unseen) == 4:
        ax3.bar(x - width/2, dengue_seen, width, label='Seen Districts (2023-2024)', color='#2b83ba')
        ax3.bar(x + width/2, dengue_unseen, width, label='15% Unseen Spatial Holdout', color='#fdae61')

    ax3.set_title('(C) Spatial Generalization Benchmark (Dengue R²)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Forecast Horizon (Weeks Ahead)', fontsize=11)
    ax3.set_ylabel('Test R² Score', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons)
    ax3.set_ylim(0.0, 1.0)
    ax3.legend(loc='upper right', frameon=True, fontsize=10)

    # -------------------------------------------------------------------------
    # Panel D: Malaria Multi-Horizon Accuracy Comparison
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    for model_name, col in models_palette.items():
        sub = df_seen[(df_seen['disease'] == 'malaria') & (df_seen['model'] == model_name)]
        if len(sub) > 0:
            ax4.plot(horizons, sub['r2'], marker='s', lw=2.4, color=col, label=f'Malaria: {model_name}')

    ax4.set_title('(D) Malaria Multi-Horizon Accuracy Decay (R² across t+1 to t+4)', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Forecast Horizon (Weeks Ahead)', fontsize=11)
    ax4.set_ylabel('Test Set R² Score', fontsize=11)
    ax4.set_ylim(0.0, 1.05)
    ax4.legend(loc='lower left', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=300, bbox_inches='tight')
    print(f"Forecasting evaluation figure successfully saved to: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
