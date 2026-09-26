#!/usr/bin/env python3
"""
Phase 8 Visualization: Spatiotemporal Feature Store Diagnostics
VectorHotspot Project

Generates publication-quality figures evaluating:
1. Autoregressive and Spatial Lag Correlation Decay.
2. Feature Correlation Matrix with Multi-Horizon Targets.
3. Biophysical Suitability vs Observed Outbreak Risk.
4. Spatiotemporal Train/Val/Test Partitioning Schema.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_OUT = OUTPUTS_DIR / "feature_store_diagnostics.png"


def main():
    print("Generating Phase 8 Feature Store diagnostic plots...")

    train_dengue = PROCESSED_DATA_DIR / "features_dengue_train.parquet"
    if not train_dengue.exists():
        print(f"Waiting for {train_dengue.name} to complete generation...")
        return

    df_dengue = pd.read_parquet(train_dengue)
    df_sample = df_dengue.sample(min(len(df_dengue), 100000), random_state=42)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel A: Autoregressive & Spatial Correlation Decay
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    lags = [1, 2, 3, 4, 8]
    auto_corr = [
        np.corrcoef(df_sample['cases_lag_1'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['cases_lag_2'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['cases_lag_3'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['cases_lag_4'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['cases_lag_8'], df_sample['target_lead_1'])[0, 1]
    ]

    nbr_corr = [
        np.corrcoef(df_sample['neighbor_cases_k1_lag1'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['neighbor_cases_k1_lag2'], df_sample['target_lead_1'])[0, 1],
        np.corrcoef(df_sample['neighbor_cases_k2_lag1'], df_sample['target_lead_1'])[0, 1]
    ]

    ax1.plot(lags, auto_corr, marker='o', lw=2.5, color='#d95f02', label='Hexagon Autoregressive Lag (t-k)')
    ax1.plot([1, 2], nbr_corr[:2], marker='s', lw=2.5, linestyle='--', color='#7570b3', label='H3 k=1 Neighbor Spillover (t-k)')
    ax1.scatter([1], [nbr_corr[2]], marker='^', s=100, color='#1b9e77', label='H3 k=2 Neighbor Spillover (t-1)', zorder=5)

    ax1.set_title('(A) Spatiotemporal Autocorrelation & Spatial Neighbor Decay', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Historical Lag Horizon (Weeks)', fontsize=11)
    ax1.set_ylabel('Correlation with Target (t+1)', fontsize=11)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc='upper right', frameon=True, fontsize=10)

    # -------------------------------------------------------------------------
    # Panel B: Feature Correlation Ranking with 1-Week Ahead Target
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    feature_cols = [
        'cases_lag_1', 'cases_roll_mean_4w', 'neighbor_cases_k1_lag1',
        'neighbor_cases_k2_lag1', 'cases_lag_4', 'cases_roll_mean_12w',
        'suitability_lag_1', 'log_population', 'frac_built',
        'tmean_lag_1', 'rain_roll_sum_4w', 'ndvi_mean'
    ]

    corrs = [np.corrcoef(df_sample[c], df_sample['target_lead_1'])[0, 1] for c in feature_cols]
    feature_labels = [
        'Cases (t-1)', 'Cases (4w Moving Mean)', 'Neighbor Cases k=1 (t-1)',
        'Neighbor Cases k=2 (t-1)', 'Cases (t-4)', 'Cases (12w Moving Mean)',
        'Biophysical Suitability (t-1)', 'Log Population', 'Urban Built Fraction',
        'Mean Temperature (t-1)', 'Rainfall 4w Cumulative', 'MODIS NDVI Mean'
    ]

    sorted_idx = np.argsort(corrs)
    y_pos = np.arange(len(corrs))
    colors = ['#386cb0' if 'Cases' in feature_labels[i] or 'Neighbor' in feature_labels[i] else '#f0027f' for i in sorted_idx]

    ax2.barh(y_pos, [corrs[i] for i in sorted_idx], color=colors, height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([feature_labels[i] for i in sorted_idx], fontsize=9.5)
    ax2.set_title('(B) Feature Correlation with 1-Week Ahead Target y(t+1)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=11)
    ax2.set_xlim(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Panel C: Outbreak Probability vs Biophysical Suitability Deciles
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    df_sample['suit_bin'] = pd.qcut(df_sample['suitability_lag_1'], q=10, duplicates='drop')
    bin_stats = df_sample.groupby('suit_bin', observed=True).agg(
        mean_suit=('suitability_lag_1', 'mean'),
        outbreak_rate=('outbreak_lead_1', 'mean')
    ).reset_index()

    ax3.plot(bin_stats['mean_suit'], bin_stats['outbreak_rate'] * 100, marker='o', lw=2.5, color='#e41a1c')
    ax3.fill_between(bin_stats['mean_suit'], 0, bin_stats['outbreak_rate'] * 100, color='#e41a1c', alpha=0.15)
    ax3.set_title('(C) Outbreak Probability Across Biophysical Suitability Deciles', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Biophysical Suitability Index (t-1)', fontsize=11)
    ax3.set_ylabel('Outbreak Probability (% cells > 90th percentile)', fontsize=11)

    # -------------------------------------------------------------------------
    # Panel D: Multi-Horizon Target Correlation Matrix
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    target_cols = ['cases_lag_1', 'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4']
    target_names = ['Current (t)', 'Lead t+1', 'Lead t+2', 'Lead t+3', 'Lead t+4']
    corr_mat = df_sample[target_cols].corr()

    sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax4,
                xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Pearson r'})
    ax4.set_title('(D) Autoregressive Persistence Across Forecast Horizons (t+1 to t+4)', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=300, bbox_inches='tight')
    print(f"Feature diagnostics plot successfully saved to: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
