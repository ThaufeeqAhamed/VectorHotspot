#!/usr/bin/env python3
"""
Phase 11 Visualization: Future Hotspot Prediction Performance
VectorHotspot Project

Analyzes the output of Phase 10 (Getis-Ord Gi* Fusion) to generate diagnostic figures:
1. Analysis of Spatial Intersection over Union (IoU) between Predicted and Observed hotspots.
2. Distribution of the 4-Tier Hotspot Taxonomy (Emerging, Intensifying, Persistent, Diminishing).
3. Early Warning Lead Time verification (how early we catch future outbreaks).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HOTSPOTS_DIR = PROJECT_ROOT / "outputs" / "hotspots"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_OUT = OUTPUTS_FIGURES_DIR / "hotspot_fusion_diagnostics.png"
HORIZONS = [1, 2, 3, 4]


def calculate_hit_metrics(disease):
    """
    Computes Spatial IoU, Precision, and Recall between Predicted hotspots
    (from Gi* on Predicted risk) vs Actual hotspots (from Gi* on Actual cases).
    """
    path = HOTSPOTS_DIR / f"{disease}_hotspots_test_2023_2024.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    metrics = []

    for k in HORIZONS:
        col_pred = f'is_hotspot_pred_lead_{k}'
        col_act = f'is_hotspot_act_lead_{k}'

        intersection = ((df[col_pred] == 1) & (df[col_act] == 1)).sum()
        union = ((df[col_pred] == 1) | (df[col_act] == 1)).sum()

        pred_pos = df[col_pred].sum()
        act_pos = df[col_act].sum()

        iou = intersection / union if union > 0 else 0
        precision = intersection / pred_pos if pred_pos > 0 else 0
        recall = intersection / act_pos if act_pos > 0 else 0

        metrics.append({
            'horizon': f't+{k}',
            'iou': iou,
            'precision': precision,
            'recall': recall,
        })

    return pd.DataFrame(metrics)


def main():
    print("Generating Phase 10 Hotspot Fusion Diagnostics...")

    tax_dengue_path = HOTSPOTS_DIR / "dengue_hotspot_taxonomy_2024.parquet"
    tax_mal_path = HOTSPOTS_DIR / "malaria_hotspot_taxonomy_2024.parquet"

    metrics_dengue = calculate_hit_metrics('dengue')
    metrics_malaria = calculate_hit_metrics('malaria')

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    width = 0.25

    # -------------------------------------------------------------------------
    # Panel A: Dengue Predicted vs Observed Spatial IoU / F1
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if len(metrics_dengue) > 0:
        x = np.arange(len(metrics_dengue['horizon']))

        ax1.bar(x - width, metrics_dengue['iou'], width, label='Spatial IoU', color='#2b83ba')
        ax1.bar(x, metrics_dengue['precision'], width, label='Precision', color='#abdda4')
        ax1.bar(x + width, metrics_dengue['recall'], width, label='Recall', color='#d7191c')

        ax1.set_title('(A) Dengue Future Hotspot Detection Performance', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('Forecast Horizon (Weeks Ahead)', fontsize=11)
        ax1.set_ylabel('Score [0 to 1]', fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_dengue['horizon'])
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel B: Malaria Predicted vs Observed Spatial IoU / F1
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    if len(metrics_malaria) > 0:
        x = np.arange(len(metrics_malaria['horizon']))

        ax2.bar(x - width, metrics_malaria['iou'], width, label='Spatial IoU', color='#2b83ba')
        ax2.bar(x, metrics_malaria['precision'], width, label='Precision', color='#abdda4')
        ax2.bar(x + width, metrics_malaria['recall'], width, label='Recall', color='#d7191c')

        ax2.set_title('(B) Malaria Future Hotspot Detection Performance', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Forecast Horizon (Weeks Ahead)', fontsize=11)
        ax2.set_ylabel('Score [0 to 1]', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_malaria['horizon'])
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel C & D: Taxonomy Distribution
    # -------------------------------------------------------------------------
    colors = {
        'Emerging Hotspot': '#e41a1c',
        'Intensifying Hotspot': '#ff7f00',
        'Persistent Hotspot': '#377eb8',
        'Diminishing Hotspot': '#4daf4a'
    }

    if tax_dengue_path.exists():
        ax3 = fig.add_subplot(gs[1, 0])
        dengue_counts = pd.read_parquet(tax_dengue_path)['hotspot_taxonomy'].value_counts()
        ax3.pie(dengue_counts.values, labels=dengue_counts.index,
                autopct='%1.1f%%', startangle=140,
                colors=[colors.get(c, '#999999') for c in dengue_counts.index],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
        ax3.set_title('(C) Dengue Early-Warning Flags by Type (2024)', fontsize=13, fontweight='bold', pad=10)

    if tax_mal_path.exists():
        ax4 = fig.add_subplot(gs[1, 1])
        mal_counts = pd.read_parquet(tax_mal_path)['hotspot_taxonomy'].value_counts()
        ax4.pie(mal_counts.values, labels=mal_counts.index,
                autopct='%1.1f%%', startangle=140,
                colors=[colors.get(c, '#999999') for c in mal_counts.index],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
        ax4.set_title('(D) Malaria Early-Warning Flags by Type (2024)', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=300, bbox_inches='tight')
    print(f"Hotspot evaluation figure successfully saved to: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
