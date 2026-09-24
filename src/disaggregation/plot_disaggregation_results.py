#!/usr/bin/env python3
"""
Generate publication-quality comparative visualization of 2024 disaggregated
dengue and malaria risk surfaces across India.

Saves output to: outputs/figures/disaggregation_real_results_2024.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading datasets for 2024 visualization...")
    grid = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'india_h3_grid_res7.csv',
        encoding='utf-8-sig'
    )[['h3_index', 'center_lat', 'center_lon']]

    # Load 2024 Dengue
    print("Loading 2024 Dengue disaggregated predictions...")
    dengue = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'dengue_hex_annual.csv',
        encoding='utf-8-sig'
    )
    d2024 = dengue[dengue['year'] == 2024].merge(grid, on='h3_index', how='inner')

    # Load 2024 Malaria
    print("Loading 2024 Malaria disaggregated predictions...")
    malaria = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'malaria_hex_annual.csv',
        encoding='utf-8-sig'
    )
    m2024 = malaria[malaria['year'] == 2024].merge(grid, on='h3_index', how='inner')

    print(f"Dengue 2024 total cases: {d2024['dengue_cases'].sum():,.0f} across {len(d2024):,} hexagons")
    print(f"Malaria 2024 total cases: {m2024['malaria_cases'].sum():,.0f} across {len(m2024):,} hexagons")

    # Plot side-by-side comparison
    print("Rendering 2-panel comparative map...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor='white')

    # Set background color
    for ax in (ax1, ax2):
        ax.set_facecolor('#f8f9fa')
        ax.set_aspect('equal')
        ax.set_xlim(68, 98)
        ax.set_ylim(6, 38)
        ax.set_xlabel('Longitude (deg E)', fontsize=11)
        ax.set_ylabel('Latitude (deg N)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

    # 1. Dengue Plot
    # Filter for non-zero cases for log scale visualization
    d_nonzero = d2024[d2024['dengue_cases'] > 0.001].copy()
    # Plot background (zero/very low case cells)
    ax1.scatter(d2024['center_lon'], d2024['center_lat'], s=0.2, color='#e2e8f0', alpha=0.5, rasterized=True)
    # Plot positive cases with colormap
    sc1 = ax1.scatter(
        d_nonzero['center_lon'], d_nonzero['center_lat'],
        c=d_nonzero['dengue_cases'],
        cmap='YlOrRd',
        norm=LogNorm(vmin=0.01, vmax=max(50, d_nonzero['dengue_cases'].quantile(0.999))),
        s=0.8,
        alpha=0.85,
        rasterized=True
    )
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.035, pad=0.04)
    cbar1.set_label('Estimated Cases / Hexagon (~5.2 km2)', fontsize=10)
    ax1.set_title(
        '2024 Disaggregated Dengue Risk Surface (H3 Res 7)\n'
        f'Total Cases: {d2024["dengue_cases"].sum():,.0f} | b1 = +0.766 (Urban-concentrated)',
        fontsize=12, fontweight='bold', pad=12
    )

    # 2. Malaria Plot
    m_nonzero = m2024[m2024['malaria_cases'] > 0.001].copy()
    ax2.scatter(grid['center_lon'], grid['center_lat'], s=0.2, color='#e2e8f0', alpha=0.5, rasterized=True)
    sc2 = ax2.scatter(
        m_nonzero['center_lon'], m_nonzero['center_lat'],
        c=m_nonzero['malaria_cases'],
        cmap='YlGnBu',
        norm=LogNorm(vmin=0.01, vmax=max(20, m_nonzero['malaria_cases'].quantile(0.999))),
        s=0.8,
        alpha=0.85,
        rasterized=True
    )
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.035, pad=0.04)
    cbar2.set_label('Estimated Cases / Hexagon (~5.2 km2)', fontsize=10)
    ax2.set_title(
        '2024 Disaggregated Malaria Risk Surface (H3 Res 7)\n'
        f'Total Cases: {m2024["malaria_cases"].sum():,.0f} | b1 = -1.249 (Rural/Forest-concentrated)',
        fontsize=12, fontweight='bold', pad=12
    )

    plt.suptitle(
        'VectorHotspot: Dual-Disease Mass-Preserving Spatial Disaggregation (India 2024)',
        fontsize=15, fontweight='bold', y=0.98
    )

    out_path = OUTPUT_DIR / 'disaggregation_real_results_2024.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SUCCESS: Figure saved to {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == '__main__':
    main()
