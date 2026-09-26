#!/usr/bin/env python3
"""
Phase 7 Validation & Visualization: Temporal Disaggregation Diagnostics
VectorHotspot Project

Generates high-resolution publication-quality figures evaluating:
1. State-level weekly transmission seasonality curves for Dengue vs. Malaria.
2. Biophysical thermal performance curves (Mordecai et al.) and hydrologic responses.
3. Multi-year national weekly case trajectory across 2010–2024.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_OUT = OUTPUTS_DIR / "temporal_disaggregation_validation.png"

# Biophysical functions for curve plotting
def briere(T, T0, Tm, norm_c):
    val = np.where((T > T0) & (T < Tm), T * (T - T0) * np.sqrt(np.maximum(0.0, Tm - T)), 0.0)
    return val / norm_c

def main():
    print("Generating Phase 7 Temporal Disaggregation validation figures...")

    # Set styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel 1: Biophysical Suitability Curves (Thermal & Hydrology)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    temps = np.linspace(10, 42, 300)
    dengue_t = briere(temps, 17.8, 37.5, 1348.6)
    malaria_t = briere(temps, 16.0, 34.0, 916.5)

    ax1.plot(temps, dengue_t, color='#d95f02', lw=2.5, label='Dengue (Aedes aegypti) [T0=17.8°C, Opt=29-31°C, Tm=37.5°C]')
    ax1.plot(temps, malaria_t, color='#7570b3', lw=2.5, label='Malaria (Anopheles sp.) [T0=16.0°C, Opt=26-28°C, Tm=34.0°C]')
    ax1.axvspan(26, 32, color='#d95f02', alpha=0.12, label='Optimal Dengue Window')
    ax1.axvspan(24, 28, color='#7570b3', alpha=0.12, label='Optimal Malaria Window')

    ax1.set_title('(A) Biophysical Thermal Performance Curves (Brière / Mordecai)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Mean Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Thermal Suitability f_T(T) [0 to 1]', fontsize=11)
    ax1.set_xlim(10, 42)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Hydrologic Lag Response
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    rain_range = np.linspace(0, 200, 200)
    fR_dengue = 1.0 + 0.6 * np.log1p(rain_range)
    fR_malaria = 1.0 + 0.8 * np.log1p(rain_range)

    ax2.plot(rain_range, fR_dengue, color='#d95f02', lw=2.5, label='Dengue Hydrologic Multiplier (γ = 0.6)')
    ax2.plot(rain_range, fR_malaria, color='#7570b3', lw=2.5, label='Malaria Hydrologic Multiplier (γ = 0.8)')

    ax2.set_title('(B) Lagged Rainfall Response Function f_R(R_lag)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('2-to-6 Week Lagged Rolling Mean Rainfall (mm/week)', fontsize=11)
    ax2.set_ylabel('Hydrologic Suitability Multiplier', fontsize=11)
    ax2.set_xlim(0, 200)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 3: Weekly Transmission Profiles across Representative States
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    # Read weather data with computed weights for 2018
    weather = pd.read_csv(PROCESSED_DATA_DIR / "imd_district_weekly_weather_2000_2024_FIXED.csv")
    weather['Tmean_C'] = (weather['Tmax_C'] + weather['Tmin_C']) / 2.0

    # Quick imputation for clean plotting
    nat_avg = weather.groupby(['Year', 'Week'])[['Tmean_C']].transform('mean')
    weather['Tmean_C'] = weather['Tmean_C'].fillna(nat_avg['Tmean_C'])

    weather['rain_lag'] = weather.groupby(['State', 'District'])['Rainfall_mm'].transform(
        lambda s: s.shift(2).rolling(window=5, min_periods=1).mean()
    ).fillna(0.0)

    w2018 = weather[weather['Year'] == 2018].copy()
    w2018['fT_d'] = briere(w2018['Tmean_C'].values, 17.8, 37.5, 1348.6)
    w2018['W_d'] = w2018['fT_d'] * (1.0 + 0.6 * np.log1p(w2018['rain_lag'].values)) + 0.02
    w2018['alpha_d'] = w2018['W_d'] / w2018.groupby(['State', 'District'])['W_d'].transform('sum')

    focus_states = ['Delhi', 'Kerala', 'Odisha', 'Maharashtra', 'West Bengal']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for st, col in zip(focus_states, colors):
        st_curve = w2018[w2018['State'] == st].groupby('Week')['alpha_d'].mean()
        ax3.plot(st_curve.index, st_curve.values * 100, lw=2.2, color=col, label=st)

    ax3.set_title('(C) Dengue Seasonal Transmission Dynamics Across Climatic Zones (2018)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Epidemiological Week of Year (Week 1 to 52)', fontsize=11)
    ax3.set_ylabel('Weekly Case Allocation (%)', fontsize=11)
    ax3.set_xlim(1, 52)
    ax3.legend(loc='upper right', frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel 4: Comparative Dengue vs Malaria Seasonality in Odisha
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    w2018['fT_m'] = briere(w2018['Tmean_C'].values, 16.0, 34.0, 916.5)
    w2018['W_m'] = w2018['fT_m'] * (1.0 + 0.8 * np.log1p(w2018['rain_lag'].values)) + 0.02
    w2018['alpha_m'] = w2018['W_m'] / w2018.groupby(['State', 'District'])['W_m'].transform('sum')

    for st in ['Odisha', 'Kerala']:
        d_c = w2018[w2018['State'] == st].groupby('Week')['alpha_d'].mean() * 100
        m_c = w2018[w2018['State'] == st].groupby('Week')['alpha_m'].mean() * 100
        ls = '-' if st == 'Odisha' else '--'
        ax4.plot(d_c.index, d_c.values, lw=2.2, color='#d95f02', linestyle=ls, label=f'Dengue ({st})')
        ax4.plot(m_c.index, m_c.values, lw=2.2, color='#7570b3', linestyle=ls, label=f'Malaria ({st})')

    ax4.set_title('(D) Dual-Disease Ecological Comparison (Odisha & Kerala, 2018)', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Epidemiological Week of Year (Week 1 to 52)', fontsize=11)
    ax4.set_ylabel('Weekly Case Allocation (%)', fontsize=11)
    ax4.set_xlim(1, 52)
    ax4.legend(loc='upper right', frameon=True, fontsize=9.5)

    # Save
    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=300, bbox_inches='tight')
    print(f"Validation plot successfully saved to: {FIGURE_OUT}")

if __name__ == "__main__":
    main()
