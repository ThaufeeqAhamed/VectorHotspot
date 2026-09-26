#!/usr/bin/env python3
"""
Phase 7: Biophysical Temporal Disaggregation (Annual -> Weekly)
VectorHotspot Project

Converts annual H3 hexagon case totals (from Phase 5 & 6 spatial disaggregation)
into weekly hexagon case time-series (620,742 cells x 52 weeks) driven by
meteorological conditions (temperature, rainfall), while strictly preserving
annual case totals (sum(weeks) == annual_cases with 0.000000 deviation).

Mathematical Formulation:
  1. Temperature suitability f_T(T): Briët / Mordecai Brière thermal curve
     - Dengue (Aedes aegypti): T0=17.8C, Tm=37.5C, optimal ~29-32C
     - Malaria (Anopheles culicifacies/stephensi): T0=16.0C, Tm=34.0C, optimal ~26-28C
  2. Hydrological suitability f_R(R_lag): 2-to-6 week lagged cumulative/mean rainfall
     f_R(R_lag) = 1 + gamma * log(1 + R_lag)
  3. Combined weekly weight:
     W_{h, y, w} = f_T(T_{h, y, w}) * f_R(R_{lag, h, y, w}) + epsilon
  4. Mass-preserving allocation:
     C_{h, y, w} = C_{h, y} * (W_{h, y, w} / sum_{w'}(W_{h, y, w'}))
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import os
import time
import shutil
import gc

# --------------------------------------------------------------------------------
# Configurations & Paths
# --------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
H3_GRID_PATH = PROCESSED_DATA_DIR / "india_h3_grid_res7.csv"
WEATHER_PATH = PROCESSED_DATA_DIR / "imd_district_weekly_weather_2000_2024_FIXED.csv"
DENGUE_ANNUAL_PATH = PROCESSED_DATA_DIR / "dengue_hex_annual.csv"
MALARIA_ANNUAL_PATH = PROCESSED_DATA_DIR / "malaria_hex_annual.csv"

# Outputs
DENGUE_WEEKLY_OUT = PROCESSED_DATA_DIR / "dengue_hex_weekly.parquet"
MALARIA_WEEKLY_OUT = PROCESSED_DATA_DIR / "malaria_hex_weekly.parquet"

# Biophysical Parameters (Mordecai et al. Brière thermal response curves)
DENGUE_T0 = 17.8
DENGUE_TM = 37.5
DENGUE_C_NORM = 1348.6  # normalizes max value to 1.0
DENGUE_GAMMA = 0.6      # rainfall scaling sensitivity

MALARIA_T0 = 16.0
MALARIA_TM = 34.0
MALARIA_C_NORM = 916.5  # normalizes max value to 1.0
MALARIA_GAMMA = 0.8     # rainfall scaling sensitivity

SEASONAL_EPSILON = 0.02 # Non-zero baseline transmission floor


def briere_thermal_suitability(T, T0, Tm, norm_c):
    """
    Compute thermal suitability f_T(T) normalized to [0, 1].
    Formula: f_T(T) = (1 / norm_c) * T * (T - T0) * sqrt(max(0, Tm - T)) for T0 < T < Tm, else 0
    """
    val = np.where(
        (T > T0) & (T < Tm),
        T * (T - T0) * np.sqrt(np.maximum(0.0, Tm - T)),
        0.0
    )
    return val / norm_c


def prepare_weather_weights():
    """
    Load IMD district weather, impute isolated missing values, and calculate
    weekly allocation weights alpha_{d, y, w} for Dengue and Malaria.
    """
    print("Loading IMD weather dataset...")
    weather = pd.read_csv(WEATHER_PATH)

    # Standard sorting by location and time
    weather = weather.sort_values(['State', 'District', 'Year', 'Week']).reset_index(drop=True)

    # 1. Impute isolated missing temperature values
    weather['Tmean_C'] = (weather['Tmax_C'] + weather['Tmin_C']) / 2.0
    state_avg = weather.groupby(['State', 'Year', 'Week'])[['Tmax_C', 'Tmin_C', 'Tmean_C']].transform('mean')
    weather['Tmax_C'] = weather['Tmax_C'].fillna(state_avg['Tmax_C'])
    weather['Tmin_C'] = weather['Tmin_C'].fillna(state_avg['Tmin_C'])
    weather['Tmean_C'] = weather['Tmean_C'].fillna(state_avg['Tmean_C'])

    nat_avg = weather.groupby(['Year', 'Week'])[['Tmax_C', 'Tmin_C', 'Tmean_C']].transform('mean')
    weather['Tmax_C'] = weather['Tmax_C'].fillna(nat_avg['Tmax_C'])
    weather['Tmin_C'] = weather['Tmin_C'].fillna(nat_avg['Tmin_C'])
    weather['Tmean_C'] = weather['Tmean_C'].fillna(nat_avg['Tmean_C'])

    # 2. Compute 2-to-6 week lagged hydrology feature (R_{lag})
    print("Computing 2-to-6 week lagged rainfall hydrology...")
    # Shift by 2 weeks, rolling 5-week window average
    weather['rain_lag_2_6'] = weather.groupby(['State', 'District'])['Rainfall_mm'].transform(
        lambda s: s.shift(2).rolling(window=5, min_periods=1).mean()
    ).fillna(0.0)

    # 3. Compute Biophysical Suitability Weights
    print("Computing Brière thermal performance curves and hydrologic weights...")
    T = weather['Tmean_C'].values
    R = np.log1p(weather['rain_lag_2_6'].values)

    # Dengue
    fT_den = briere_thermal_suitability(T, DENGUE_T0, DENGUE_TM, DENGUE_C_NORM)
    weather['W_dengue'] = fT_den * (1.0 + DENGUE_GAMMA * R) + SEASONAL_EPSILON

    # Malaria
    fT_mal = briere_thermal_suitability(T, MALARIA_T0, MALARIA_TM, MALARIA_C_NORM)
    weather['W_malaria'] = fT_mal * (1.0 + MALARIA_GAMMA * R) + SEASONAL_EPSILON

    # 4. Compute mass-preserving normalized weekly fraction alpha_{d, y, w}
    weather['alpha_dengue'] = weather['W_dengue'] / weather.groupby(['State', 'District', 'Year'])['W_dengue'].transform('sum')
    weather['alpha_malaria'] = weather['W_malaria'] / weather.groupby(['State', 'District', 'Year'])['W_malaria'].transform('sum')

    keep_cols = [
        'State', 'District', 'Year', 'Week',
        'Tmax_C', 'Tmin_C', 'Tmean_C', 'Rainfall_mm',
        'alpha_dengue', 'alpha_malaria', 'W_dengue', 'W_malaria'
    ]
    return weather[keep_cols]


def process_disease_disaggregation(weather, annual_path, grid_df, output_path, disease_col="dengue_cases", alpha_col="alpha_dengue"):
    """
    Perform vectorized, exact mass-preserving temporal disaggregation.
    Saves partitioned Parquet datasets by year with ZSTD compression.
    """
    print(f"\n{'='*70}")
    print(f" [{disease_col.upper()}] Starting Temporal Disaggregation")
    print(f"{'='*70}")
    print(f"Loading annual cases from {annual_path.name}...")

    annual_df = pd.read_csv(annual_path)
    years = sorted(annual_df['year'].unique())
    print(f"Loaded {len(annual_df):,} annual rows spanning {len(years)} years: {years[0]} - {years[-1]}")

    # Ensure canonical state and district from official H3 grid
    if 'state' in annual_df.columns:
        annual_df.drop('state', axis=1, inplace=True)
    if 'district' in annual_df.columns:
        annual_df.drop('district', axis=1, inplace=True)

    print(f"Attaching canonical (state, district) mapping from H3 grid...")
    annual_df = annual_df.merge(grid_df[['h3_index', 'state', 'district']], on='h3_index', how='left')

    print(f"Target output directory: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    total_processed_hex = 0
    total_mass_in = 0.0
    total_mass_out = 0.0
    max_absolute_error_overall = 0.0

    suit_col = alpha_col.replace('alpha', 'W')

    for yr in years:
        start_t = time.time()
        print(f"  -> Processing Year {yr}...", end=" ", flush=True)

        # 1. Slice annual and weather data for current year
        ann_yr = annual_df[annual_df['year'] == yr].copy()
        wx_yr = weather[weather['Year'] == yr].copy()

        wx_yr = wx_yr[['State', 'District', 'Week', 'Tmax_C', 'Tmin_C', 'Tmean_C', 'Rainfall_mm', alpha_col, suit_col]]
        wx_yr.rename(columns={'State': 'state', 'District': 'district', 'Week': 'week', suit_col: 'suitability'}, inplace=True)

        # 2. Vectorized merge: expands 1 annual row to 52/53 weekly rows per hexagon
        weekly_df = ann_yr.merge(wx_yr, on=['state', 'district'], how='inner')

        # 3. Disaggregate annual cases to weekly: C_{h, y, w} = C_{h, y} * alpha_{d, y, w}
        # Keep cases as float64 for exact numerical precision
        weekly_df[disease_col] = (weekly_df[disease_col] * weekly_df[alpha_col]).astype(np.float64)
        weekly_df['population'] = weekly_df['population'].astype(np.float32)
        weekly_df['Tmax_C'] = weekly_df['Tmax_C'].astype(np.float32)
        weekly_df['Tmin_C'] = weekly_df['Tmin_C'].astype(np.float32)
        weekly_df['Tmean_C'] = weekly_df['Tmean_C'].astype(np.float32)
        weekly_df['Rainfall_mm'] = weekly_df['Rainfall_mm'].astype(np.float32)
        weekly_df['suitability'] = weekly_df['suitability'].astype(np.float32)
        weekly_df['week'] = weekly_df['week'].astype(np.int8)
        weekly_df['year'] = weekly_df['year'].astype(np.int16)

        # Select tight schema
        keep_weekly = [
            'h3_index', 'state', 'district', 'year', 'week',
            disease_col, 'population',
            'Tmax_C', 'Tmin_C', 'Tmean_C', 'Rainfall_mm', 'suitability'
        ]
        weekly_df = weekly_df[keep_weekly]

        # 4. Strict mass-preservation verification
        c_in = float(ann_yr[disease_col].sum())
        c_out = float(weekly_df[disease_col].sum())
        max_err = float(np.abs(c_in - c_out))
        if max_err > max_absolute_error_overall:
            max_absolute_error_overall = max_err

        total_mass_in += c_in
        total_mass_out += c_out
        total_processed_hex += len(weekly_df)

        assert max_err < 1e-4, f"CRITICAL: Mass discrepancy detected! Year {yr}: Input {c_in} vs Output {c_out} (Err: {max_err})"

        # 5. Write partitioned Parquet
        yr_dir = output_path / f"year={yr}"
        yr_dir.mkdir(exist_ok=True)

        table = pa.Table.from_pandas(weekly_df, preserve_index=False)
        pq.write_table(table, yr_dir / "data.parquet", compression='zstd')

        ela = time.time() - start_t
        print(f"done. {len(weekly_df):,} weekly rows (Max Error: {max_err:.8f}) in {ela:.1f}s")

        del weekly_df, ann_yr, wx_yr, table
        gc.collect()

    print(f"\n[{disease_col.upper()}] Disaggregation Summary:")
    print(f"  Total weekly records generated: {total_processed_hex:,}")
    print(f"  Total annual mass ingested:     {total_mass_in:,.4f}")
    print(f"  Total weekly mass output:       {total_mass_out:,.4f}")
    print(f"  Max Absolute Error Across Years:{max_absolute_error_overall:.8f}")
    print(f"  Mass delta:                     {np.abs(total_mass_in - total_mass_out):.8f}")
    print(f"  Saved in partitioned Parquet directory: {output_path}")


def main():
    print("=" * 70)
    print(" VECTORHOTSPOT - PHASE 7: BIOPHYSICAL TEMPORAL DISAGGREGATION")
    print("=" * 70)

    t0 = time.time()
    weather = prepare_weather_weights()

    print("\nLoading canonical H3 grid (india_h3_grid_res7.csv)...")
    grid = pd.read_csv(H3_GRID_PATH, usecols=['h3_index', 'state', 'district'])

    # Process Dengue (2010-2024, 13 years)
    process_disease_disaggregation(
        weather=weather,
        annual_path=DENGUE_ANNUAL_PATH,
        grid_df=grid,
        output_path=DENGUE_WEEKLY_OUT,
        disease_col="dengue_cases",
        alpha_col="alpha_dengue"
    )

    # Process Malaria (2000-2024, 25 years)
    process_disease_disaggregation(
        weather=weather,
        annual_path=MALARIA_ANNUAL_PATH,
        grid_df=grid,
        output_path=MALARIA_WEEKLY_OUT,
        disease_col="malaria_cases",
        alpha_col="alpha_malaria"
    )

    total_ela = (time.time() - t0) / 60
    print("\n" + "=" * 70)
    print(f" PHASE 7 EXECUTION COMPLETE: Total Run Time = {total_ela:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
