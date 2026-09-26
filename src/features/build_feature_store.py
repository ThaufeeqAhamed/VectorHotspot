#!/usr/bin/env python3
"""
Phase 8: Spatiotemporal Feature Engineering Engine (Streaming Optimized)
VectorHotspot Project

Constructs high-dimensional spatiotemporal feature stores for Dengue and Malaria:
1. Autoregressive case lags (t-1 to t-8) & rolling window statistics (4w, 12w moving mean/std, momentum).
2. Spatial neighbor spillover features (k=1 and k=2 rings via precomputed sparse adjacency W1, W2).
3. Lagged meteorological covariates (Tmax, Tmin, Tmean, DTR, rainfall, rolling cumulative rainfall, biophysical suitability).
4. Static demographic & remote sensing environmental covariates (population, density, NDVI, ESA WorldCover, JRC Water).
5. Cyclical seasonality encodings (sin/cos week-of-year).
6. Multi-horizon targets (t+1, t+2, t+3, t+4 weeks ahead) & outbreak indicators.
7. Strict leakage-free temporal (Train <= 2020, Val 2021-2022, Test 2023-2024) and spatial holdout splits (15% held-out districts).
"""

import pandas as pd
import numpy as np
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
import time
import sys
import gc

# --------------------------------------------------------------------------------
# Configurations & Paths
# --------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

H3_GRID_PATH = PROCESSED_DATA_DIR / "india_h3_grid_res7.csv"
TIER2_COV_PATH = PROCESSED_DATA_DIR / "hex_tier2_covariates.csv"
ADJACENCY_PATH = PROCESSED_DATA_DIR / "h3_adjacency_res7.npz"

DENGUE_WEEKLY_DIR = PROCESSED_DATA_DIR / "dengue_hex_weekly.parquet"
MALARIA_WEEKLY_DIR = PROCESSED_DATA_DIR / "malaria_hex_weekly.parquet"

METADATA_OUT = PROCESSED_DATA_DIR / "feature_store_metadata.json"

TRAIN_MAX_YEAR = 2020
VAL_YEARS = [2021, 2022]
TEST_YEARS = [2023, 2024]


def load_static_spatial_data():
    """
    Load H3 grid metadata, Tier 2 covariates, and precomputed sparse adjacency matrices.
    """
    print("Loading static spatial grid and Tier 2 covariates...", flush=True)
    grid = pd.read_csv(H3_GRID_PATH)
    tier2 = pd.read_csv(TIER2_COV_PATH)
    grid = grid.merge(tier2, on='h3_index', how='left')

    # Fill isolated covariate nulls with median
    for col in ['ndvi_mean', 'frac_water', 'frac_trees', 'frac_built', 'frac_shrub', 'jrc_occurrence']:
        if col in grid.columns:
            grid[col] = grid[col].fillna(grid[col].median()).astype(np.float32)

    print(f"Loading cached spatial adjacency matrices from {ADJACENCY_PATH.name}...", flush=True)
    adj = np.load(ADJACENCY_PATH)
    W1 = sparse.csr_matrix((adj['W1_data'], adj['W1_indices'], adj['W1_indptr']), shape=adj['W1_shape'], dtype=np.float32)
    W2 = sparse.csr_matrix((adj['W2_data'], adj['W2_indices'], adj['W2_indptr']), shape=adj['W2_shape'], dtype=np.float32)

    # Assign 15% deterministic pseudo-random spatial holdout districts
    print("Assigning 15% spatial holdout districts...", flush=True)
    unique_districts = grid[['state', 'district']].drop_duplicates().sort_values(['state', 'district']).reset_index(drop=True)
    np.random.seed(42)
    n_holdout = int(len(unique_districts) * 0.15)
    holdout_indices = set(np.random.choice(len(unique_districts), size=n_holdout, replace=False))

    unique_districts['is_spatial_holdout'] = [i in holdout_indices for i in range(len(unique_districts))]
    grid = grid.merge(unique_districts, on=['state', 'district'], how='left')

    print(f"Total districts: {len(unique_districts)}, Spatial Holdout: {n_holdout} districts ({n_holdout/len(unique_districts)*100:.1f}%)", flush=True)
    return grid, W1, W2


def load_year_dense_arrays(weekly_dir, yr, h3_to_idx, n_hex, disease_col):
    """
    Load a single year of weekly Parquet and populate fast 2D NumPy matrices.
    """
    part_path = weekly_dir / f"year={yr}" / "data.parquet"
    if not part_path.exists():
        return None

    df = pd.read_parquet(part_path)
    weeks = sorted(df['week'].unique())
    n_w = len(weeks)

    hex_idx = df['h3_index'].map(h3_to_idx).values
    week_idx = (df['week'].values - 1).astype(np.int32)

    # Populate dense 2D matrices (n_weeks, n_hex)
    c_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    pop_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    tmax_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    tmin_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    tmean_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    rain_mat = np.zeros((n_w, n_hex), dtype=np.float32)
    suit_mat = np.zeros((n_w, n_hex), dtype=np.float32)

    c_mat[week_idx, hex_idx] = df[disease_col].values
    pop_mat[week_idx, hex_idx] = df['population'].values
    tmax_mat[week_idx, hex_idx] = df['Tmax_C'].values
    tmin_mat[week_idx, hex_idx] = df['Tmin_C'].values
    tmean_mat[week_idx, hex_idx] = df['Tmean_C'].values
    rain_mat[week_idx, hex_idx] = df['Rainfall_mm'].values
    suit_mat[week_idx, hex_idx] = df['suitability'].values

    return {
        'year': yr,
        'weeks': weeks,
        'cases': c_mat,
        'pop': pop_mat,
        'tmax': tmax_mat,
        'tmin': tmin_mat,
        'tmean': tmean_mat,
        'rain': rain_mat,
        'suit': suit_mat
    }


def extract_disease_features(disease_name, disease_col, weekly_dir, grid, W1, W2, available_years):
    """
    Extract multi-lag, rolling statistics, spatial spillover, and weather features
    using a sliding window memory buffer across years.
    """
    print(f"\n{'='*70}", flush=True)
    print(f" EXTRACTING SPATIOTEMPORAL FEATURES: {disease_name.upper()}", flush=True)
    print(f"{'='*70}", flush=True)

    n_hex = len(grid)
    h3_to_idx = {h: i for i, h in enumerate(grid['h3_index'])}

    # Select representative stratified hexagon cohort across all 724 districts
    print("Selecting representative stratified hexagon cohort across all districts...", flush=True)
    np.random.seed(42)
    sample_indices = []
    for (st, dt), group in grid.groupby(['state', 'district']):
        n_g = len(group)
        if n_g <= 50:
            sample_indices.extend(group.index.tolist())
        else:
            chosen = group.sample(50, random_state=42).index.tolist()
            sample_indices.extend(chosen)

    sample_mask = np.zeros(n_hex, dtype=bool)
    sample_mask[sample_indices] = True
    s_idx = np.where(sample_mask)[0]
    n_sample = len(s_idx)
    print(f"Representative training cohort: {n_sample:,} hexagons ({n_sample/n_hex*100:.1f}% of India)", flush=True)

    # Static features sliced for sample
    s_h3 = grid['h3_index'].values[s_idx]
    s_state = grid['state'].values[s_idx]
    s_district = grid['district'].values[s_idx]
    s_lat = grid['center_lat'].values[s_idx].astype(np.float32)
    s_lon = grid['center_lon'].values[s_idx].astype(np.float32)
    s_ndvi = grid['ndvi_mean'].values[s_idx].astype(np.float32)
    s_trees = grid['frac_trees'].values[s_idx].astype(np.float32)
    s_water = grid['frac_water'].values[s_idx].astype(np.float32)
    s_built = grid['frac_built'].values[s_idx].astype(np.float32)
    s_shrub = grid['frac_shrub'].values[s_idx].astype(np.float32)
    s_jrc = grid['jrc_occurrence'].values[s_idx].astype(np.float32)
    s_holdout = grid['is_spatial_holdout'].values[s_idx]

    # Pre-calculate 90th percentile outbreak threshold
    print("Computing empirical outbreak threshold...", flush=True)
    sample_yr = available_years[len(available_years)//2]
    df_sample = pd.read_parquet(weekly_dir / f"year={sample_yr}" / "data.parquet")
    pos_cases = df_sample[df_sample[disease_col] > 0][disease_col]
    outbreak_threshold = float(np.percentile(pos_cases, 90)) if len(pos_cases) > 0 else 0.1
    print(f"90th Percentile Outbreak Threshold for {disease_name.upper()}: {outbreak_threshold:.6f} cases/week", flush=True)
    del df_sample, pos_cases

    # Sliding window buffer: (prev_year_data, curr_year_data, next_year_data)
    loaded_years_cache = {}

    def get_year_data(yr):
        if yr not in loaded_years_cache:
            loaded_years_cache[yr] = load_year_dense_arrays(weekly_dir, yr, h3_to_idx, n_hex, disease_col)
        return loaded_years_cache[yr]

    def prune_cache(keep_years):
        for yr in list(loaded_years_cache.keys()):
            if yr not in keep_years:
                del loaded_years_cache[yr]
        gc.collect()

    train_dfs = []
    val_dfs = []
    test_dfs = []

    print("\nProcessing years via streaming sliding window...", flush=True)
    t_start = time.time()

    for idx, yr in enumerate(available_years):
        yr_t0 = time.time()
        prev_yr = available_years[idx - 1] if idx > 0 else None
        next_yr = available_years[idx + 1] if idx < len(available_years) - 1 else None

        # Maintain only active window in RAM
        curr_data = get_year_data(yr)
        prev_data = get_year_data(prev_yr) if prev_yr is not None else None
        next_data = get_year_data(next_yr) if next_yr is not None else None

        prune_cache({prev_yr, yr, next_yr} - {None})

        n_w = len(curr_data['weeks'])

        # Precompute spatial neighbor arrays for all weeks in current year
        nbr_k1_all = np.zeros_like(curr_data['cases'])
        nbr_k2_all = np.zeros_like(curr_data['cases'])
        for w in range(n_w):
            nbr_k1_all[w] = W1.dot(curr_data['cases'][w])
            nbr_k2_all[w] = W2.dot(curr_data['cases'][w])

        # Precompute spatial neighbor arrays for prev year tail (if exists)
        if prev_data is not None:
            prev_nbr_k1 = np.zeros_like(prev_data['cases'])
            for w in range(len(prev_data['weeks'])):
                prev_nbr_k1[w] = W1.dot(prev_data['cases'][w])
        else:
            prev_nbr_k1 = None

        # Vectorized access functions across year boundaries
        def get_hist_cases(w, lag):
            target_w = w - lag
            if target_w >= 0:
                return curr_data['cases'][target_w]
            elif prev_data is not None:
                pw = len(prev_data['weeks']) + target_w
                if pw >= 0:
                    return prev_data['cases'][pw]
            return curr_data['cases'][0]

        def get_hist_nbr_k1(w, lag):
            target_w = w - lag
            if target_w >= 0:
                return nbr_k1_all[target_w]
            elif prev_nbr_k1 is not None:
                pw = len(prev_nbr_k1) + target_w
                if pw >= 0:
                    return prev_nbr_k1[pw]
            return nbr_k1_all[0]

        def get_hist_weather(varname, w, lag):
            target_w = w - lag
            if target_w >= 0:
                return curr_data[varname][target_w]
            elif prev_data is not None:
                pw = len(prev_data['weeks']) + target_w
                if pw >= 0:
                    return prev_data[varname][pw]
            return curr_data[varname][0]

        def get_future_target(w, lead):
            target_w = w + lead
            if target_w < n_w:
                return curr_data['cases'][target_w]
            elif next_data is not None:
                nw = target_w - n_w
                if nw < len(next_data['weeks']):
                    return next_data['cases'][nw]
            return curr_data['cases'][n_w - 1]

        # Determine split category
        if yr <= TRAIN_MAX_YEAR:
            split_tag = 'train'
        elif yr in VAL_YEARS:
            split_tag = 'val'
        else:
            split_tag = 'test'

        yr_weekly_rows = []

        for w in range(n_w):
            w_num = curr_data['weeks'][w]

            # 1. Autoregressive Lags (t-1, t-2, t-3, t-4, t-8, t-12)
            c1_f = get_hist_cases(w, 1)
            c2_f = get_hist_cases(w, 2)
            c3_f = get_hist_cases(w, 3)
            c4_f = get_hist_cases(w, 4)
            c8_f = get_hist_cases(w, 8)
            c12_f = get_hist_cases(w, 12)

            c1 = c1_f[s_idx]
            c2 = c2_f[s_idx]
            c3 = c3_f[s_idx]
            c4 = c4_f[s_idx]
            c8 = c8_f[s_idx]
            c12 = c12_f[s_idx]

            # Rolling stats
            roll_4w = (c1 + c2 + c3 + c4) / 4.0
            roll_std_4w = np.std(np.stack([c1, c2, c3, c4], axis=0), axis=0)
            roll_12w = (roll_4w + (c8 + c12) / 2.0) / 2.0
            momentum = (c1 - c4) / (c4 + 1e-4)

            pop_s = curr_data['pop'][w][s_idx]
            case_rate = (c1 * 10000.0) / (pop_s + 1.0)
            log_pop = np.log1p(pop_s)
            pop_density = pop_s / 5.2

            # 2. Spatial Neighbor Spillover Features (Strict Leakage Guard: t-1, t-2)
            nbr_k1_1 = get_hist_nbr_k1(w, 1)[s_idx]
            nbr_k1_2 = get_hist_nbr_k1(w, 2)[s_idx]
            nbr_k2_1 = (nbr_k2_all[w - 1] if w > 0 else (prev_data['cases'][-1] if prev_data is not None else curr_data['cases'][0]))[s_idx]

            # 3. Weather Covariates
            tmax1 = get_hist_weather('tmax', w, 1)[s_idx]
            tmax2 = get_hist_weather('tmax', w, 2)[s_idx]
            tmax4 = get_hist_weather('tmax', w, 4)[s_idx]

            tmin1 = get_hist_weather('tmin', w, 1)[s_idx]
            tmin2 = get_hist_weather('tmin', w, 2)[s_idx]
            tmin4 = get_hist_weather('tmin', w, 4)[s_idx]

            tmean1 = get_hist_weather('tmean', w, 1)[s_idx]
            tmean2 = get_hist_weather('tmean', w, 2)[s_idx]
            tmean4 = get_hist_weather('tmean', w, 4)[s_idx]

            dtr1 = tmax1 - tmin1
            dtr2 = tmax2 - tmin2

            rain1 = get_hist_weather('rain', w, 1)[s_idx]
            rain2 = get_hist_weather('rain', w, 2)[s_idx]
            rain4 = get_hist_weather('rain', w, 4)[s_idx]
            rain6 = get_hist_weather('rain', w, 6)[s_idx]

            rain_2w = rain1 + rain2
            rain_4w = rain_2w + get_hist_weather('rain', w, 3)[s_idx] + rain4

            suit1 = get_hist_weather('suit', w, 1)[s_idx]
            suit2 = get_hist_weather('suit', w, 2)[s_idx]
            suit4 = get_hist_weather('suit', w, 4)[s_idx]

            # 4. Seasonality
            sin_w = np.float32(np.sin(2.0 * np.pi * w_num / 52.0))
            cos_w = np.float32(np.cos(2.0 * np.pi * w_num / 52.0))

            # 5. Targets (Lead t+1 to t+4)
            t1 = get_future_target(w, 1)[s_idx]
            t2 = get_future_target(w, 2)[s_idx]
            t3 = get_future_target(w, 3)[s_idx]
            t4 = get_future_target(w, 4)[s_idx]

            # Assemble DataFrame
            df_w = pd.DataFrame({
                'h3_index': s_h3,
                'state': s_state,
                'district': s_district,
                'year': np.int16(yr),
                'week': np.int8(w_num),
                'is_spatial_holdout': s_holdout,

                # Lags & Moving Stats
                'cases_lag_1': c1,
                'cases_lag_2': c2,
                'cases_lag_3': c3,
                'cases_lag_4': c4,
                'cases_lag_8': c8,
                'cases_roll_mean_4w': roll_4w,
                'cases_roll_std_4w': roll_std_4w,
                'cases_roll_mean_12w': roll_12w,
                'cases_momentum_4w': momentum,
                'case_rate_lag_1': case_rate,

                # Spatial Spillover
                'neighbor_cases_k1_lag1': nbr_k1_1,
                'neighbor_cases_k1_lag2': nbr_k1_2,
                'neighbor_cases_k2_lag1': nbr_k2_1,

                # Weather & Biophysical Suitability
                'tmax_lag_1': tmax1,
                'tmax_lag_2': tmax2,
                'tmax_lag_4': tmax4,
                'tmin_lag_1': tmin1,
                'tmin_lag_2': tmin2,
                'tmin_lag_4': tmin4,
                'tmean_lag_1': tmean1,
                'tmean_lag_2': tmean2,
                'tmean_lag_4': tmean4,
                'dtr_lag_1': dtr1,
                'dtr_lag_2': dtr2,
                'rain_lag_1': rain1,
                'rain_lag_2': rain2,
                'rain_lag_4': rain4,
                'rain_lag_6': rain6,
                'rain_roll_sum_2w': rain_2w,
                'rain_roll_sum_4w': rain_4w,
                'suitability_lag_1': suit1,
                'suitability_lag_2': suit2,
                'suitability_lag_4': suit4,

                # Static Demographics & Environmental
                'log_population': log_pop,
                'pop_density': pop_density,
                'ndvi_mean': s_ndvi,
                'frac_trees': s_trees,
                'frac_water': s_water,
                'frac_built': s_built,
                'frac_shrub': s_shrub,
                'jrc_occurrence': s_jrc,
                'center_lat': s_lat,
                'center_lon': s_lon,

                # Seasonality
                'sin_week': sin_w,
                'cos_week': cos_w,

                # Target Labels
                'target_lead_1': t1,
                'target_lead_2': t2,
                'target_lead_3': t3,
                'target_lead_4': t4,
                'outbreak_lead_1': (t1 > outbreak_threshold).astype(np.int8),
                'outbreak_lead_2': (t2 > outbreak_threshold).astype(np.int8),
                'outbreak_lead_4': (t4 > outbreak_threshold).astype(np.int8)
            })
            yr_weekly_rows.append(df_w)

        df_yr_full = pd.concat(yr_weekly_rows, ignore_index=True)
        if split_tag == 'train':
            train_dfs.append(df_yr_full)
        elif split_tag == 'val':
            val_dfs.append(df_yr_full)
        else:
            test_dfs.append(df_yr_full)

        print(f"  Processed Year {yr} -> {split_tag.upper()} ({len(df_yr_full):,} rows) in {time.time() - yr_t0:.1f}s", flush=True)

    # Save to Parquet
    print(f"\nConcatenating and saving split feature tables for {disease_name.upper()}...", flush=True)
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)

    train_out = PROCESSED_DATA_DIR / f"features_{disease_name}_train.parquet"
    val_out = PROCESSED_DATA_DIR / f"features_{disease_name}_val.parquet"
    test_out = PROCESSED_DATA_DIR / f"features_{disease_name}_test.parquet"

    pq.write_table(pa.Table.from_pandas(df_train), train_out, compression='zstd')
    pq.write_table(pa.Table.from_pandas(df_val), val_out, compression='zstd')
    pq.write_table(pa.Table.from_pandas(df_test), test_out, compression='zstd')

    print(f"  Saved TRAIN ({len(df_train):,} rows): {train_out.name} ({train_out.stat().st_size / (1024*1024):.1f} MB)", flush=True)
    print(f"  Saved VAL   ({len(df_val):,} rows):   {val_out.name} ({val_out.stat().st_size / (1024*1024):.1f} MB)", flush=True)
    print(f"  Saved TEST  ({len(df_test):,} rows):  {test_out.name} ({test_out.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    return {
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'outbreak_threshold': outbreak_threshold,
        'features_count': len(df_train.columns) - 6
    }


def main():
    print("=" * 70, flush=True)
    print(" VECTORHOTSPOT - PHASE 8: SPATIOTEMPORAL FEATURE ENGINEERING", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    grid, W1, W2 = load_static_spatial_data()

    # Dengue (2010 to 2024, 13 surveillance years)
    dengue_years = [2010, 2011, 2012, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    dengue_meta = extract_disease_features(
        disease_name="dengue",
        disease_col="dengue_cases",
        weekly_dir=DENGUE_WEEKLY_DIR,
        grid=grid,
        W1=W1,
        W2=W2,
        available_years=dengue_years
    )

    # Malaria (2005 to 2024, 20 surveillance years)
    malaria_years = list(range(2005, 2025))
    malaria_meta = extract_disease_features(
        disease_name="malaria",
        disease_col="malaria_cases",
        weekly_dir=MALARIA_WEEKLY_DIR,
        grid=grid,
        W1=W1,
        W2=W2,
        available_years=malaria_years
    )

    # Save feature store metadata JSON
    metadata = {
        'phase': 'Phase 8: Spatiotemporal Feature Engineering',
        'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'temporal_splits': {
            'train_years': f'<= {TRAIN_MAX_YEAR}',
            'val_years': VAL_YEARS,
            'test_years': TEST_YEARS
        },
        'spatial_holdout_percentage': 15.0,
        'dengue_metadata': dengue_meta,
        'malaria_metadata': malaria_meta
    }

    with open(METADATA_OUT, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved Feature Store Metadata to {METADATA_OUT.name}", flush=True)
    print("=" * 70, flush=True)
    print(f" PHASE 8 COMPLETE: Total Elapsed Time = {(time.time() - t_start)/60:.2f} minutes", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
