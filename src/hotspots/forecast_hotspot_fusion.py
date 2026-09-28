#!/usr/bin/env python3
"""
Phase 10: Forecast-to-Hotspot Fusion Engine (Getis-Ord Gi*)
VectorHotspot Project

Applies local spatial autocorrelation (Getis-Ord Gi*) directly on PREDICTED
future risk surfaces (t+1, t+2, t+3, t+4 weeks ahead) to detect and classify
statistically significant disease outbreak hotspots before they manifest.

Mathematical Formulation:
  G_i^*(k) = [ sum_j w_ij * y_j - X_bar * sum_j w_ij ] / [ S * sqrt((N*sum_j w_ij^2 - (sum_j w_ij)^2) / (N-1)) ]
  where:
    w_ij is binary spatial adjacency with self-loops (w_ii = 1, w_ij = 1 if j in k-disk(i, 1))
    X_bar is global spatial mean predicted cases at horizon k
    S is global spatial standard deviation of predicted cases at horizon k

Hotspot Taxonomy:
  - Emerging Hotspot: G_i^*(t+k) >= 1.96 (p < 0.05), while baseline G_i^*(t) < 1.96
  - Intensifying Hotspot: G_i^*(t+1) < G_i^*(t+2) < G_i^*(t+3) < G_i^*(t+4) and G_i^*(t+4) >= 1.96
  - Persistent Hotspot: G_i^* >= 2.58 (p < 0.01) across all horizons t+1 to t+4
  - Diminishing Hotspot: G_i^*(t) >= 1.96, but drops below 1.96 at horizon t+4
"""

import pandas as pd
import numpy as np
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
import h3
from pathlib import Path
import time
import os

# Thread capping to prevent CPU overload and ensure smooth multitasking
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_HOTSPOTS_DIR = PROJECT_ROOT / "outputs" / "hotspots"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUTS_HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 4]


def build_spatial_weight_matrix(unique_hexes):
    """
    Build binary spatial adjacency matrix with self-loops (W_star)
    for the test hexagon cohort using H3 k-disk(1).
    """
    n = len(unique_hexes)
    h3_to_idx = {h: i for i, h in enumerate(unique_hexes)}

    rows, cols, vals = [], [], []
    for i, h in enumerate(unique_hexes):
        # Disk of radius 1 includes hexagon itself + up to 6 neighbors
        disk1 = [h3_to_idx[nbr] for nbr in h3.grid_disk(h, 1) if nbr in h3_to_idx]
        for j in disk1:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)

    W_star = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    return W_star, h3_to_idx


def compute_getis_ord_gi(W_star, spatial_values_mat):
    """
    Vectorized computation of Getis-Ord Gi* z-scores across all time slices.
    spatial_values_mat: 2D array of shape (N_hexagons, N_time_slices)
    Returns: 2D array of Gi* z-scores of shape (N_hexagons, N_time_slices)
    """
    n = W_star.shape[0]

    # W_i = sum_j w_ij (shape: n)
    W_i = np.array(W_star.sum(axis=1), dtype=np.float64).flatten()
    # W2_i = sum_j w_ij^2 (shape: n)
    W2_i = np.array(W_star.multiply(W_star).sum(axis=1), dtype=np.float64).flatten()

    # Precompute denominator variance scale factor
    term = np.sqrt(np.maximum(1e-9, (n * W2_i - W_i**2) / (n - 1.0)))

    # Global spatial mean and sample standard deviation per time slice
    X_bar = np.mean(spatial_values_mat, axis=0, dtype=np.float64)  # (N_time,)
    S = np.std(spatial_values_mat, axis=0, ddof=1, dtype=np.float64)  # (N_time,)

    # Numerator = W * X - X_bar * W_i
    W_X = W_star.dot(spatial_values_mat)  # (n, N_time)
    numerator = W_X - np.outer(W_i, X_bar)
    denominator = np.outer(term, S)

    # Compute z-scores with guard against zero variance
    gi_zscores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9)
    return gi_zscores.astype(np.float32)


def process_disease_hotspots(disease_name):
    print(f"\n{'='*75}")
    print(f" RUNNING FORECAST-TO-HOTSPOT FUSION: {disease_name.upper()} (2023-2024 TEST SET)")
    print(f"{'='*75}", flush=True)

    test_path = PROCESSED_DATA_DIR / f"features_{disease_name}_test.parquet"
    print(f"Loading {disease_name.capitalize()} test feature dataset from {test_path.name}...", flush=True)
    df_test = pd.read_parquet(test_path)
    print(f"Loaded {len(df_test):,} test set rows across 104 weeks (2023–2024).", flush=True)

    meta_cols = {
        'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
        'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4',
        'outbreak_lead_1', 'outbreak_lead_2', 'outbreak_lead_4'
    }
    feature_cols = [c for c in df_test.columns if c not in meta_cols]

    # Load trained LightGBM models for all horizons
    print("Loading trained multi-horizon LightGBM models (t+1 ... t+4)...", flush=True)
    models = {}
    for k in HORIZONS:
        m_path = MODELS_DIR / f"lgbm_{disease_name}_lead{k}.joblib"
        m = joblib.load(m_path)
        m.set_params(n_jobs=4)
        models[k] = m

    # 1. Multi-Horizon Batch Predictions
    print("Executing batch multi-horizon model predictions...", flush=True)
    t0 = time.time()
    X_test_mat = df_test[feature_cols].values
    base_cases = df_test['cases_lag_1'].values

    for k in HORIZONS:
        t_k = time.time()
        delta_pred = models[k].predict(X_test_mat)
        df_test[f'pred_lead_{k}'] = np.maximum(0.0, base_cases + delta_pred).astype(np.float32)
        print(f"  Predicted horizon t+{k} ({len(df_test):,} samples) in {time.time()-t_k:.1f}s", flush=True)

    print(f"Total prediction inference time: {time.time()-t0:.1f}s", flush=True)

    # 2. Construct Spatial Adjacency Matrix
    print("Constructing spatial adjacency matrix (W_star) with H3 k-disk(1)...", flush=True)
    unique_hexes = df_test['h3_index'].drop_duplicates().values
    n_hex = len(unique_hexes)
    W_star, h3_to_idx = build_spatial_weight_matrix(unique_hexes)
    print(f"  W_star built for {n_hex:,} hexagons: {W_star.nnz:,} edges ({W_star.nnz/n_hex:.2f} avg neighbors/hex).", flush=True)

    # 3. Compute Getis-Ord Gi* on Predictions, Baseline, and Ground Truth
    print("Computing Getis-Ord Gi* local spatial autocorrelation statistics...", flush=True)
    t_gi = time.time()

    # Sort df_test by (year, week, h3_index) to align with matrix structure
    df_test['hex_idx'] = df_test['h3_index'].map(h3_to_idx)
    df_test = df_test.sort_values(['year', 'week', 'hex_idx']).reset_index(drop=True)

    # Reshape each variable to 2D matrix: (n_hex, n_time_slices)
    time_slices = df_test[['year', 'week']].drop_duplicates().values
    n_time = len(time_slices)
    print(f"  Total time slices: {n_time} weeks (2023–2024)", flush=True)

    # Current baseline Gi* (from historical cases at t)
    curr_cases_mat = df_test['cases_lag_1'].values.reshape(n_time, n_hex).T
    gi_curr_mat = compute_getis_ord_gi(W_star, curr_cases_mat)
    df_test['gi_zscore_current'] = gi_curr_mat.T.flatten()

    # Multi-horizon predicted and actual Gi*
    for k in HORIZONS:
        # Gi* on PREDICTED risk surface
        pred_mat_k = df_test[f'pred_lead_{k}'].values.reshape(n_time, n_hex).T
        gi_pred_k = compute_getis_ord_gi(W_star, pred_mat_k)
        df_test[f'gi_zscore_pred_lead_{k}'] = gi_pred_k.T.flatten()
        df_test[f'is_hotspot_pred_lead_{k}'] = (df_test[f'gi_zscore_pred_lead_{k}'] >= 1.96).astype(np.int8)

        # Gi* on OBSERVED actual cases (Ground Truth)
        act_mat_k = df_test[f'target_lead_{k}'].values.reshape(n_time, n_hex).T
        gi_act_k = compute_getis_ord_gi(W_star, act_mat_k)
        df_test[f'gi_zscore_act_lead_{k}'] = gi_act_k.T.flatten()
        df_test[f'is_hotspot_act_lead_{k}'] = (df_test[f'gi_zscore_act_lead_{k}'] >= 1.96).astype(np.int8)

    print(f"  Computed all Gi* statistics in {time.time()-t_gi:.2f}s", flush=True)

    # 4. Classify 4-Tier Hotspot Taxonomy
    print("Classifying Hotspot Taxonomy (Emerging, Intensifying, Persistent, Diminishing)...", flush=True)
    z_curr = df_test['gi_zscore_current'].values
    z1 = df_test['gi_zscore_pred_lead_1'].values
    z2 = df_test['gi_zscore_pred_lead_2'].values
    z3 = df_test['gi_zscore_pred_lead_3'].values
    z4 = df_test['gi_zscore_pred_lead_4'].values

    # Taxonomy conditions
    cond_persistent = (z1 >= 2.58) & (z2 >= 2.58) & (z3 >= 2.58) & (z4 >= 2.58)
    cond_emerging = (z_curr < 1.96) & (z4 >= 1.96)
    cond_intensifying = (z4 > z3) & (z3 > z2) & (z2 > z1) & (z4 >= 1.96)
    cond_diminishing = (z_curr >= 1.96) & (z4 < 1.96)

    taxonomy = np.full(len(df_test), "Non-Hotspot", dtype=object)
    taxonomy[cond_diminishing] = "Diminishing Hotspot"
    taxonomy[cond_emerging] = "Emerging Hotspot"
    taxonomy[cond_intensifying] = "Intensifying Hotspot"
    taxonomy[cond_persistent] = "Persistent Hotspot"  # Overriding priority

    df_test['hotspot_taxonomy'] = taxonomy

    # 5. Calculate Spatial Verification & Accuracy Metrics
    print("\nCalculating Spatial Hotspot Detection Metrics (IoU, Precision, Recall, F1)...", flush=True)
    summary_metrics = []
    for k in HORIZONS:
        pred_h = df_test[f'is_hotspot_pred_lead_{k}'].values
        act_h = df_test[f'is_hotspot_act_lead_{k}'].values

        intersection = np.sum((pred_h == 1) & (act_h == 1))
        union = np.sum((pred_h == 1) | (act_h == 1))
        pred_pos = np.sum(pred_h == 1)
        act_pos = np.sum(act_h == 1)

        iou = float(intersection / union) if union > 0 else 0.0
        precision = float(intersection / pred_pos) if pred_pos > 0 else 0.0
        recall = float(intersection / act_pos) if act_pos > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        summary_metrics.append({
            'disease': disease_name,
            'horizon': f't+{k}',
            'spatial_iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predicted_hotspots': int(pred_pos),
            'actual_hotspots': int(act_pos)
        })

        print(f"  Horizon t+{k}: Spatial IoU = {iou:.4f} | Precision = {precision:.4f} | Recall = {recall:.4f} | F1 = {f1:.4f}")

    # 6. Save Hotspot Parquet Tables
    out_parquet = OUTPUTS_HOTSPOTS_DIR / f"{disease_name}_hotspots_test_2023_2024.parquet"
    keep_cols = [
        'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
        'cases_lag_1', 'gi_zscore_current',
        'pred_lead_1', 'gi_zscore_pred_lead_1', 'is_hotspot_pred_lead_1', 'target_lead_1', 'is_hotspot_act_lead_1',
        'pred_lead_2', 'gi_zscore_pred_lead_2', 'is_hotspot_pred_lead_2', 'target_lead_2', 'is_hotspot_act_lead_2',
        'pred_lead_3', 'gi_zscore_pred_lead_3', 'is_hotspot_pred_lead_3', 'target_lead_3', 'is_hotspot_act_lead_3',
        'pred_lead_4', 'gi_zscore_pred_lead_4', 'is_hotspot_pred_lead_4', 'target_lead_4', 'is_hotspot_act_lead_4',
        'hotspot_taxonomy'
    ]

    print(f"\nSaving consolidated hotspot dataset to {out_parquet.name}...", flush=True)
    pq.write_table(pa.Table.from_pandas(df_test[keep_cols]), out_parquet, compression='zstd')
    print(f"  Saved {out_parquet.name} ({out_parquet.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    print("\nTaxonomy Summary:")
    print(df_test['hotspot_taxonomy'].value_counts())

    return summary_metrics


def main():
    print("=" * 75)
    print(" VECTORHOTSPOT - PHASE 10: FORECAST-TO-HOTSPOT FUSION ENGINE")
    print("=" * 75, flush=True)

    t_start = time.time()

    dengue_metrics = process_disease_hotspots("dengue")
    malaria_metrics = process_disease_hotspots("malaria")

    # Save summary table
    df_metrics = pd.DataFrame(dengue_metrics + malaria_metrics)
    csv_out = OUTPUTS_TABLES_DIR / "hotspot_fusion_evaluation_metrics.csv"
    df_metrics.to_csv(csv_out, index=False)
    print(f"\nSaved hotspot evaluation metrics to {csv_out.name}")

    print("\n" + "=" * 75)
    print(" SUMMARY SPATIAL HOTSPOT EVALUATION (2023-2024 TEST SET)")
    print("=" * 75)
    print(df_metrics.to_string(index=False))

    print("\n" + "=" * 75)
    print(f" PHASE 10 COMPLETE: Total Elapsed Time = {(time.time() - t_start)/60:.2f} minutes")
    print("=" * 75)


if __name__ == "__main__":
    main()
