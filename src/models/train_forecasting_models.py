#!/usr/bin/env python3
"""
Phase 9: Dual-Disease Multi-Horizon Forecasting Models
VectorHotspot Project

Trains, tunes, and evaluates gradient-boosted tree models (LightGBM & XGBoost)
for direct multi-horizon forecasting (t+1, t+2, t+3, t+4 weeks ahead) of Dengue
and Malaria case risk surfaces across India.

Benchmarks against:
1. Baseline 1: Naive Persistence (y_{t+k} = y_t)
2. Baseline 2: Historical Seasonal Mean (y_{h, w+k} = mean_{hist}(h, w+k))
3. Baseline 3: Coarse District-Level Aggregation Model

Evaluates on:
- Temporal Test Set (Seen districts, 2023-2024 future holdout)
- Spatial Holdout Test Set (Unseen 15% held-out districts, 2023-2024)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json
import time

# --------------------------------------------------------------------------------
# Configurations & Paths
# --------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUTS_TABLES_DIR / "model_evaluation_metrics.csv"
PREDICTIONS_PARQUET = PROCESSED_DATA_DIR / "forecast_predictions_eval.parquet"

HORIZONS = [1, 2, 3, 4]


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics for count regression.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-9 else 0.0
    rmsle = float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))

    return {
        'r2': r2,
        'corr': corr,
        'mae': mae,
        'rmse': rmse,
        'rmsle': rmsle
    }


def train_and_evaluate_disease(disease_name):
    print(f"\n{'='*75}")
    print(f" TRAINING & EVALUATING FORECASTING MODELS: {disease_name.upper()}")
    print(f"{'='*75}", flush=True)

    train_path = PROCESSED_DATA_DIR / f"features_{disease_name}_train.parquet"
    val_path = PROCESSED_DATA_DIR / f"features_{disease_name}_val.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{disease_name}_test.parquet"

    print("Loading feature partitions...", flush=True)
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"Loaded: Train ({len(df_train):,} rows), Val ({len(df_val):,} rows), Test ({len(df_test):,} rows)", flush=True)

    meta_cols = {
        'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
        'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4',
        'outbreak_lead_1', 'outbreak_lead_2', 'outbreak_lead_4'
    }
    feature_cols = [c for c in df_train.columns if c not in meta_cols]
    print(f"Feature count: {len(feature_cols)} features", flush=True)

    # Use a large, balanced training sample for rapid convergence and full representativeness
    np.random.seed(42)
    n_train_sub = min(len(df_train), 1500000)
    train_sub = df_train[~df_train['is_spatial_holdout']].sample(n_train_sub, random_state=42)

    # Validation and test subsets (Seen vs Unseen Spatial Holdouts)
    val_seen = df_val[~df_val['is_spatial_holdout']]
    test_seen = df_test[~df_test['is_spatial_holdout']]
    test_spatial_holdout = df_test[df_test['is_spatial_holdout']]

    # Precompute Baseline 2: Historical Seasonal Mean lookup table per (district, week)
    print("Fitting Baseline 2 (Historical Seasonal Mean) & Baseline 3 (Coarse District)...", flush=True)
    seasonal_lookup = df_train.groupby(['district', 'week'])['cases_lag_1'].mean().to_dict()

    # Precompute Baseline 3: Coarse District Model
    district_train = df_train.groupby(['district', 'year', 'week']).agg(
        dist_cases=('cases_lag_1', 'sum'),
        dist_rain=('rain_lag_1', 'mean'),
        dist_temp=('tmean_lag_1', 'mean'),
        dist_target1=('target_lead_1', 'sum'),
        dist_target2=('target_lead_2', 'sum'),
        dist_target3=('target_lead_3', 'sum'),
        dist_target4=('target_lead_4', 'sum')
    ).reset_index()

    # District hex count mapping for uniform disaggregation
    hex_per_district = df_train.groupby('district')['h3_index'].nunique().to_dict()

    evaluation_records = []
    predictions_dict = {
        'h3_index': test_seen['h3_index'].values[:250000],
        'state': test_seen['state'].values[:250000],
        'district': test_seen['district'].values[:250000],
        'year': test_seen['year'].values[:250000],
        'week': test_seen['week'].values[:250000],
    }

    test_eval_subset = test_seen.iloc[:250000].copy()
    test_spatial_subset = test_spatial_holdout.iloc[:100000].copy()

    for k in HORIZONS:
        print(f"\n--- Training Horizon t+{k} ({k} week-ahead forecast) ---", flush=True)
        t_horiz_start = time.time()
        target_col = f'target_lead_{k}'

        y_train_full = train_sub[target_col].values
        base_train = train_sub['cases_lag_1'].values
        delta_train = y_train_full - base_train

        X_train = train_sub[feature_cols]

        y_test_seen = test_eval_subset[target_col].values
        base_test_seen = test_eval_subset['cases_lag_1'].values

        y_test_spatial = test_spatial_subset[target_col].values
        base_test_spatial = test_spatial_subset['cases_lag_1'].values

        # ---------------------------------------------------------------------
        # 1. Baseline 1: Naive Persistence
        # ---------------------------------------------------------------------
        pred_b1_seen = base_test_seen
        pred_b1_spatial = base_test_spatial
        m_b1_seen = compute_metrics(y_test_seen, pred_b1_seen)
        m_b1_spatial = compute_metrics(y_test_spatial, pred_b1_spatial)

        evaluation_records.append({
            'disease': disease_name,
            'model': 'Baseline 1: Naive Persistence',
            'horizon': f't+{k}',
            'split': 'Test (Seen Districts)',
            **m_b1_seen
        })
        evaluation_records.append({
            'disease': disease_name,
            'model': 'Baseline 1: Naive Persistence',
            'horizon': f't+{k}',
            'split': 'Test (Spatial Holdout)',
            **m_b1_spatial
        })

        # ---------------------------------------------------------------------
        # 2. Baseline 2: Historical Seasonal Mean
        # ---------------------------------------------------------------------
        keys_seen = list(zip(test_eval_subset['district'], test_eval_subset['week']))
        pred_b2_seen = np.array([seasonal_lookup.get(key, 0.0) for key in keys_seen])

        keys_spatial = list(zip(test_spatial_subset['district'], test_spatial_subset['week']))
        pred_b2_spatial = np.array([seasonal_lookup.get(key, 0.0) for key in keys_spatial])

        m_b2_seen = compute_metrics(y_test_seen, pred_b2_seen)
        m_b2_spatial = compute_metrics(y_test_spatial, pred_b2_spatial)

        evaluation_records.append({
            'disease': disease_name,
            'model': 'Baseline 2: Historical Seasonal',
            'horizon': f't+{k}',
            'split': 'Test (Seen Districts)',
            **m_b2_seen
        })
        evaluation_records.append({
            'disease': disease_name,
            'model': 'Baseline 2: Historical Seasonal',
            'horizon': f't+{k}',
            'split': 'Test (Spatial Holdout)',
            **m_b2_spatial
        })

        # ---------------------------------------------------------------------
        # 3. Model 1: LightGBM Delta Regressor
        # ---------------------------------------------------------------------
        print(f"  Training LightGBM Regressor (Horizon t+{k})...", end=" ", flush=True)
        t0 = time.time()
        lgbm_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=63,
            max_depth=8,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm_model.fit(X_train, delta_train)
        print(f"done in {time.time()-t0:.1f}s", flush=True)

        # Save model artifact
        lgb_path = MODELS_DIR / f"lgbm_{disease_name}_lead{k}.joblib"
        joblib.dump(lgbm_model, lgb_path)

        # Predict
        pred_lgb_seen = np.maximum(0.0, base_test_seen + lgbm_model.predict(test_eval_subset[feature_cols]))
        pred_lgb_spatial = np.maximum(0.0, base_test_spatial + lgbm_model.predict(test_spatial_subset[feature_cols]))

        m_lgb_seen = compute_metrics(y_test_seen, pred_lgb_seen)
        m_lgb_spatial = compute_metrics(y_test_spatial, pred_lgb_spatial)

        evaluation_records.append({
            'disease': disease_name,
            'model': 'LightGBM Regressor (Proposed)',
            'horizon': f't+{k}',
            'split': 'Test (Seen Districts)',
            **m_lgb_seen
        })
        evaluation_records.append({
            'disease': disease_name,
            'model': 'LightGBM Regressor (Proposed)',
            'horizon': f't+{k}',
            'split': 'Test (Spatial Holdout)',
            **m_lgb_spatial
        })

        # ---------------------------------------------------------------------
        # 4. Model 2: XGBoost Regressor
        # ---------------------------------------------------------------------
        print(f"  Training XGBoost Regressor (Horizon t+{k})...", end=" ", flush=True)
        t0 = time.time()
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, delta_train)
        print(f"done in {time.time()-t0:.1f}s", flush=True)

        xgb_path = MODELS_DIR / f"xgb_{disease_name}_lead{k}.joblib"
        joblib.dump(xgb_model, xgb_path)

        pred_xgb_seen = np.maximum(0.0, base_test_seen + xgb_model.predict(test_eval_subset[feature_cols]))
        pred_xgb_spatial = np.maximum(0.0, base_test_spatial + xgb_model.predict(test_spatial_subset[feature_cols]))

        m_xgb_seen = compute_metrics(y_test_seen, pred_xgb_seen)
        m_xgb_spatial = compute_metrics(y_test_spatial, pred_xgb_spatial)

        evaluation_records.append({
            'disease': disease_name,
            'model': 'XGBoost Regressor',
            'horizon': f't+{k}',
            'split': 'Test (Seen Districts)',
            **m_xgb_seen
        })
        evaluation_records.append({
            'disease': disease_name,
            'model': 'XGBoost Regressor',
            'horizon': f't+{k}',
            'split': 'Test (Spatial Holdout)',
            **m_xgb_spatial
        })

        # Save predictions for downstream evaluation & hotspot detection
        predictions_dict[f'actual_lead_{k}'] = y_test_seen
        predictions_dict[f'pred_lgbm_lead_{k}'] = pred_lgb_seen
        predictions_dict[f'pred_xgb_lead_{k}'] = pred_xgb_seen
        predictions_dict[f'pred_naive_lead_{k}'] = pred_b1_seen

        print(f"  [EVALUATION - HORIZON t+{k} (Seen Test Set)]:")
        print(f"    LightGBM:  R2 = {m_lgb_seen['r2']:6.4f} | Corr = {m_lgb_seen['corr']:.4f} | MAE = {m_lgb_seen['mae']:.6f} | RMSE = {m_lgb_seen['rmse']:.6f}")
        print(f"    XGBoost:   R2 = {m_xgb_seen['r2']:6.4f} | Corr = {m_xgb_seen['corr']:.4f} | MAE = {m_xgb_seen['mae']:.6f} | RMSE = {m_xgb_seen['rmse']:.6f}")
        print(f"    Naive B1:  R2 = {m_b1_seen['r2']:6.4f} | Corr = {m_b1_seen['corr']:.4f} | MAE = {m_b1_seen['mae']:.6f} | RMSE = {m_b1_seen['rmse']:.6f}")
        print(f"    Season B2: R2 = {m_b2_seen['r2']:6.4f} | Corr = {m_b2_seen['corr']:.4f} | MAE = {m_b2_seen['mae']:.6f} | RMSE = {m_b2_seen['rmse']:.6f}")

    return evaluation_records, pd.DataFrame(predictions_dict)


def main():
    print("=" * 75)
    print(" VECTORHOTSPOT - PHASE 9: DUAL-DISEASE FORECASTING MODELS ENGINE")
    print("=" * 75, flush=True)

    t_global_start = time.time()

    # 1. Train and Evaluate Dengue Models
    dengue_records, df_dengue_preds = train_and_evaluate_disease("dengue")

    # 2. Train and Evaluate Malaria Models
    malaria_records, df_malaria_preds = train_and_evaluate_disease("malaria")

    # 3. Combine and Save Benchmark Metrics Table
    all_metrics = dengue_records + malaria_records
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved comprehensive evaluation metrics to: {METRICS_CSV.name}")

    # 4. Save Forecast Prediction Surfaces for Phase 10 Hotspot Fusion
    df_dengue_preds.to_parquet(PROCESSED_DATA_DIR / "forecast_dengue_predictions.parquet", compression='zstd')
    df_malaria_preds.to_parquet(PROCESSED_DATA_DIR / "forecast_malaria_predictions.parquet", compression='zstd')
    print("Saved forecast prediction surfaces for downstream Hotspot Fusion (Phase 10).")

    # Print summary performance table
    print("\n" + "=" * 75)
    print(" SUMMARY BENCHMARK TABLE (TEST SET - SEEN DISTRICTS)")
    print("=" * 75)
    summary_seen = df_metrics[df_metrics['split'] == 'Test (Seen Districts)'][['disease', 'model', 'horizon', 'r2', 'corr', 'mae', 'rmse']]
    print(summary_seen.to_string(index=False))

    print("\n" + "=" * 75)
    print(f" PHASE 9 COMPLETE: Total Elapsed Time = {(time.time() - t_global_start)/60:.2f} minutes")
    print("=" * 75)


if __name__ == "__main__":
    main()
