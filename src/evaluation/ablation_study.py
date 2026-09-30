#!/usr/bin/env python3
"""
Feature Ablation Study
VectorHotspot Project

This script performs a formal ablation study on the trained forecasting models.
It iteratively removes specific groups of features (e.g., Weather, Spatial, Ecology)
and retrains a LightGBM regressor to quantify exactly how much predictive power
each feature group contributes to the final model (Horizon 1).

Outputs:
- outputs/tables/ablation_study_results.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import time
import json

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_CSV = OUTPUTS_TABLES_DIR / "ablation_study_results.csv"

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }

def run_ablation_for_disease(disease_name):
    print(f"\n{'='*70}")
    print(f" RUNNING FEATURE ABLATION STUDY: {disease_name.upper()}")
    print(f"{'='*70}")

    train_path = PROCESSED_DATA_DIR / f"features_{disease_name}_train.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{disease_name}_test.parquet"

    print("Loading datasets...", flush=True)
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Use a subset of training data for faster ablation
    n_train_sub = min(len(df_train), 500000)
    train_sub = df_train[~df_train['is_spatial_holdout']].sample(n_train_sub, random_state=42)
    test_seen = df_test[~df_test['is_spatial_holdout']].iloc[:100000].copy()

    # Define metadata columns to exclude from features
    meta_cols = {
        'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
        'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4',
        'outbreak_lead_1', 'outbreak_lead_2', 'outbreak_lead_4'
    }
    
    all_features = [c for c in df_train.columns if c not in meta_cols]

    # Categorize features into logical groups based on prefixes/names
    feature_groups = {
        'Base_Cases': [f for f in all_features if 'case' in f and 'neighbor' not in f],
        'Weather': [f for f in all_features if 'tmean' in f or 'tmin' in f or 'tmax' in f or 'rain' in f or 'humid' in f or 'suitability' in f],
        'Spatial_Neighbors': [f for f in all_features if 'neighbor' in f],
        'Temporal_Seasonality': ['sin_week', 'cos_week', 'month'],
        'Ecology_Population': [f for f in all_features if 'pop' in f or 'frac_' in f or 'ndvi' in f or 'elevation' in f]
    }

    # Verify grouping completeness
    grouped_features = set()
    for g, feats in feature_groups.items():
        grouped_features.update(feats)
    
    # Catch any ungrouped features and put them in a 'Misc' group
    misc_features = [f for f in all_features if f not in grouped_features]
    if misc_features:
        feature_groups['Misc'] = misc_features

    print("\nFeature Group Definitions:")
    for group_name, feats in feature_groups.items():
        print(f"  - {group_name}: {len(feats)} features")

    target_col = 'target_lead_1'
    y_train_full = train_sub[target_col].values
    base_train = train_sub['cases_lag_1'].values
    delta_train = y_train_full - base_train
    
    y_test = test_seen[target_col].values
    base_test = test_seen['cases_lag_1'].values

    # Define ablation scenarios
    scenarios = [
        ("All Features (Baseline)", all_features),
        ("No Weather", [f for f in all_features if f not in feature_groups['Weather']]),
        ("No Spatial (Neighbors)", [f for f in all_features if f not in feature_groups['Spatial_Neighbors']]),
        ("No Temporal (Seasonality)", [f for f in all_features if f not in feature_groups['Temporal_Seasonality']]),
        ("No Ecology & Population", [f for f in all_features if f not in feature_groups['Ecology_Population']]),
        ("Only Base Cases (No Exogenous)", feature_groups['Base_Cases'])
    ]

    results = []

    for scenario_name, active_features in scenarios:
        print(f"\nTraining Scenario: '{scenario_name}' ({len(active_features)} features)...", end=" ", flush=True)
        t0 = time.time()
        
        # Handle case where features might not be in dataframe (e.g., if group is empty)
        valid_features = [f for f in active_features if f in df_train.columns]
        
        X_train = train_sub[valid_features]
        X_test = test_seen[valid_features]

        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=150,  # slightly reduced for faster ablation
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, delta_train)
        
        # Predict delta and add to base
        pred = np.maximum(0.0, base_test + model.predict(X_test))
        metrics = compute_metrics(y_test, pred)
        
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")
        print(f"   --> R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}")

        results.append({
            'Disease': disease_name.capitalize(),
            'Scenario': scenario_name,
            'Features_Used': len(valid_features),
            'R2': round(metrics['r2'], 4),
            'MAE': round(metrics['mae'], 4),
            'RMSE': round(metrics['rmse'], 4)
        })

    return results

def main():
    print("Starting VectorHotspot Feature Ablation Study...")
    
    dengue_results = run_ablation_for_disease("dengue")
    malaria_results = run_ablation_for_disease("malaria")
    
    all_results = dengue_results + malaria_results
    df_results = pd.DataFrame(all_results)
    
    df_results.to_csv(ABLATION_CSV, index=False)
    
    print("\n" + "="*70)
    print(" ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(df_results.to_string(index=False))
    print(f"\nDetailed results saved to: {ABLATION_CSV}")

if __name__ == "__main__":
    main()
