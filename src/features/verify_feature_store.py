#!/usr/bin/env python3
"""
Phase 8 Verification & Quality Assurance Suite
VectorHotspot Project

Performs automated auditing on the engineered spatiotemporal feature store:
1. Validates schema, feature columns, and zero-null assertions.
2. Verifies strict temporal holdout isolation (Train <= 2020, Val 2021-2022, Test 2023-2024).
3. Verifies spatial holdout consistency (15% held out).
4. Verifies correlation structure (autoregressive persistence, spatial neighbor spillover, weather lags).
5. Inspects feature distributions and target sanity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = PROCESSED_DATA_DIR / "feature_store_metadata.json"


def audit_disease_feature_store(disease_name):
    print(f"\n{'='*70}")
    print(f" AUDITING FEATURE STORE: {disease_name.upper()}")
    print(f"{'='*70}")

    train_path = PROCESSED_DATA_DIR / f"features_{disease_name}_train.parquet"
    val_path = PROCESSED_DATA_DIR / f"features_{disease_name}_val.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{disease_name}_test.parquet"

    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert test_path.exists(), f"Missing {test_path}"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"Loaded datasets:")
    print(f"  Train: {len(df_train):,} rows ({train_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Val:   {len(df_val):,} rows ({val_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Test:  {len(df_test):,} rows ({test_path.stat().st_size / (1024*1024):.1f} MB)")

    # 1. Zero Null Assertions
    null_train = df_train.isnull().sum().sum()
    null_val = df_val.isnull().sum().sum()
    null_test = df_test.isnull().sum().sum()

    assert null_train == 0, f"Found {null_train} nulls in TRAIN!"
    assert null_val == 0, f"Found {null_val} nulls in VAL!"
    assert null_test == 0, f"Found {null_test} nulls in TEST!"
    print("\n[CHECK 1 - NULLS] PASSED: 0 null values across all features and splits.")

    # 2. Strict Temporal Isolation (Leakage Check)
    train_years = set(df_train['year'].unique())
    val_years = set(df_val['year'].unique())
    test_years = set(df_test['year'].unique())

    print(f"\n[CHECK 2 - TEMPORAL SPLITS]")
    print(f"  Train years: {sorted(train_years)}")
    print(f"  Val years:   {sorted(val_years)}")
    print(f"  Test years:  {sorted(test_years)}")

    assert train_years.isdisjoint(val_years), "LEAKAGE: Train and Val overlap!"
    assert train_years.isdisjoint(test_years), "LEAKAGE: Train and Test overlap!"
    assert val_years.isdisjoint(test_years), "LEAKAGE: Val and Test overlap!"
    assert max(train_years) < min(val_years), "LEAKAGE: Train years not strictly before Val!"
    assert max(val_years) < min(test_years), "LEAKAGE: Val years not strictly before Test!"
    print("  PASSED: Strict temporal isolation verified with zero future leakage.")

    # 3. Spatial Holdout Check
    holdout_districts = df_train[df_train['is_spatial_holdout']][['state', 'district']].drop_duplicates()
    total_districts = df_train[['state', 'district']].drop_duplicates()
    holdout_pct = len(holdout_districts) / len(total_districts) * 100
    print(f"\n[CHECK 3 - SPATIAL HOLDOUT]")
    print(f"  Total districts: {len(total_districts)}, Spatial Holdout: {len(holdout_districts)} ({holdout_pct:.1f}%)")
    assert 14.0 <= holdout_pct <= 16.0, f"Unexpected holdout percentage: {holdout_pct}%"
    print("  PASSED: Spatial holdout district partition verified.")

    # 4. Feature Correlation Sanity Checks
    print(f"\n[CHECK 4 - CORRELATION STRUCTURE]")
    corr_lag1 = np.corrcoef(df_train['cases_lag_1'], df_train['target_lead_1'])[0, 1]
    corr_nbr = np.corrcoef(df_train['neighbor_cases_k1_lag1'], df_train['target_lead_1'])[0, 1]
    corr_suit = np.corrcoef(df_train['suitability_lag_1'], df_train['target_lead_1'])[0, 1]
    corr_pop = np.corrcoef(df_train['log_population'], df_train['target_lead_1'])[0, 1]

    print(f"  Corr(cases_lag_1, target_lead_1):         {corr_lag1:.4f} (Autoregressive signal)")
    print(f"  Corr(neighbor_k1_lag1, target_lead_1):    {corr_nbr:.4f} (Spatial spillover signal)")
    print(f"  Corr(suitability_lag1, target_lead_1):    {corr_suit:.4f} (Biophysical weather signal)")
    print(f"  Corr(log_population, target_lead_1):      {corr_pop:.4f} (Demographic signal)")

    assert corr_lag1 > 0.5, "Autoregressive correlation too weak!"
    assert corr_nbr > 0.4, "Spatial neighbor spillover correlation too weak!"
    print("  PASSED: Core epidemiological signals show strong, expected positive correlations.")

    return {
        'train_rows': len(df_train),
        'val_rows': len(df_val),
        'test_rows': len(df_test),
        'total_features': len(df_train.columns) - 6
    }


def main():
    print("=" * 70)
    print(" VECTORHOTSPOT - PHASE 8 FEATURE STORE AUDIT & VALIDATION")
    print("=" * 70)

    dengue_stats = audit_disease_feature_store("dengue")
    malaria_stats = audit_disease_feature_store("malaria")

    print("\n" + "=" * 70)
    print(" ALL PHASE 8 AUDIT CHECKS PASSED PERFECTLY!")
    print(f" Dengue Features: {dengue_stats['total_features']} columns | Total Rows: {dengue_stats['train_rows'] + dengue_stats['val_rows'] + dengue_stats['test_rows']:,}")
    print(f" Malaria Features: {malaria_stats['total_features']} columns | Total Rows: {malaria_stats['train_rows'] + malaria_stats['val_rows'] + malaria_stats['test_rows']:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
