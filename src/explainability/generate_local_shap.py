#!/usr/bin/env python3
"""
VectorHotspot — Local SHAP Generator (per-cell)
Outputs: outputs/explainability/local_shap_{disease}_lead{h}.parquet
         Columns: h3_index | feature_1 | ... | feature_N (signed SHAP values)
"""
import pandas as pd
import numpy as np
import shap
import joblib
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
EXPLAIN_DIR   = PROJECT_ROOT / "outputs" / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = ["dengue", "malaria"]
HORIZONS = [1, 2, 3, 4]

META_COLS = {
    'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
    'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4',
    'outbreak_lead_1', 'outbreak_lead_2', 'outbreak_lead_4',
}

def compute_local_shap(disease, horizon):
    print(f"\n--- {disease.upper()} t+{horizon} ---")
    model_path = MODELS_DIR / f"lgbm_env_{disease}_lead{horizon}.joblib"
    if not model_path.exists():
        print(f"  [SKIP] Model not found"); return
    model = joblib.load(model_path)

    feat_path = PROCESSED_DIR / f"features_{disease}_test.parquet"
    print(f"  Loading features...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(feat_path)
    print(f"done ({len(df):,} rows, {time.time()-t0:.1f}s)")

    latest_year = int(df['year'].max())
    latest_week = int(df[df['year'] == latest_year]['week'].max())
    df_latest = df[(df['year'] == latest_year) & (df['week'] == latest_week)].copy()
    print(f"  Latest week: {latest_year} W{latest_week} — {len(df_latest):,} cells")

    feature_cols = [c for c in df_latest.columns
                    if c not in META_COLS and 'case' not in c and 'neighbor' not in c]
    X = df_latest[feature_cols].values
    h3_indices = df_latest['h3_index'].values

    print(f"  Computing SHAP for {len(X):,} cells...", end=" ", flush=True)
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    print(f"done ({time.time()-t0:.1f}s)")

    df_shap = pd.DataFrame(shap_vals, columns=feature_cols)
    df_shap.insert(0, 'h3_index', h3_indices)

    out_path = EXPLAIN_DIR / f"local_shap_{disease}_lead{horizon}.parquet"
    df_shap.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df_shap):,} rows)")

for disease in DISEASES:
    for horizon in HORIZONS:
        out_path = EXPLAIN_DIR / f"local_shap_{disease}_lead{horizon}.parquet"
        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue
        compute_local_shap(disease, horizon)

print("\nDone. Restart backend to pick up new SHAP files.")
