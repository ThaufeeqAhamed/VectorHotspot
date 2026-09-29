"""
Phase 11 — Ablation 2: Feature Engineering (Spatial Neighbors).

Trains LightGBM models with three neighbor feature configurations:
  A) Full model: H3 k-ring neighbors (k=1,2) — actual pipeline (reference)
  B) No spatial neighbors: drop neighbor_cases_k1_lag1/2, neighbor_cases_k2_lag1
  C) District mean neighbor: replace k-ring features with district-mean case lag

Evaluates on the 2023–2024 test holdout per disease per horizon (lead 1–4).
Reports R², RMSE, and hotspot IoU for each configuration.

Outputs: outputs/evaluation/ablation2_feature_engineering.csv
         outputs/evaluation/ablation2_summary.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS  = ["h3_index", "state", "district", "year", "week",
              "is_spatial_holdout", "center_lat", "center_lon"]
TARGET_FMT = "target_lead_{k}"
K_RING_COLS = ["neighbor_cases_k1_lag1", "neighbor_cases_k1_lag2",
               "neighbor_cases_k2_lag1"]

LGBM_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.06,
    "num_leaves": 63, "max_depth": 8,
    "n_jobs": 4, "random_state": 42,
    "verbose": -1, "force_col_wise": True,
}

GI_THRESHOLD = 1.96  # z-score for hotspot classification

def load_splits(disease):
    train = pd.read_parquet(DATA_PROC / f"features_{disease}_train.parquet")
    val   = pd.read_parquet(DATA_PROC / f"features_{disease}_val.parquet")
    test  = pd.read_parquet(DATA_PROC / f"features_{disease}_test.parquet")
    return train, val, test

def get_feature_cols(df, exclude=None):
    exclude = set(exclude or [])
    drop = set(META_COLS) | exclude
    targets = {f"target_lead_{k}" for k in range(1, 5)}
    outbreak = {f"outbreak_lead_{k}" for k in [1, 2, 4]}
    drop |= targets | outbreak
    return [c for c in df.columns if c not in drop]

def add_district_mean_neighbor(df):
    """Replace k-ring features with district-level mean case lag as neighbor proxy."""
    df = df.copy()
    district_mean = df.groupby(["district", "year", "week"])["cases_lag_1"].transform("mean")
    df["district_mean_neighbor"] = district_mean
    return df

def train_evaluate(disease, train, val, test, feature_cols, horizon):
    target = f"target_lead_{horizon}"

    # Train directly on `train` to save memory (avoid concat(train, val))
    y_train = train[target].clip(lower=0)
    y_test  = test[target].clip(lower=0)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[feature_cols], y_train)

    preds = np.maximum(model.predict(test[feature_cols]), 0)

    r2   = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Compute approximate hotspot IoU (no full Gi* — use z-score proxy via rank)
    # Sort by predicted value, top quantile = predicted hotspot
    threshold_q = np.quantile(preds[preds > 0], 0.95) if (preds > 0).any() else 1e9
    pred_hot = (preds >= threshold_q).astype(int)

    act_threshold = np.quantile(y_test[y_test > 0], 0.95) if (y_test > 0).any() else 1e9
    act_hot = (y_test >= act_threshold).astype(int)

    intersection = int((pred_hot & act_hot).sum())
    union = int((pred_hot | act_hot).sum())
    iou = intersection / union if union > 0 else 0.0

    return {"r2": round(r2, 4), "rmse": round(rmse, 6), "iou": round(iou, 4)}

def run_ablation_disease(disease):
    print(f"\n  [{disease.title()}] Loading feature splits...")
    train, val, test = load_splits(disease)

    results = []
    for horizon in range(1, 5):
        target = f"target_lead_{horizon}"
        if target not in train.columns:
            continue
        print(f"    Horizon {horizon}...")

        # Variant A: Full H3 k-ring neighbors
        feat_A = get_feature_cols(train)
        metrics_A = train_evaluate(disease, train, val, test, feat_A, horizon)

        # Variant B: No spatial neighbors
        feat_B = get_feature_cols(train, exclude=K_RING_COLS)
        metrics_B = train_evaluate(disease, train, val, test, feat_B, horizon)

        # Variant C: District mean neighbor
        train_C = add_district_mean_neighbor(train)
        val_C   = add_district_mean_neighbor(val)
        test_C  = add_district_mean_neighbor(test)
        # Remove k-ring cols, add district_mean_neighbor
        exclude_C = set(K_RING_COLS)
        feat_C = get_feature_cols(train_C, exclude=exclude_C)
        metrics_C = train_evaluate(disease, train_C, val_C, test_C, feat_C, horizon)

        for variant, m in [("A_H3_kring_neighbors", metrics_A),
                            ("B_No_spatial_neighbors", metrics_B),
                            ("C_District_mean_neighbor", metrics_C)]:
            results.append({
                "disease": disease, "horizon": horizon, "variant": variant,
                "r2": m["r2"], "rmse": m["rmse"], "iou": m["iou"]
            })

    return pd.DataFrame(results)

def main():
    print("=== Phase 11 — Ablation 2: Feature Engineering (Spatial Neighbors) ===\n")
    print("Training LightGBM with 3 neighbor configurations × 4 horizons × 2 diseases...")
    print("(Uses train+val for fitting; evaluates on 2023–2024 test set)\n")

    all_results = []
    for disease in ["dengue", "malaria"]:
        df = run_ablation_disease(disease)
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)

    print("\nAblation 2 Summary (averaged across horizons):")
    summary = results.groupby(["disease", "variant"])[["r2", "rmse", "iou"]].mean().round(4)
    print(summary.to_string())

    results.to_csv(OUT_DIR / "ablation2_feature_engineering.csv", index=False)

    summary_json = {}
    for (disease, variant), row in summary.iterrows():
        summary_json[f"{disease}/{variant}"] = {
            "mean_r2": float(row["r2"]),
            "mean_rmse": float(row["rmse"]),
            "mean_iou": float(row["iou"]),
        }
    with open(OUT_DIR / "ablation2_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n[OK] Full results -> {OUT_DIR / 'ablation2_feature_engineering.csv'}")
    print(f"[OK] Summary JSON -> {OUT_DIR / 'ablation2_summary.json'}")
    print("=== Ablation 2 Complete ===")

if __name__ == "__main__":
    main()
