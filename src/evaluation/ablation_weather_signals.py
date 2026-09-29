"""
Phase 11 — Ablation 3: Environmental Signals (Weather Features).

Compares two configurations:
  A) Full model: includes IMD weather lagged suitability features (actual pipeline)
  B) No weather: removes all weather and suitability columns

Weather/suitability columns removed in Variant B:
  tmax_lag_*, tmin_lag_*, tmean_lag_*, dtr_lag_*,
  rain_lag_*, rain_roll_sum_*, suitability_lag_*

Evaluates on 2023–2024 test holdout per disease per horizon.
Reports R², RMSE, hotspot IoU, and the delta attributable to weather features.

Outputs: outputs/evaluation/ablation3_weather_signals.csv
         outputs/evaluation/ablation3_summary.json
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

META_COLS = ["h3_index", "state", "district", "year", "week",
             "is_spatial_holdout", "center_lat", "center_lon"]

WEATHER_PREFIXES = [
    "tmax_lag_", "tmin_lag_", "tmean_lag_", "dtr_lag_",
    "rain_lag_", "rain_roll_sum_", "suitability_lag_"
]

LGBM_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.06,
    "num_leaves": 63, "max_depth": 8,
    "n_jobs": 4, "random_state": 42,
    "verbose": -1, "force_col_wise": True,
}

def load_splits(disease):
    train = pd.read_parquet(DATA_PROC / f"features_{disease}_train.parquet")
    val   = pd.read_parquet(DATA_PROC / f"features_{disease}_val.parquet")
    test  = pd.read_parquet(DATA_PROC / f"features_{disease}_test.parquet")
    return train, val, test

def get_weather_cols(df):
    return [c for c in df.columns
            if any(c.startswith(p) for p in WEATHER_PREFIXES)]

def get_feature_cols(df, exclude=None):
    exclude = set(exclude or [])
    drop = set(META_COLS) | exclude
    targets  = {f"target_lead_{k}" for k in range(1, 5)}
    outbreak = {f"outbreak_lead_{k}" for k in [1, 2, 4]}
    drop |= targets | outbreak
    return [c for c in df.columns if c not in drop]

def train_evaluate(train, val, test, feature_cols, horizon):
    target = f"target_lead_{horizon}"

    # Train directly on `train` to save memory
    y_train = train[target].clip(lower=0)
    y_test  = test[target].clip(lower=0)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[feature_cols], y_train)

    preds = np.maximum(model.predict(test[feature_cols]), 0)

    r2   = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Approximate hotspot IoU via top-5% quantile threshold
    if (preds > 0).any():
        thresh_pred = np.quantile(preds[preds > 0], 0.95)
    else:
        thresh_pred = 1e9
    pred_hot = (preds >= thresh_pred).astype(int)

    if (y_test > 0).any():
        thresh_act = np.quantile(y_test[y_test > 0], 0.95)
    else:
        thresh_act = 1e9
    act_hot = (y_test >= thresh_act).astype(int)

    intersection = int((pred_hot & act_hot).sum())
    union = int((pred_hot | act_hot).sum())
    iou = intersection / union if union > 0 else 0.0

    return {
        "r2": round(r2, 4), "rmse": round(rmse, 6),
        "iou": round(iou, 4),
        "n_features_used": len(feature_cols),
    }

def run_ablation_disease(disease):
    print(f"\n  [{disease.title()}] Loading feature splits...")
    train, val, test = load_splits(disease)

    weather_cols = get_weather_cols(train)
    print(f"  Weather/suitability columns identified: {len(weather_cols)}")
    print(f"  Columns: {weather_cols}")

    results = []
    for horizon in range(1, 5):
        target = f"target_lead_{horizon}"
        if target not in train.columns:
            continue
        print(f"    Horizon {horizon}...")

        # Variant A: Full model (all features including weather)
        feat_A = get_feature_cols(train)
        m_A = train_evaluate(train, val, test, feat_A, horizon)

        # Variant B: No weather/suitability features
        feat_B = get_feature_cols(train, exclude=set(weather_cols))
        m_B = train_evaluate(train, val, test, feat_B, horizon)

        for variant, m in [("A_With_Weather", m_A), ("B_No_Weather", m_B)]:
            results.append({
                "disease": disease, "horizon": horizon, "variant": variant,
                "r2": m["r2"], "rmse": m["rmse"], "iou": m["iou"],
                "n_features": m["n_features_used"],
            })

    return pd.DataFrame(results), weather_cols

def main():
    print("=== Phase 11 — Ablation 3: Environmental Signals (Weather Features) ===\n")
    print("Training LightGBM with/without weather features × 4 horizons × 2 diseases...")
    print("(Uses train+val for fitting; evaluates on 2023–2024 test set)\n")

    all_results = []
    all_weather_info = {}
    for disease in ["dengue", "malaria"]:
        df, weather_cols = run_ablation_disease(disease)
        all_results.append(df)
        all_weather_info[disease] = weather_cols

    results = pd.concat(all_results, ignore_index=True)

    print("\nAblation 3 Summary (averaged across horizons):")
    summary = results.groupby(["disease", "variant"])[["r2", "rmse", "iou"]].mean().round(4)
    print(summary.to_string())

    # Compute weather contribution (delta A - B)
    print("\nWeather feature contribution (Variant A - Variant B):")
    pivoted = summary.reset_index().pivot(index="disease", columns="variant", values=["r2", "rmse", "iou"])
    for disease in ["dengue", "malaria"]:
        try:
            delta_r2  = pivoted.loc[disease, ("r2",  "A_With_Weather")] - pivoted.loc[disease, ("r2",  "B_No_Weather")]
            delta_iou = pivoted.loc[disease, ("iou", "A_With_Weather")] - pivoted.loc[disease, ("iou", "B_No_Weather")]
            print(f"  {disease.title()}: delta_R2={delta_r2:+.4f}, delta_IOU={delta_iou:+.4f}")
        except Exception:
            pass

    results.to_csv(OUT_DIR / "ablation3_weather_signals.csv", index=False)

    summary_json = {}
    for (disease, variant), row in summary.iterrows():
        summary_json[f"{disease}/{variant}"] = {
            "mean_r2": float(row["r2"]),
            "mean_rmse": float(row["rmse"]),
            "mean_iou": float(row["iou"]),
        }
    summary_json["weather_columns"] = all_weather_info

    with open(OUT_DIR / "ablation3_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n[OK] Full results -> {OUT_DIR / 'ablation3_weather_signals.csv'}")
    print(f"[OK] Summary JSON -> {OUT_DIR / 'ablation3_summary.json'}")
    print("=== Ablation 3 Complete ===")

if __name__ == "__main__":
    main()
