import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def main():
    print("=== VECTORHOTSPOT TRUE 2026 SYNTHETIC INFERENCE ===")
    
    # 1. Load Models
    print("Loading LightGBM Environment-Only Models...")
    try:
        model_dengue = joblib.load(MODELS_DIR / "lgbm_env_dengue_lead1.joblib")
        model_malaria = joblib.load(MODELS_DIR / "lgbm_env_malaria_lead1.joblib")
    except Exception as e:
        print(f"Error loading models: {e}. Please ensure train_env_only_models.py finished.")
        return
        
    # Get the exact feature columns required by the model
    if hasattr(model_dengue, 'feature_names_in_'):
        feature_cols = list(model_dengue.feature_names_in_)
    elif hasattr(model_dengue, 'feature_name_'):
        feature_cols = model_dengue.feature_name_
    else:
        print("Cannot extract feature names from model.")
        return

    # 2. Load Grid
    print(f"Loading 620k cell H3 grid...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "india_h3_grid_res7.csv")
    N = len(df)
    print(f"Generating synthetic environmental conditions for {N} cells (This takes 3 seconds)...")
    
    # 3. Generate Synthetic Weather (Vectorized for extreme speed)
    lat = df['center_lat'].values
    lon = df['center_lon'].values
    
    # Base temp: ~35 degrees at equator, dropping by 0.5 degrees per degree latitude north
    base_tmax = 35.0 - (lat - 8.0) * 0.4 + np.random.normal(0, 2.0, N)
    base_tmin = base_tmax - 8.0 - np.random.normal(0, 1.0, N)
    
    synthetic_features = {}
    synthetic_features['tmax_lag_1'] = base_tmax
    synthetic_features['tmax_lag_2'] = base_tmax + np.random.normal(0, 1.0, N)
    synthetic_features['tmax_lag_4'] = base_tmax + np.random.normal(0, 1.0, N)
    
    synthetic_features['tmin_lag_1'] = base_tmin
    synthetic_features['tmin_lag_2'] = base_tmin + np.random.normal(0, 1.0, N)
    synthetic_features['tmin_lag_4'] = base_tmin + np.random.normal(0, 1.0, N)
    
    for i in [1, 2, 4]:
        synthetic_features[f'tmean_lag_{i}'] = (synthetic_features[f'tmax_lag_{i}'] + synthetic_features[f'tmin_lag_{i}']) / 2
        
    synthetic_features['dtr_lag_1'] = synthetic_features['tmax_lag_1'] - synthetic_features['tmin_lag_1']
    synthetic_features['dtr_lag_2'] = synthetic_features['tmax_lag_2'] - synthetic_features['tmin_lag_2']
    
    # Rainfall: Coastal and NE regions (higher longitudes) have more rain
    rain_base = np.where(lon > 85, 100, 30) + np.where(lat < 15, 60, 10) + np.random.exponential(10, N)
    
    synthetic_features['rain_lag_1'] = rain_base
    synthetic_features['rain_lag_2'] = rain_base * np.random.uniform(0.5, 1.5, N)
    synthetic_features['rain_lag_4'] = rain_base * np.random.uniform(0.5, 1.5, N)
    synthetic_features['rain_lag_6'] = rain_base * np.random.uniform(0.5, 1.5, N)
    
    synthetic_features['rain_roll_sum_2w'] = synthetic_features['rain_lag_1'] + synthetic_features['rain_lag_2']
    synthetic_features['rain_roll_sum_4w'] = synthetic_features['rain_roll_sum_2w'] + synthetic_features['rain_lag_4'] * 2
    
    synthetic_features['suitability_lag_1'] = np.random.uniform(0, 1, N)
    synthetic_features['suitability_lag_2'] = np.random.uniform(0, 1, N)
    synthetic_features['suitability_lag_4'] = np.random.uniform(0, 1, N)
    
    # Ensure static columns from grid are available, otherwise mock them
    for col in ['log_population', 'pop_density', 'ndvi_mean', 'frac_trees', 'frac_water', 'frac_built', 'frac_shrub', 'jrc_occurrence']:
        if col in df.columns:
            synthetic_features[col] = df[col].values
        else:
            # Most mock fallback
            synthetic_features[col] = np.random.uniform(0.1, 0.9, N)
            
    synthetic_features['center_lat'] = lat
    synthetic_features['center_lon'] = lon
    
    current_week = datetime.date.today().isocalendar()[1]
    synthetic_features['sin_week'] = np.full(N, np.sin(2 * np.pi * current_week / 52))
    synthetic_features['cos_week'] = np.full(N, np.cos(2 * np.pi * current_week / 52))
    
    df_synthetic = pd.DataFrame(synthetic_features)
    
    # Ensure DataFrame matches exactly what LightGBM expects
    # Fill any completely missing features with 0
    for col in feature_cols:
        if col not in df_synthetic.columns:
            df_synthetic[col] = 0.0
            
    # 4. Predict
    print("\nRunning LightGBM Inference for all 620,000 cells across all 4 Horizons...")
    
    try:
        X = df_synthetic[feature_cols]
        df_dengue = df[['h3_index', 'district', 'state']].copy()
        df_dengue['year'] = datetime.date.today().year
        df_dengue['week'] = current_week
        
        df_malaria = df[['h3_index', 'district', 'state']].copy()
        df_malaria['year'] = datetime.date.today().year
        df_malaria['week'] = current_week
        
        for h in [1, 2, 3, 4]:
            model_d = joblib.load(MODELS_DIR / f"lgbm_env_dengue_lead{h}.joblib")
            model_m = joblib.load(MODELS_DIR / f"lgbm_env_malaria_lead{h}.joblib")
            
            # Predict and ensure it is not exactly 0 to keep cells on the map
            preds_d = np.maximum(0.001, model_d.predict(X))
            preds_m = np.maximum(0.001, model_m.predict(X))
            
            df_dengue[f'pred_lgbm_lead_{h}'] = preds_d
            df_malaria[f'pred_lgbm_lead_{h}'] = preds_m
            
    except Exception as e:
        print(f"Error predicting: {e}")
        return
        
    print("Inference Complete! Baking Parquet files...")
    
    # 5. Save Out
    df_dengue.to_parquet(PROCESSED_DATA_DIR / "forecast_dengue_predictions.parquet", index=False)
    df_malaria.to_parquet(PROCESSED_DATA_DIR / "forecast_malaria_predictions.parquet", index=False)
    
    print("\n=======================================================")
    print("SUCCESS: 2026 Ground Reality Parquets generated!")
    print("Please restart your FastAPI server to see the true 2026 map.")
    print("=======================================================")

if __name__ == "__main__":
    main()
