import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
MODELS_DIR = PROJECT_ROOT / "models"

def run_live_inference(disease="dengue", horizon=1):
    print(f"--- Running Live 2026 Inference for {disease.upper()} (Horizon t+{horizon}) ---")
    
    # 1. Load Live Weather Data
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    weather_path = LIVE_DATA_DIR / f"live_weather_{today_str}.json"
    
    if not weather_path.exists():
        print(f"Live weather data not found for today ({today_str}). Please run fetch_live_weather.py first.")
        return
        
    with open(weather_path, 'r') as f:
        weather_data = json.load(f)
        
    print(f"Loaded live weather for {len(weather_data)} H3 cells.")
    
    # 2. Extract and Compute Features
    live_features = []
    
    for cell in weather_data:
        h3_id = cell['h3_index']
        daily = cell.get('daily', {})
        
        # Open-Meteo returns lists of data per day. We need to aggregate the past 4 weeks (28 days).
        # This is a simplified mockup of the feature engineering pipeline.
        tmax = daily.get('temperature_2m_max', [])
        tmin = daily.get('temperature_2m_min', [])
        rain = daily.get('precipitation_sum', [])
        
        # Ensure we have data
        if not tmax or len(tmax) < 28:
            continue
            
        # Mock calculation: average over the past 7 days for 'lag_1'
        tmax_lag_1 = np.mean(tmax[-7:])
        tmin_lag_1 = np.mean(tmin[-7:])
        tmean_lag_1 = (tmax_lag_1 + tmin_lag_1) / 2
        dtr_lag_1 = tmax_lag_1 - tmin_lag_1
        rain_lag_1 = np.sum(rain[-7:])
        
        # Mock rolling sum for past 4 weeks
        rain_roll_sum_4w = np.sum(rain[-28:])
        
        # Example feature dict (Must match the 32 features expected by Environment-Only Model)
        feature_row = {
            'h3_index': h3_id,
            'tmax_lag_1': tmax_lag_1,
            'tmax_lag_2': np.mean(tmax[-14:-7]),
            'tmax_lag_4': np.mean(tmax[-28:-21]),
            'tmin_lag_1': tmin_lag_1,
            'tmin_lag_2': np.mean(tmin[-14:-7]),
            'tmin_lag_4': np.mean(tmin[-28:-21]),
            'tmean_lag_1': tmean_lag_1,
            'tmean_lag_2': (np.mean(tmax[-14:-7]) + np.mean(tmin[-14:-7])) / 2,
            'tmean_lag_4': (np.mean(tmax[-28:-21]) + np.mean(tmin[-28:-21])) / 2,
            'dtr_lag_1': dtr_lag_1,
            'dtr_lag_2': np.mean(tmax[-14:-7]) - np.mean(tmin[-14:-7]),
            'rain_lag_1': rain_lag_1,
            'rain_lag_2': np.sum(rain[-14:-7]),
            'rain_lag_4': np.sum(rain[-28:-21]),
            'rain_lag_6': np.sum(rain[-42:-35]) if len(rain) >= 42 else 0,
            'rain_roll_sum_2w': np.sum(rain[-14:]),
            'rain_roll_sum_4w': rain_roll_sum_4w,
            # (Note: In a full pipeline, you would join this with static variables like population, NDVI)
            'suitability_lag_1': 0.5, 'suitability_lag_2': 0.5, 'suitability_lag_4': 0.5,
            'log_population': 10.0, 'pop_density': 500.0, 'ndvi_mean': 0.4,
            'frac_trees': 0.2, 'frac_water': 0.05, 'frac_built': 0.3, 'frac_shrub': 0.1,
            'jrc_occurrence': 0, 'center_lat': 20.0, 'center_lon': 78.0,
            'sin_week': np.sin(2 * np.pi * datetime.date.today().isocalendar()[1] / 52),
            'cos_week': np.cos(2 * np.pi * datetime.date.today().isocalendar()[1] / 52)
        }
        live_features.append(feature_row)
        
    df_live = pd.DataFrame(live_features)
    print(f"Extracted features for {len(df_live)} cells.")
    
    # 3. Load Model and Predict
    model_path = MODELS_DIR / f"lgbm_env_{disease}_lead{horizon}.joblib"
    print(f"Loading Environment-Only model from {model_path}...")
    
    if not model_path.exists():
        print("Model file not found. Please ensure train_env_only_models.py has completed.")
        return
        
    model = joblib.load(model_path)
    
    # The columns must perfectly match the order used during training.
    # In a real script, we would enforce column order here based on the training feature list.
    feature_cols = [c for c in df_live.columns if c != 'h3_index']
    
    print("Running Inference...")
    preds = model.predict(df_live[feature_cols])
    
    # Force predictions to be non-negative
    preds = np.maximum(0.0, preds)
    
    df_live[f'pred_lgbm_lead_{horizon}'] = preds
    
    # 4. Output Results
    # Normally we would save this to forecast_{disease}_predictions.parquet, but we save locally for demo.
    out_file = LIVE_DATA_DIR / f"live_predictions_{disease}_lead{horizon}_{today_str}.csv"
    df_live[['h3_index', f'pred_lgbm_lead_{horizon}']].to_csv(out_file, index=False)
    
    print(f"\nSUCCESS: Generated LIVE 2026 predictions!")
    print(f"Top 5 highest risk cells right now:")
    top5 = df_live.nlargest(5, f'pred_lgbm_lead_{horizon}')
    for idx, row in top5.iterrows():
        print(f"  {row['h3_index']}: {row[f'pred_lgbm_lead_{horizon}']:.4f}")
        
    print(f"\nSaved full predictions to: {out_file}")

if __name__ == "__main__":
    run_live_inference("dengue", 1)
