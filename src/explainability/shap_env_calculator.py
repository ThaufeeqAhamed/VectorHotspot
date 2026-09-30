import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPLAIN_DIR = PROJECT_ROOT / "outputs" / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

def calculate_env_shap(disease="dengue", horizon=1):
    print(f"\n--- Calculating SHAP for Environment-Only Model ({disease.upper()} t+{horizon}) ---")
    
    # 1. Load Model
    model_path = MODELS_DIR / f"lgbm_env_{disease}_lead{horizon}.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    model = joblib.load(model_path)
    
    # 2. Load Test Data
    test_path = PROCESSED_DATA_DIR / f"features_{disease}_test.parquet"
    print("Loading test data...")
    df_test = pd.read_parquet(test_path)
    
    # Extract the exact feature columns used by the Environment-Only model
    meta_cols = {
        'h3_index', 'state', 'district', 'year', 'week', 'is_spatial_holdout',
        'target_lead_1', 'target_lead_2', 'target_lead_3', 'target_lead_4',
        'outbreak_lead_1', 'outbreak_lead_2', 'outbreak_lead_4'
    }
    feature_cols = [c for c in df_test.columns if c not in meta_cols and 'case' not in c and 'neighbor' not in c]
    
    # Sample 10,000 rows for SHAP background (SHAP is computationally heavy)
    print("Sampling 10,000 background rows...")
    X_background = df_test.sample(n=10000, random_state=42)[feature_cols]
    
    # 3. Compute SHAP values
    print("Computing TreeExplainer SHAP values (this may take a minute)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    print(f"Done in {time.time()-t0:.1f}s")
    
    # 4. Calculate Global Importance (Mean Absolute SHAP)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    df_imp = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    })
    df_imp = df_imp.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    df_imp['rank'] = df_imp.index + 1
    
    # Add metadata
    df_imp.insert(0, "horizon", f"t+{horizon}")
    df_imp.insert(0, "disease", disease)
    
    # 5. Save to the exact path expected by data_manager.py
    out_path = EXPLAIN_DIR / f"global_shap_importance_{disease}_lead{horizon}.csv"
    df_imp.to_csv(out_path, index=False)
    
    print(f"\nTop 5 Drivers of Outbreaks for {disease.title()} (t+{horizon}):")
    for _, row in df_imp.head(5).iterrows():
        print(f"  {int(row['rank'])}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        
    print(f"\nSaved Global SHAP to: {out_path}")
    print("The Dashboard Analytics Panel will now display these genuine environmental drivers!")

if __name__ == "__main__":
    calculate_env_shap("dengue", 1)
