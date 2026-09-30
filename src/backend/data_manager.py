import pandas as pd
from pathlib import Path

class DataManager:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.processed_dir = self.project_root / "data" / "processed"
        self.explain_dir = self.project_root / "outputs" / "explainability"
        
        self.grid_df = None
        self.predictions = {}
        self.latest_preds = {}
        self.shap_global = {}
        
        self.latest_year = 2024
        self.latest_week = 52
        self.diseases = ["dengue", "malaria"]
        self.horizons = [1, 2, 3, 4]
        
    def load_data(self):
        print("Loading H3 Grid...")
        grid_path = self.processed_dir / "india_h3_grid_res7.csv"
        self.grid_df = pd.read_csv(grid_path)
        
        for disease in self.diseases:
            print(f"Loading {disease} predictions...")
            pred_path = self.processed_dir / f"forecast_{disease}_predictions.parquet"
            # Load predictions, filtering to 2023-2024 (test set) to keep memory manageable if needed,
            # but they are already just the test set (2023-2024) in these files.
            df = pd.read_parquet(pred_path)
            self.predictions[disease] = df
            
            # Cache the latest week for fast map rendering
            self.latest_preds[disease] = df[(df['year'] == self.latest_year) & (df['week'] == self.latest_week)].copy()
            
            print(f"Loading {disease} global SHAP...")
            self.shap_global[disease] = {}
            for h in self.horizons:
                shap_path = self.explain_dir / f"global_shap_importance_{disease}_lead{h}.csv"
                if shap_path.exists():
                    self.shap_global[disease][h] = pd.read_csv(shap_path)
                else:
                    self.shap_global[disease][h] = pd.DataFrame()
                    
        print("Data Manager initialized successfully.")

    def get_latest_risk_geojson(self, disease: str, horizon: int) -> dict:
        """Returns GeoJSON point collection for the latest predictions to render as H3 grid."""
        df = self.latest_preds.get(disease)
        if df is None or df.empty:
            return {"type": "FeatureCollection", "features": []}
            
        # Join with grid to get lat/lon
        df_geo = df.merge(self.grid_df, on="h3_index", how="inner")
        
        features = []
        pred_col = f"pred_lgbm_lead_{horizon}"
        actual_col = f"actual_lead_{horizon}"
        
        # Optimize by taking only non-zero predictions or a subset if too large, 
        # but let's return all to render the full map. We will drop zero-risk cells to save bandwidth.
        df_geo = df_geo[df_geo[pred_col] > 1e-4]
        
        for _, row in df_geo.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['center_lon'], row['center_lat']] # GeoJSON is [lon, lat]
                },
                "properties": {
                    "h3_index": row['h3_index'],
                    "district": row['district_x'], # From predictions df
                    "state": row['state_x'],
                    "risk_score": float(row[pred_col]),
                    "actual": float(row[actual_col]) if pd.notnull(row[actual_col]) else 0.0
                }
            }
            features.append(feature)
            
        return {"type": "FeatureCollection", "features": features}

    def get_hotspots(self, disease: str, horizon: int, top_n: int = 100):
        """Returns top N highest risk cells for the latest week."""
        df = self.latest_preds.get(disease)
        if df is None or df.empty:
            return []
            
        pred_col = f"pred_lgbm_lead_{horizon}"
        top_df = df.nlargest(top_n, pred_col)
        
        hotspots = []
        for _, row in top_df.iterrows():
            hotspots.append({
                "h3_index": row['h3_index'],
                "district": row['district'],
                "state": row['state'],
                "risk_score": float(row[pred_col])
            })
        return hotspots

    def get_forecast_history(self, disease: str, horizon: int, district: str = None, h3_index: str = None):
        """Returns time series of predictions vs actuals."""
        df = self.predictions.get(disease)
        if df is None:
            return []
            
        if h3_index:
            mask = (df['h3_index'] == h3_index)
            # Take last 12 weeks
            df_hist = df[mask].sort_values(['year', 'week']).tail(12)
        elif district:
            mask = (df['district'] == district)
            # Aggregate by district
            df_dist = df[mask].groupby(['year', 'week']).agg({
                f"actual_lead_{horizon}": 'mean',
                f"pred_lgbm_lead_{horizon}": 'mean',
                f"pred_xgb_lead_{horizon}": 'mean',
                f"pred_naive_lead_{horizon}": 'mean'
            }).reset_index()
            df_hist = df_dist.sort_values(['year', 'week']).tail(12)
        else:
            # National aggregation
            df_nat = df.groupby(['year', 'week']).agg({
                f"actual_lead_{horizon}": 'mean',
                f"pred_lgbm_lead_{horizon}": 'mean',
                f"pred_xgb_lead_{horizon}": 'mean',
                f"pred_naive_lead_{horizon}": 'mean'
            }).reset_index()
            df_hist = df_nat.sort_values(['year', 'week']).tail(12)
            
        history = []
        for _, row in df_hist.iterrows():
            history.append({
                "year": int(row['year']),
                "week": int(row['week']),
                "actual": float(row[f"actual_lead_{horizon}"]),
                "pred_lgbm": float(row[f"pred_lgbm_lead_{horizon}"]),
                "pred_xgb": float(row[f"pred_xgb_lead_{horizon}"]),
                "pred_naive": float(row[f"pred_naive_lead_{horizon}"])
            })
        return history

    def get_global_shap(self, disease: str, horizon: int):
        shap_df = self.shap_global.get(disease, {}).get(horizon)
        if shap_df is None or shap_df.empty:
            return []
        
        # Return top 15 features
        records = shap_df.head(15).to_dict('records')
        return records

# Global singleton
data_manager = DataManager()
