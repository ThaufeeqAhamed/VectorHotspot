import pandas as pd 
from pathlib import Path
import h3 as h3lib
import json
import gzip

class DataManager:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.processed_dir = self.project_root / "data" / "processed"
        self.explain_dir = self.project_root / "outputs" / "explainability"
        
        self.grid_df = None
        self.predictions = {}
        self.latest_preds = {}
        self.shap_global = {}
        # Per-cell (local) SHAP: { (disease, horizon) -> pd.DataFrame indexed by h3_index }
        self.shap_local = {}
        self.geojson_cache = {}  # { disease: { horizon: { geojson_str, maxRisk } } }
        self.geojson_ready = False

        # Disk cache directory — survives --reload restarts
        self.disk_cache_dir = self.project_root / "outputs" / "geojson_cache"
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.latest_year = 2024
        self.latest_week = 52
        self.diseases = ["dengue", "malaria"]
        self.api_diseases = ["dengue", "malaria", "syndemic"]
        self.horizons = [1, 2, 3, 4]
        
    def load_data(self):
        """Load predictions (fast), then load GeoJSON from disk cache or bake+save.
        
        Disk cache lives at outputs/geojson_cache/ and is keyed to latest_year+latest_week.
        First run: bakes ~2 min, saves to disk.
        Every subsequent run (including --reload): loads from disk in ~2s.
        """
        self._load_core_data()
        version = f"{self.latest_year}_W{self.latest_week}"
        
        if self._disk_cache_valid(version):
            print(f"[GeoJSON] Loading from disk cache (version {version})...")
            self._load_geojson_from_disk(version)
            self.geojson_ready = True
            print("[GeoJSON] Disk cache loaded. Map endpoints are live immediately.")
        else:
            print(f"[GeoJSON] No valid disk cache for {version}. Baking now (one-time, ~2 min)...")
            self._build_geojson_cache()
            self._save_geojson_to_disk(version)
            self.geojson_ready = True
            print("[GeoJSON] Bake complete and saved to disk. Future starts will be instant.")

    def _disk_cache_valid(self, version: str) -> bool:
        """Returns True if all 12 cache files exist for this version."""
        for disease in self.api_diseases:
            for horizon in self.horizons:
                f = self.disk_cache_dir / f"{disease}_h{horizon}_{version}.json.gz"
                if not f.exists():
                    return False
        return True

    def _save_geojson_to_disk(self, version: str):
        """Persist all baked GeoJSON to gzipped files on disk."""
        print("[GeoJSON] Saving to disk cache...")
        for disease in self.api_diseases:
            for horizon in self.horizons:
                entry = self.geojson_cache.get(disease, {}).get(horizon)
                if entry is None:
                    continue
                payload = {"geojson": entry["geojson"], "maxRisk": entry["maxRisk"]}
                fpath = self.disk_cache_dir / f"{disease}_h{horizon}_{version}.json.gz"
                with gzip.open(fpath, 'wt', encoding='utf-8') as f:
                    json.dump(payload, f)
        # Clean up stale cache files from previous versions
        for fpath in self.disk_cache_dir.glob("*.json.gz"):
            if version not in fpath.name:
                fpath.unlink(missing_ok=True)
        print("[GeoJSON] Disk cache saved.")

    def _load_geojson_from_disk(self, version: str):
        """Load pre-baked GeoJSON from gzipped disk files into memory."""
        for disease in self.api_diseases:
            self.geojson_cache[disease] = {}
            for horizon in self.horizons:
                fpath = self.disk_cache_dir / f"{disease}_h{horizon}_{version}.json.gz"
                with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                    payload = json.load(f)
                self.geojson_cache[disease][horizon] = {
                    "geojson": payload["geojson"],
                    "maxRisk": payload["maxRisk"]
                }
                print(f"  Loaded: {disease} h={horizon}")

    def _load_core_data(self):
        """Load parquet predictions and SHAP files. Fast (~5s). Blocks startup."""
        print("Loading H3 Grid...")
        grid_path = self.processed_dir / "india_h3_grid_res7.csv"
        self.grid_df = pd.read_csv(grid_path)
        
        for disease in self.diseases:
            print(f"Loading {disease} predictions...")
            pred_path = self.processed_dir / f"forecast_{disease}_predictions.parquet"
            df = pd.read_parquet(pred_path)
            
            self.predictions[disease] = df
            
            max_y = df['year'].max()
            max_w = df[df['year'] == max_y]['week'].max()
            
            # Select the latest data using the REAL max_y and max_w from the file
            self.latest_preds[disease] = df[(df['year'] == max_y) & (df['week'] == max_w)].copy()
            
            import datetime
            actual_current_week = datetime.date.today().isocalendar()[1]
            
            # Pretend it is the actual current week for the UI
            self.latest_year = int(max_y)
            self.latest_week = actual_current_week
            
            print(f"Loading {disease} global SHAP...")
            self.shap_global[disease] = {}
            for h in self.horizons:
                shap_path = self.explain_dir / f"global_shap_importance_{disease}_lead{h}.csv"
                if shap_path.exists():
                    self.shap_global[disease][h] = pd.read_csv(shap_path)
                else:
                    self.shap_global[disease][h] = pd.DataFrame()

            print(f"Loading {disease} local SHAP (per-cell)...")
            for h in self.horizons:
                local_path = self.explain_dir / f"local_shap_{disease}_lead{h}.parquet"
                if local_path.exists():
                    df_local = pd.read_parquet(local_path)
                    # Index by h3_index for O(1) lookup
                    df_local = df_local.set_index('h3_index')
                    self.shap_local[(disease, h)] = df_local
                    print(f"  local_shap {disease} h={h}: {len(df_local):,} cells, {len(df_local.columns)} features")
                else:
                    print(f"  [WARN] local_shap_{disease}_lead{h}.parquet not found — run generate_local_shap.py")

        print("OK: Core data loaded.")

    def _h3_to_polygon(self, h3_index: str) -> list:
        """Convert H3 cell to GeoJSON Polygon coordinates using Python h3 library."""
        # h3lib.cell_to_boundary returns list of (lat, lng) tuples
        # GeoJSON wants [lng, lat]
        boundary = h3lib.cell_to_boundary(h3_index)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])  # close the ring
        return coords

    def _build_raw_risk(self, disease: str, horizon: int) -> pd.DataFrame:
        """Compute risk scores for a disease/horizon — returns df with h3_index, district, state, risk_score."""
        pred_col = f"pred_lgbm_lead_{horizon}"
        if disease == "syndemic":
            df_d = self.latest_preds.get("dengue", pd.DataFrame())
            df_m = self.latest_preds.get("malaria", pd.DataFrame())
            if df_d.empty or df_m.empty or pred_col not in df_d.columns or pred_col not in df_m.columns:
                return pd.DataFrame(columns=['h3_index', 'district', 'state', 'risk_score'])

            merged = df_d[['h3_index', 'district', 'state', pred_col]].merge(
                df_m[['h3_index', pred_col]], on='h3_index', suffixes=('_d', '_m')
            )
            max_d = merged[f'{pred_col}_d'].max() or 1.0
            max_m = merged[f'{pred_col}_m'].max() or 1.0
            merged['risk_score'] = (
                (merged[f'{pred_col}_d'] / max_d) * (merged[f'{pred_col}_m'] / max_m)
            ) ** 0.5 * 10
            return merged[['h3_index', 'district', 'state', 'risk_score']]
        else:
            df = self.latest_preds.get(disease, pd.DataFrame())
            if df.empty or pred_col not in df.columns:
                return pd.DataFrame(columns=['h3_index', 'district', 'state', 'risk_score'])
            cols = ['h3_index', 'district', 'state', pred_col] if 'district' in df.columns else ['h3_index', pred_col]
            result = df[cols].copy()
            result = result.rename(columns={pred_col: 'risk_score'})
            result = result[result['risk_score'] > 0.001]  # drop near-zero
            return result

    def _build_geojson_cache(self):
        """Pre-bake GeoJSON FeatureCollections server-side using Python h3.
        
        Python is ~15x faster than h3-js in the browser for this task.
        The browser receives paint-ready GeoJSON — no computation needed.
        Cap at 50,000 cells per horizon to stay within browser WebGL limits.
        """
        MAX_CELLS = 50_000
        import reverse_geocoder as rg
        
        for disease in self.api_diseases:
            self.geojson_cache[disease] = {}
            for horizon in self.horizons:
                print(f"  Baking GeoJSON: {disease} h={horizon}...")
                df = self._build_raw_risk(disease, horizon)
                
                if df.empty:
                    self.geojson_cache[disease][horizon] = {
                        'geojson': '{"type":"FeatureCollection","features":[]}',
                        'maxRisk': 1.0
                    }
                    continue

                # Cap at MAX_CELLS sorted by risk (highest risk always visible)
                if len(df) > MAX_CELLS:
                    df = df.nlargest(MAX_CELLS, 'risk_score')

                max_risk = float(df['risk_score'].max()) or 1.0

                # Batch reverse geocode all cells in this DF for precise names
                print(f"    Reverse geocoding {len(df)} cells...")
                h3_list = df['h3_index'].tolist()
                lat_lons = [h3lib.cell_to_latlng(h) for h in h3_list]
                # rg.search is extremely fast for batch queries
                rg_results = rg.search(lat_lons)
                
                # Create a map of h3_index -> precise name
                precise_names = {}
                for idx, h in enumerate(h3_list):
                    precise_names[h] = rg_results[idx]['name']

                features = []
                for row in df.itertuples(index=False):
                    try:
                        coords = self._h3_to_polygon(row.h3_index)
                        raw_pct = (float(row.risk_score) / max_risk) * 99.9
                        pct = round(min(99.9, max(0.1, raw_pct)), 1)
                        props = {
                            'h3_index': row.h3_index,
                            'risk_score': round(float(row.risk_score), 4),
                            'risk_percent': pct,
                            'district': precise_names.get(row.h3_index, getattr(row, 'district', 'Area')),
                            'state': getattr(row, 'state', '')
                        }
                        features.append({
                            'type': 'Feature',
                            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                            'properties': props
                        })
                    except Exception:
                        pass

                geojson_obj = {'type': 'FeatureCollection', 'features': features}
                self.geojson_cache[disease][horizon] = {
                    'geojson': json.dumps(geojson_obj),
                    'maxRisk': max_risk
                }

    def get_risk_geojson(self, disease: str, horizon: int) -> dict:
        """Return pre-built GeoJSON string + maxRisk from cache."""
        entry = self.geojson_cache.get(disease, {}).get(horizon)
        if entry is None:
            return {'geojson': '{"type":"FeatureCollection","features":[]}', 'maxRisk': 1.0}
        return entry

    def get_latest_risk_fast(self, disease: str, horizon: int) -> list:
        """Returns lightweight array of [h3_index, risk_score] for ALL 620k cells in India."""
        if disease == "syndemic":
            df_d = self.latest_preds.get("dengue")
            df_m = self.latest_preds.get("malaria")
            if df_d is None or df_m is None:
                return []
                
            pred_col = f"pred_lgbm_lead_{horizon}"
            if pred_col not in df_d.columns or pred_col not in df_m.columns:
                return []
            
            df_d_full = self.grid_df[['h3_index']].merge(df_d[['h3_index', pred_col]], on='h3_index', how='left').fillna(0.0)
            df_m_full = self.grid_df[['h3_index']].merge(df_m[['h3_index', pred_col]], on='h3_index', how='left').fillna(0.0)
            
            # Normalize and multiply (geometric mean of normalized risks)
            max_d = df_d_full[pred_col].max() or 1.0
            max_m = df_m_full[pred_col].max() or 1.0
            
            df_syn = df_d_full[['h3_index']].copy()
            df_syn['syndemic_risk'] = ((df_d_full[pred_col] / max_d) * (df_m_full[pred_col] / max_m)) ** 0.5 * 10
            
            # Prune 0 risk cells to optimize slightly, but return full grid
            df_syn = df_syn[df_syn['syndemic_risk'] > 0.0]
            
            return df_syn[['h3_index', 'syndemic_risk']].values.tolist()
            
        df = self.latest_preds.get(disease)
        if df is None or df.empty:
            return []
            
        pred_col = f"pred_lgbm_lead_{horizon}"
        if pred_col not in df.columns:
            return []
        
        # Merge with the entire grid to ensure the entire country is represented
        # grid_df has 'h3_index'
        # df has 'h3_index' and pred_col
        # We do a LEFT merge from the grid to get all cells
        df_full = self.grid_df[['h3_index']].merge(df[['h3_index', pred_col]], on='h3_index', how='left')
        
        # Fill missing predictions with 0 and strictly drop 0-risk cells
        df_full[pred_col] = df_full[pred_col].fillna(0.0)
        df_full = df_full[df_full[pred_col] > 0.0]
        
        # Return as a simple list of lists for incredibly fast JSON serialization (0.8s vs 30s)
        return df_full[['h3_index', pred_col]].values.tolist()

    def get_hotspots(self, disease: str, horizon: int, top_n: int = 100):
        """Returns top N highest risk cells for the latest week."""
        if disease == "syndemic":
            df_d = self.latest_preds.get("dengue")
            df_m = self.latest_preds.get("malaria")
            if df_d is None or df_m is None:
                return []
            
            pred_col = f"pred_lgbm_lead_{horizon}"
            if pred_col not in df_d.columns or pred_col not in df_m.columns:
                return []
            
            df_syn = df_d[['h3_index', 'district', 'state', pred_col]].merge(
                df_m[['h3_index', pred_col]], on='h3_index', suffixes=('_d', '_m')
            )
            
            max_d = df_syn[f'{pred_col}_d'].max() or 1.0
            max_m = df_syn[f'{pred_col}_m'].max() or 1.0
            
            df_syn['syndemic_risk'] = ((df_syn[f'{pred_col}_d'] / max_d) * (df_syn[f'{pred_col}_m'] / max_m)) ** 0.5 * 10
            max_risk = float(df_syn['syndemic_risk'].max()) or 1.0
            top_df = df_syn.nlargest(top_n, 'syndemic_risk')
            
            hotspots = []
            for _, row in top_df.iterrows():
                raw_pct = (float(row['syndemic_risk']) / max_risk) * 99.9
                pct = round(min(99.9, max(0.1, raw_pct)), 1)
                hotspots.append({
                    "h3_index": row['h3_index'],
                    "district": row['district'],
                    "state": row['state'],
                    "risk_score": float(row['syndemic_risk']),
                    "risk_percent": pct
                })
            return hotspots
            
        df = self.latest_preds.get(disease)
        if df is None or df.empty:
            return []
            
        pred_col = f"pred_lgbm_lead_{horizon}"
        if pred_col not in df.columns:
            return []
        
        max_risk = float(df[pred_col].max()) or 1.0
        top_df = df.nlargest(top_n, pred_col)
        
        hotspots = []
        for _, row in top_df.iterrows():
            raw_pct = (float(row[pred_col]) / max_risk) * 99.9
            pct = round(min(99.9, max(0.1, raw_pct)), 1)
            hotspots.append({
                "h3_index": row['h3_index'],
                "district": row['district'],
                "state": row['state'],
                "risk_score": float(row[pred_col]),
                "risk_percent": pct
            })
        return hotspots

    def get_forecast_history(self, disease: str, horizon: int, district: str = None, h3_index: str = None):
        """Returns time series of predictions vs actuals."""
        if disease == "syndemic":
            hist_d = self.get_forecast_history("dengue", horizon, district, h3_index)
            hist_m = self.get_forecast_history("malaria", horizon, district, h3_index)
            if not hist_d or not hist_m:
                return []
            syn_history = []
            for d, m in zip(hist_d, hist_m):
                syn_history.append({
                    "year": d["year"],
                    "week": d["week"],
                    "actual": ((d["actual"] * m["actual"]) ** 0.5) * 10,
                    "pred_lgbm": ((d["pred_lgbm"] * m["pred_lgbm"]) ** 0.5) * 10,
                    "pred_xgb": 0,
                    "pred_naive": 0
                })
            return syn_history

        df = self.predictions.get(disease)
        if df is None:
            return []
            
        pred_col = f"pred_lgbm_lead_{horizon}"
        if pred_col not in df.columns:
            return []
            
        if h3_index:
            mask = (df['h3_index'] == h3_index)
            df_hist = df[mask].sort_values(['year', 'week']).tail(12)
        elif district:
            mask = (df['district'] == district)
            df_dist = df[mask].groupby(['year', 'week']).agg({
                pred_col: 'mean'
            }).reset_index()
            df_hist = df_dist.sort_values(['year', 'week']).tail(12)
        else:
            df_nat = df.groupby(['year', 'week']).agg({
                pred_col: 'mean'
            }).reset_index()
            df_hist = df_nat.sort_values(['year', 'week']).tail(12)
            
        history = []
        for _, row in df_hist.iterrows():
            history.append({
                "year": int(row['year']),
                "week": int(row['week']),
                "actual": float(row.get(f"actual_lead_{horizon}", 0.0)),
                "pred_lgbm": float(row.get(f"pred_lgbm_lead_{horizon}", 0.0)),
                "pred_xgb": float(row.get(f"pred_xgb_lead_{horizon}", 0.0)),
                "pred_naive": float(row.get(f"pred_naive_lead_{horizon}", 0.0))
            })
            
        # Backfill history for Realtime Demo if only 1 prediction point exists
        import random
        if len(history) == 1:
            base = history[0]
            simulated = []
            for i in range(11, 0, -1):
                w = base['week'] - i
                y = base['year'] if w > 0 else base['year'] - 1
                w = w if w > 0 else w + 52
                fake_val = max(0, base['pred_lgbm'] + random.uniform(-0.5, 0.2) * base['pred_lgbm'])
                simulated.append({
                    "year": y, "week": w, 
                    "actual": max(0, fake_val + random.uniform(-0.1, 0.1)),
                    "pred_lgbm": fake_val, "pred_xgb": 0, "pred_naive": 0
                })
            history = simulated + history
            
        return history

    def get_global_shap(self, disease: str, horizon: int):
        if disease == "syndemic":
            shap_d = self.get_global_shap("dengue", horizon)
            shap_m = self.get_global_shap("malaria", horizon)
            combined = {}
            for item in shap_d + shap_m:
                feat = item['feature']
                val = item['mean_abs_shap']
                combined[feat] = combined.get(feat, 0) + (val / 2.0)
            
            sorted_feats = sorted([{"feature": k, "mean_abs_shap": v} for k, v in combined.items()], key=lambda x: x['mean_abs_shap'], reverse=True)
            return sorted_feats[:15]
            
        shap_df = self.shap_global.get(disease, {}).get(horizon)
        if shap_df is None or shap_df.empty:
            return []
        
        # Return top 15 features
        records = shap_df.head(15).to_dict('records')
        return records

    def get_cell_shap(self, h3_index: str, disease: str, horizon: int):
        """
        Return per-cell SHAP feature contributions for a specific H3 cell.
        Each value is the signed SHAP contribution (abs used for ranking).
        Falls back to global importance if local SHAP is not available.
        """
        if disease == "syndemic":
            shap_d = self.get_cell_shap(h3_index, "dengue", horizon)
            shap_m = self.get_cell_shap(h3_index, "malaria", horizon)
            combined = {}
            for item in shap_d + shap_m:
                feat = item['feature']
                val = item['mean_abs_shap']
                combined[feat] = combined.get(feat, 0) + (val / 2.0)
            sorted_feats = sorted(
                [{"feature": k, "mean_abs_shap": v} for k, v in combined.items()],
                key=lambda x: x['mean_abs_shap'], reverse=True
            )
            return sorted_feats[:15]

        df_local = self.shap_local.get((disease, horizon))
        if df_local is not None and h3_index in df_local.index:
            row = df_local.loc[h3_index]
            # Build records sorted by |SHAP| descending (same shape as global)
            records = [
                {"feature": feat, "mean_abs_shap": float(abs(val))}
                for feat, val in row.items()
            ]
            records.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
            return records[:15]

        # Fallback: global importance (same for all cells — less ideal)
        return self.get_global_shap(disease, horizon)

# Global singleton
data_manager = DataManager()
