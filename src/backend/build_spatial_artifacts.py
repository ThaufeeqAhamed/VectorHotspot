"""
build_spatial_artifacts.py
==========================
Precomputes offline spatial artifacts for VectorHotspot:
1. SQLite spatial database with R*Tree index (data/processed/vectorhotspot_spatial.db)
   - Stores all 620,742 H3 Res-7 cells with full prediction values across all 4 horizons.
   - Spatial indexing via SQLite R*Tree allows viewport bounding-box queries in <10ms.
2. National Overview GeoJSON files (H3 Res-4 parent aggregation)
   - 2,030 parent hexagons covering 100% of India.
   - Pre-bakes GeoJSON for all diseases and horizons into outputs/geojson_cache/.
   - Instant national map loads with zero missing regions.
"""

import sqlite3
import pandas as pd
import numpy as np
import h3 as h3lib
import json
import gzip
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "geojson_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "vectorhotspot_spatial.db"

def main():
    print("=" * 65)
    print("VECTORHOTSPOT SPATIAL ARTIFACTS BUILDER")
    print("=" * 65)

    # 1. Load Datasets
    print("\n[1/4] Loading H3 grid and predictions...")
    t0 = time.time()
    grid_csv = PROCESSED_DIR / "india_h3_grid_res7.csv"
    dengue_pq = PROCESSED_DIR / "forecast_dengue_predictions.parquet"
    malaria_pq = PROCESSED_DIR / "forecast_malaria_predictions.parquet"

    df_grid = pd.read_csv(grid_csv)
    df_d = pd.read_parquet(dengue_pq)
    df_m = pd.read_parquet(malaria_pq)
    print(f"  Loaded {len(df_grid):,} grid cells in {time.time()-t0:.2f}s.")

    # Prepare combined dataframe
    df = df_grid[['h3_index', 'center_lat', 'center_lon', 'district', 'state']].copy()
    for h in [1, 2, 3, 4]:
        df[f'd_h{h}'] = df_d[f'pred_lgbm_lead_{h}'].values
        df[f'm_h{h}'] = df_m[f'pred_lgbm_lead_{h}'].values
        
        # Calculate syndemic risk (geometric mean normalized * 10)
        max_d = float(df[f'd_h{h}'].max()) or 1.0
        max_m = float(df[f'm_h{h}'].max()) or 1.0
        syn = ((df[f'd_h{h}'] / max_d) * (df[f'm_h{h}'] / max_m)) ** 0.5 * 10
        df[f's_h{h}'] = syn.values

    # Determine data version
    import datetime
    year = int(df_d['year'].iloc[0]) if 'year' in df_d.columns else 2026
    week = datetime.date.today().isocalendar()[1]
    version = f"{year}_W{week}"
    print(f"  Data version: {version}")

    # 2. Build SQLite Database with R*Tree (Skip if already built and complete)
    print("\n[2/4] Checking SQLite Spatial Database with R*Tree Index...")
    needs_build = True
    if DB_PATH.exists():
        try:
            test_conn = sqlite3.connect(DB_PATH)
            cur = test_conn.cursor()
            cur.execute("SELECT count(*) FROM cells")
            cnt = cur.fetchone()[0]
            test_conn.close()
            if cnt == len(df):
                print(f"  SQLite Spatial DB already exists and verified ({cnt:,} cells, {DB_PATH.stat().st_size / (1024*1024):.1f} MB). Skipping build.")
                needs_build = False
        except Exception:
            needs_build = True

    if needs_build:
        t0 = time.time()
        if DB_PATH.exists():
            try:
                os.remove(DB_PATH)
            except Exception:
                pass

        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")

        conn.execute("""
        CREATE TABLE cells (
            id INTEGER PRIMARY KEY,
            h3_index TEXT UNIQUE,
            district TEXT,
            state TEXT,
            lat REAL,
            lon REAL,
            d_h1 REAL, d_h2 REAL, d_h3 REAL, d_h4 REAL,
            m_h1 REAL, m_h2 REAL, m_h3 REAL, m_h4 REAL,
            s_h1 REAL, s_h2 REAL, s_h3 REAL, s_h4 REAL
        )""")

        conn.execute("""
        CREATE VIRTUAL TABLE cell_index USING rtree(
            id,
            min_lon, max_lon,
            min_lat, max_lat
        )""")

        df['id'] = np.arange(1, len(df) + 1, dtype=np.int64)

        # Insert cells into table
        print("  Inserting records into `cells` table...")
        cols = ['id', 'h3_index', 'district', 'state', 'center_lat', 'center_lon',
                'd_h1', 'd_h2', 'd_h3', 'd_h4',
                'm_h1', 'm_h2', 'm_h3', 'm_h4',
                's_h1', 's_h2', 's_h3', 's_h4']
        records = df[cols].values.tolist()
        conn.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)

        # Insert into R*Tree
        print("  Indexing bounding boxes in `cell_index` (R*Tree)...")
        rtree_data = []
        lat_vals = df['center_lat'].values
        lon_vals = df['center_lon'].values
        id_vals = df['id'].values
        for i in range(len(df)):
            rtree_data.append((
                int(id_vals[i]),
                float(lon_vals[i] - 0.016), float(lon_vals[i] + 0.016),
                float(lat_vals[i] - 0.016), float(lat_vals[i] + 0.016)
            ))
        conn.executemany("INSERT INTO cell_index VALUES (?,?,?,?,?)", rtree_data)

        conn.execute("CREATE INDEX idx_cells_h3 ON cells(h3_index)")
        conn.commit()
        conn.close()

        db_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"  SQLite Spatial DB built in {time.time()-t0:.2f}s ({db_mb:.1f} MB) at {DB_PATH.name}")
    else:
        db_mb = DB_PATH.stat().st_size / (1024 * 1024)

    # 3. Precompute H3 Res-4 Spatial Aggregates (National Overview)
    print("\n[3/4] Computing H3 Res-4 Spatial Aggregations (National Overview)...")
    t0 = time.time()
    
    # Compute Res 4 parent for all cells
    print("  Mapping Res-7 cells to Res-4 parents...")
    res4_parents = [h3lib.cell_to_parent(h, 4) for h in df['h3_index']]
    df['parent_res4'] = res4_parents
    unique_parents = df['parent_res4'].unique()
    print(f"  Total Res-4 parent hexagons covering India: {len(unique_parents):,}")

    # Compute parent polygons coordinates once
    print("  Generating boundary polygons for all Res-4 parents...")
    parent_polygons = {}
    for p in unique_parents:
        coords = [[lng, lat] for lat, lng in h3lib.cell_to_boundary(p)]
        coords.append(coords[0])  # Close ring
        parent_polygons[p] = coords

    def choose_state(states):
        vals = list(states)
        for hub in ['Delhi', 'Chandigarh', 'Puducherry', 'Goa', 'Dadra and Nagar Haveli and Daman and Diu']:
            if hub in vals:
                return hub
        return states.mode().iloc[0] if not states.empty else vals[0]

    # Group by parent to calculate statistics cleanly
    agg_spec = {
        'cell_count': ('h3_index', 'count'),
        'center_lat': ('center_lat', 'mean'),
        'center_lon': ('center_lon', 'mean'),
        'district': ('district', 'first'),
        'state': ('state', choose_state),
    }
    for d_prefix in ['d', 'm', 's']:
        for h in [1, 2, 3, 4]:
            agg_spec[f'{d_prefix}_h{h}_mean'] = (f'{d_prefix}_h{h}', 'mean')
            agg_spec[f'{d_prefix}_h{h}_max'] = (f'{d_prefix}_h{h}', 'max')

    df_agg = df.groupby('parent_res4').agg(**agg_spec).reset_index()

    # Pre-bake GeoJSON for all diseases and horizons
    print("\n[4/4] Baking National Overview GeoJSON files to disk...")
    disease_map = {'dengue': 'd', 'malaria': 'm', 'syndemic': 's'}
    records_list = df_agg.to_dict('records')
    
    for disease_name, prefix in disease_map.items():
        for h in [1, 2, 3, 4]:
            mean_key = f'{prefix}_h{h}_mean'
            max_key = f'{prefix}_h{h}_max'
            
            # Global maximum for normalizing risk percent
            global_max = float(df[f'{prefix}_h{h}'].max()) or 1.0

            features = []
            for rec in records_list:
                p_hex = rec['parent_res4']
                raw_mean = rec[mean_key]
                raw_max = rec[max_key]
                coords = parent_polygons[p_hex]
                
                # Hybrid risk score: 70% mean + 30% max so isolated peak hotspots stay visible in aggregate
                hybrid_risk = 0.7 * raw_mean + 0.3 * raw_max
                raw_pct = (hybrid_risk / global_max) * 99.9
                pct = round(min(99.9, max(0.1, raw_pct)), 1)

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "h3_index": p_hex,
                        "district": rec['district'],
                        "state": rec['state'],
                        "risk_score": round(float(hybrid_risk), 4),
                        "risk_percent": pct,
                        "mean_risk": round(float(raw_mean), 4),
                        "max_risk": round(float(raw_max), 4),
                        "cell_count": int(rec['cell_count']),
                        "is_aggregate": True
                    }
                })

            geojson_obj = {"type": "FeatureCollection", "features": features}
            payload = {
                "geojson": geojson_obj,
                "maxRisk": global_max,
                "cellCount": len(features),
                "is_aggregate": True
            }

            out_file = CACHE_DIR / f"national_overview_{disease_name}_h{h}_{version}.json.gz"
            with gzip.open(out_file, 'wt', encoding='utf-8') as f:
                json.dump(payload, f)
            
            f_kb = out_file.stat().st_size / 1024
            print(f"  Saved: national_overview_{disease_name}_h{h} ({len(features)} cells, {f_kb:.1f} KB gzipped)")

    print("\n" + "=" * 65)
    print("SUCCESS: All Spatial Artifacts Built!")
    print(f"SQLite DB: {DB_PATH} ({db_mb:.1f} MB)")
    print(f"National Overview files: {CACHE_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
