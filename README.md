# VectorHotspot — Complete Deep-Dive Analysis

> **Every phase, every file, every model, every formula — explained from start to finish.**

---

## TABLE OF CONTENTS

1. [Project Identity & Research Question](#1-project-identity--research-question)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Structure Map](#3-repository-structure-map)
4. [Phase 1 — Data Layer & Sources](#4-phase-1--data-layer--sources)
5. [Phase 2 — Spatial Grid Generation (H3)](#5-phase-2--spatial-grid-generation-h3)
6. [Phase 3 — Weather Data Pipeline (IMD)](#6-phase-3--weather-data-pipeline-imd)
7. [Phase 4 — Population Data (WorldPop)](#7-phase-4--population-data-worldpop)
8. [Phase 5 — Disease Disaggregation](#8-phase-5--disease-disaggregation)
9. [Phase 6 — Feature Engineering (The 45-Feature Matrix)](#9-phase-6--feature-engineering-the-45-feature-matrix)
10. [Phase 7 — Data Splits & Leakage Prevention](#10-phase-7--data-splits--leakage-prevention)
11. [Phase 8 — ML Model Training (LightGBM, XGBoost, RF)](#11-phase-8--ml-model-training-lightgbm-xgboost-rf)
12. [Phase 9 — Hotspot Detection (Getis-Ord Gi*)](#12-phase-9--hotspot-detection-getis-ord-gi)
13. [Phase 10 — Explainability (SHAP)](#13-phase-10--explainability-shap)
14. [Phase 11 — Evaluation (Ablation, Lead Time, Validation)](#14-phase-11--evaluation-ablation-lead-time-validation)
15. [Phase 12 — Backend API (FastAPI)](#15-phase-12--backend-api-fastapi)
16. [Phase 13 — Frontend Dashboard (React + MapLibre)](#16-phase-13--frontend-dashboard-react--maplibre)
17. [Saved Model Artifacts](#17-saved-model-artifacts)
18. [Output Directory Structure](#18-output-directory-structure)
19. [End-to-End Data Flow Diagram](#19-end-to-end-data-flow-diagram)
20. [Key Scientific Findings](#20-key-scientific-findings)
21. [Plan vs Implementation Gap Analysis](#21-plan-vs-implementation-gap-analysis)

---

## 1. Project Identity & Research Question

### Name
**VectorHotspot**

### One-line Description
> AI-based dengue & malaria spatio-temporal hotspot prediction and early warning system for India, operating at H3 Resolution 7 hexagonal cells (~5.16 km² each) with 1–4 week forecast horizons.

### Core Research Question
> *Can a spatio-temporal machine-learning model use historical disease records, meteorological data, environmental/satellite layers, and population data to forecast **where** future dengue/malaria risk will concentrate — and detect it statistically significant **before** it appears in observed case counts?*

### 7 Main Objectives
| # | Objective | Status |
|---|-----------|--------|
| 1 | Build H3 × Week spatio-temporal dataset | ✅ |
| 2 | Forecast risk +1 to +4 weeks ahead | ✅ |
| 3 | Detect spatial hotspots using Getis-Ord Gi* | ✅ |
| 4 | Explain predictions using SHAP | ✅ |
| 5 | Quantify prediction uncertainty | 🟡 |
| 6 | Validate early-warning capability | ✅ |
| 7 | Build interactive React + MapLibre dashboard | ✅ |

---

## 2. High-Level Architecture

```
                          VECTORHOTSPOT SYSTEM
                                   │
       ┌───────────────────────────┴──────────────────────────┐
       │                                                        │
       ▼                                                        ▼
  DATA PIPELINE                                        USER INTERFACE
       │                                                        │
 ┌─────▼──────────────────┐                    ┌───────────────▼──────────┐
 │  Raw Data Sources       │                    │  React + MapLibre GL     │
 │  • IMD Weather (2000–24)│                    │  • Dark/Light glassmorphic│
 │  • Malaria district CSV │                    │  • IndexedDB GeoJSON cache│
 │  • Dengue weekly parquet│                    │  • HotspotSidebar         │
 │  • WorldPop TIFFs       │                    │  • AnalyticsPanel (SHAP)  │
 │  • NDVI/JRC Water/Cover │                    │  • 1-4 week slider        │
 └─────┬──────────────────┘                    └───────────────┬──────────┘
       │                                                        │
       ▼                                                        │
 ┌─────────────────────────┐                                    │
 │  Data Processing        │                                    │
 │  • H3 Grid (res 7)      │                                    │
 │  • District → H3 join   │                                    │
 │  • Weather aggregation  │                                    │
 │  • Population density   │                                    │
 │  • Disaggregation       │                                    │
 └─────┬───────────────────┘                                    │
       │                                                        │
       ▼                                                        │
 ┌─────────────────────────┐     FastAPI REST endpoints         │
 │  Feature Engineering    │◄─────────────────────────────────►│
 │  45+ features per cell  │      GET /api/geojson/all          │
 │  per week (H3 × Week)   │      GET /api/hotspots             │
 └─────┬───────────────────┘      GET /api/cell/{id}/shap       │
       │                                                         │
       ▼                                                         │
 ┌─────────────────────────┐                                     │
 │  ML Models              │                                     │
 │  • LightGBM (8 models)  │──────────────────────────────────►│
 │  • XGBoost  (8 models)  │  Predictions → forecast_*.parquet  │
 │  • RF Baseline(8 models)│                                     │
 └─────┬───────────────────┘                                     │
       │                                                         │
       ▼                                                         │
 ┌─────────────────────────┐                                     │
 │  Post-Processing        │                                     │
 │  • Getis-Ord Gi* stats  │                                     │
 │  • Hotspot taxonomy     │                                     │
 │  • SHAP calculation     │                                     │
 │  • Lead time analysis   │                                     │
 └─────────────────────────┘                                     │
```

---

## 3. Repository Structure Map

```
VectorHotspot/
├── data/
│   ├── boundaries/
│   │   └── india_districts_clean.geojson       ← 724 Indian districts, WGS84
│   ├── raw/
│   │   ├── bhopal_wards.geojson                ← Ward-level sub-district geo
│   │   ├── checkpoints/                        ← Download resume state
│   │   ├── jrc_water/                          ← JRC Global Surface Water (ESA)
│   │   ├── ndvi/                               ← NDVI satellite data
│   │   ├── population/                         ← WorldPop 1km GeoTIFFs (5 years)
│   │   └── worldcover/                         ← ESA WorldCover land use
│   └── processed/
│       ├── india_h3_grid_res7.csv              ← ~620k hexagons with lat/lon/district/state
│       ├── h3_adjacency_res7.npz               ← Sparse neighbor matrix
│       ├── imd_district_weekly_weather_2000_2024.csv   ← Original IMD data
│       ├── imd_district_weekly_weather_2000_2024_FIXED.csv ← Temperature-fixed version
│       ├── malaria_district_2000_2024.csv      ← District-year malaria case counts
│       ├── district_population_2000_2020.csv   ← WorldPop aggregated to districts
│       ├── dengue_hex_annual.csv               ← Dengue at H3, annual (~548 MB)
│       ├── malaria_hex_annual.csv              ← Malaria at H3, annual (~532 MB)
│       ├── dengue_hex_weekly.parquet/          ← Directory of weekly H3 dengue
│       ├── malaria_hex_weekly.parquet/         ← Directory of weekly H3 malaria
│       ├── features_dengue_train.parquet       ← TRAIN split (~1.96 GB)
│       ├── features_dengue_val.parquet         ← VAL split (~444 MB)
│       ├── features_dengue_test.parquet        ← TEST split (~444 MB)
│       ├── features_malaria_train.parquet      ← TRAIN split (~2.2 GB)
│       ├── features_malaria_val.parquet        ← VAL split (~290 MB)
│       ├── features_malaria_test.parquet       ← TEST split (~250 MB)
│       ├── forecast_dengue_predictions.parquet ← Model predictions for dengue (~22 MB)
│       └── forecast_malaria_predictions.parquet← Model predictions for malaria (~25 MB)
│
├── models/                                     ← 24 trained .joblib model files
│   ├── lgbm_env_dengue_lead{1-4}.joblib        ← LightGBM, environment-only, dengue
│   ├── lgbm_env_malaria_lead{1-4}.joblib       ← LightGBM, environment-only, malaria
│   ├── xgb_env_dengue_lead{1-4}.joblib         ← XGBoost, environment-only, dengue
│   ├── xgb_env_malaria_lead{1-4}.joblib        ← XGBoost, environment-only, malaria
│   ├── rf_baseline_dengue_lead{1-4}.joblib     ← Random Forest baseline, dengue
│   └── rf_baseline_malaria_lead{1-4}.joblib    ← Random Forest baseline, malaria
│
├── outputs/
│   ├── hotspots/                               ← Hotspot detection outputs (12 parquets)
│   ├── explainability/                         ← SHAP value CSVs and importance CSVs
│   ├── geojson_cache/                          ← Server-side pre-baked GeoJSON (gzipped)
│   ├── metrics/                                ← Validation metric CSVs
│   ├── tables/                                 ← Model evaluation + ablation tables
│   ├── reports/                                ← Human-readable reports
│   └── alerts/                                 ← Generated alert outputs
│
└── src/
    ├── data_prep/                              ← 7 data ingestion + processing scripts
    ├── disaggregation/                         ← District → H3 downscaling
    ├── models/                                 ← 6 model training scripts
    ├── evaluation/                             ← 3 evaluation scripts + __init__.py
    ├── explainability/                         ← 3 SHAP calculation scripts + __init__.py
    ├── backend/                                ← FastAPI server (3 files)
    └── frontend/                               ← React/Vite dashboard
```

---

## 4. Phase 1 — Data Layer & Sources

### 4.1 Disease Data

#### Dengue
- **Source**: Indian epidemiological surveillance (NVBDCP or equivalent)
- **Format**: Weekly district-level confirmed dengue cases
- **Files**:
  - `dengue_hex_annual.csv` — Annual dengue cases mapped to H3 hexagons (~548 MB)
  - `dengue_hex_weekly.parquet/` — Same data broken into weekly partitions
- **Coverage**: 2000–2024, all-India districts

#### Malaria
- **Source**: National Vector Borne Disease Control Programme (NVBDCP)
- **Format**: District-level annual case records
- **Files**:
  - `malaria_district_2000_2024.csv` — Raw district-year malaria case counts (~529 KB)
  - `malaria_hex_annual.csv` — H3-disaggregated malaria (~532 MB)
  - `malaria_hex_weekly.parquet/` — Weekly partitions
- **Coverage**: 2000–2024, all-India

### 4.2 Weather Data
- **Source**: **India Meteorological Department (IMD)** via `imdlib` Python package
- **Variables**: Daily rainfall (mm), Tmax (°C), Tmin (°C)
- **Grid Resolution**: 0.25° × 0.25° (~25 km) for rainfall; 1° × 1° for temperature
- **Aggregation**: District-level weekly (sum for rainfall, mean for temperature)
- **File**: `imd_district_weekly_weather_2000_2024_FIXED.csv`
- **Coverage**: 2000–2024

### 4.3 Population Data
- **Source**: **WorldPop** 1km resolution GeoTIFF rasters (5 snapshots)
- **Years**: 2000, 2005, 2010, 2015, 2020
- **File template**: `ind_ppp_{year}_1km_Aggregated.tif`
- **Output**: `district_population_2000_2020.csv` — population totals per (state, district, year)

### 4.4 Environmental/Satellite Data
- **NDVI** (Normalized Difference Vegetation Index) — `data/raw/ndvi/` — vegetation proxy for mosquito breeding
- **JRC Global Surface Water** — `data/raw/jrc_water/` — permanent/seasonal water body occurrence
- **ESA WorldCover** — `data/raw/worldcover/` — land-use type (forest, urban, shrub, built-up)

### 4.5 Geographic Boundaries
- **File**: `data/boundaries/india_districts_clean.geojson`
- **Content**: 724 Indian administrative districts, WGS84 projection
- **Extra**: `data/raw/bhopal_wards.geojson` — sub-district ward boundaries for Bhopal (prototype)

---

## 5. Phase 2 — Spatial Grid Generation (H3)

### File: [`src/data_prep/generate_h3_grid.py`](file:///w:/MiniProject5/VectorHotspot/src/data_prep/generate_h3_grid.py)

### Why H3?
H3 is Uber's hierarchical hexagonal geospatial indexing system. Hexagons are chosen because:
- No vertex bias (equal distance from center to all neighbors)
- Consistent area at same resolution
- Clean spatial aggregation/disaggregation across resolutions

### Why Resolution 7?
| Resolution | Avg Area | India Cell Count |
|---|---|---|
| 6 | ~36 km² | ~88,000 |
| **7** | **~5.16 km²** | **~620,000** |
| 8 | ~0.74 km² | ~4,000,000 |

**Rationale (from code docstring):**
- Coarser (res 6) doesn't beat district-level granularity
- Finer (res 8+) outpaces what covariates can actually distinguish (rainfall ~25km, temperature ~100km)
- Resolution 7 sits at the **practical ceiling** of what the input data can support

### Algorithm (Step-by-Step)

```
Step 1: Load india_districts_clean.geojson → 724 district polygons (WGS84)
Step 2: Compute India-wide geometric union → single Polygon/MultiPolygon
Step 3: polygon_to_h3_set(india_union, resolution=7)
         → iterate each polygon/hole
         → h3.LatLngPoly(exterior_coords, *holes)
         → h3.polygon_to_cells(h3poly, resolution=7)
         → set of H3 cell IDs covering all of India
Step 4: Compute center (lat, lon) for each cell: h3.cell_to_latlng(cell)
Step 5: Spatial join: gpd.sjoin(hex_gdf, districts, predicate="within")
         → assigns (district, state) to each hexagon
Step 6: Fallback for unmatched hexagons (coastal edge cases):
         → find nearest district centroid
Step 7: Deduplication: drop_duplicates(subset="h3_index", keep="first")
Step 8: District coverage guarantee:
         → for any district with 0 hexagons (tiny UTs):
         → snap district centroid to nearest H3 cell
         → add that cell to the grid
Step 9: Save to data/processed/india_h3_grid_res7.csv
```

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | string | Unique H3 cell ID (e.g., `872a1072fffffff`) |
| `center_lat` | float | Cell centroid latitude |
| `center_lon` | float | Cell centroid longitude |
| `district` | string | Assigned Indian district name |
| `state` | string | Assigned Indian state name |

### Adjacency Matrix
- **File**: `data/processed/h3_adjacency_res7.npz`
- **Format**: Sparse matrix (scipy compressed sparse row)
- **Purpose**: Encodes which H3 cells are spatial neighbors
- **Usage**: Used during feature engineering to compute `neighbor_cases` features (spatial lag)

---

## 6. Phase 3 — Weather Data Pipeline (IMD)

### File: [`src/data_prep/fetch_imd_weather.py`](file:///w:/MiniProject5/VectorHotspot/src/data_prep/fetch_imd_weather.py)

### Why IMD?
IMD (India Meteorological Department) provides the most authoritative gridded climate data for India. However:
- Their portal uses POST-form submissions (not direct download links)
- `imdlib` wraps these POST requests transparently
- The script downloads ONE year at a time with retries (max 5 attempts, 45-second backoff) because IMD servers are flaky

### Data Variables
| Variable | Resolution | Source Product |
|---|---|---|
| Rainfall | 0.25° × 0.25° daily | IMD gridded rainfall |
| Tmax | 1° × 1° daily | IMD gridded max temperature |
| Tmin | 1° × 1° daily | IMD gridded min temperature |

### Pipeline Detail

```
Step 1: Download loops 2000–2024 per variable (rain, tmax, tmin)
        → imd.get_data(variable, year, year, fn_format='yearwise', file_dir=folder)
        → Retry logic: up to 5 attempts with 45s sleep between
        → Skip if year already downloaded (year_already_downloaded check)

Step 2: Open all years as xarray datasets
        rain_data = imd.open_data('rain', 2000, 2024, 'yearwise', ...)
        → rain_ds = rain_data.get_xarray()   [shape: time × lat × lon]

Step 3: For each day t in each dataset:
        day_slice = ds[varname].isel(time=i).values  [2D lat × lon array]
        
        if lats ascending: np.flipud(day_slice)  # ensure north-up for rasterio
        
        Sentinel masking:
          Rain: mask -999.0 → NaN
          Tmax/Tmin: mask -999.0, 99.9, -99.9 → NaN
          (avoids treating real values near sentinel numbers as missing)
        
        zonal_stats(districts, day_slice, affine=transform,
                    stats=['mean'], nodata=NaN, all_touched=True)
        → CRITICAL: all_touched=True prevents small districts from getting
          entirely null stats when no single pixel center falls inside them

Step 4: Aggregate daily → weekly:
        Year = date.dt.isocalendar().year   (ISO week year)
        Week = date.dt.isocalendar().week   (ISO week number)
        
        Rainfall_mm = weekly SUM (total precipitation over the week)
        Tmax_C      = weekly MEAN
        Tmin_C      = weekly MEAN

Step 5: Output: imd_district_weekly_weather_2000_2024.csv
        Columns: State, District, Year, Week, Rainfall_mm, Tmax_C, Tmin_C
```

### Temperature Fix
A separate script `recompute_temperature_fast.py` fixes known temperature data quality issues, producing the `_FIXED.csv` version that's actually used in modeling.

### Live Weather Fetching
**File**: `src/data_prep/fetch_live_weather.py` — fetches current-week weather for real-time inference.

---

## 7. Phase 4 — Population Data (WorldPop)

### File: [`src/data_prep/compute_district_population.py`](file:///w:/MiniProject5/VectorHotspot/src/data_prep/compute_district_population.py)

### Source
WorldPop 1km resolution GeoTIFFs for India: `ind_ppp_{year}_1km_Aggregated.tif`
- 5 census years: 2000, 2005, 2010, 2015, 2020
- Each pixel represents **total population** in a 1km² grid cell

### Algorithm (Pure-Python, No GDAL)

Since no GDAL/rasterio was available at time of creation, a custom pure-Python implementation was used:

```
For each year in [2000, 2005, 2010, 2015, 2020]:
    Open TIF → read as numpy array via PIL/tifffile
    Read GeoTransform: pixel scale (lon/px, lat/px) and origin (lat, lon)
    Replace nodata pixels with 0.0

    For each district polygon (from india_districts_clean.geojson):
        get_rings(geometry) → list of (exterior, holes) polygon ring pairs
        
        For each ring pair:
            Convert bounding box corners to pixel row/col indices
            Create meshgrid of pixel centers (lon_grid, lat_grid)
            
            Point-in-polygon test:
              ext_path = matplotlib.path.Path(exterior)
              mask = ext_path.contains_points(pixel_centers)
              
              For each hole:
                hole_path = matplotlib.path.Path(hole)
                mask &= ~hole_path.contains_points(pixel_centers)
            
            total_pop += arr[mask].sum()  # sum all pixel values inside polygon
        
        Record: (State, District, Year, Population)

Output: district_population_2000_2020.csv
Columns: State, District, Year, Population
```

### Population Interpolation (in Feature Engineering)
For years between the 5 census snapshots, population is **linearly interpolated** during feature construction to provide a smooth year-by-year estimate.

---

## 8. Phase 5 — Disease Disaggregation

### File: [`src/disaggregation/disaggregation_prototype.py`](file:///w:/MiniProject5/VectorHotspot/src/disaggregation/disaggregation_prototype.py)

### Problem
Disease data is collected at **district level** but the model needs it at **H3 hexagon level**. The disaggregation must be **mass-preserving** — i.e., the sum of hex-level estimates must equal the district total.

### Method: Poisson Regression with Aggregation Constraint
Inspired by the **Malaria Atlas Project's** `disaggregation` R package:

```
TRUE MODEL (hidden in reality):
  log(E[Y_i]) = β₀ + β_pop × log(pop_i) + β_ndvi × ndvi_i + β_rain × rain_i
  Y_i ~ Poisson(λ_i)   where i = fine cell

CONSTRAINT:
  For each district d: Σᵢ∈d Y_i = Y_d (observed district total)

ESTIMATION:
  Objective function: maximize Poisson log-likelihood subject to constraint
  scipy.optimize.minimize(neg_log_likelihood, beta_init, method='L-BFGS-B')
  
  Predicted cell-level count:
  ŷ_i = Y_d × (exp(Xᵢβ̂) / Σⱼ∈d exp(Xⱼβ̂))
```

This is a **Dasymetric mapping** approach where covariates (population, NDVI, rainfall) determine how district-level counts are distributed across fine cells, preserving the district total.

### Synthetic Proof-of-Concept Parameters
```python
true_beta0     = -11.0
true_beta_pop  = 0.00035   # more people → more reported cases
true_beta_ndvi = 2.0       # vegetation = mosquito breeding habitat  
true_beta_rain = 0.02      # rainfall = standing water for breeding
```

---

## 9. Phase 6 — Feature Engineering (The 45-Feature Matrix)

### Output Files
- `features_dengue_train.parquet` — ~1.96 GB training set
- `features_dengue_val.parquet` — ~444 MB validation
- `features_dengue_test.parquet` — ~444 MB test
- `features_malaria_train.parquet` — ~2.2 GB training set
- `features_malaria_val.parquet` — ~290 MB validation
- `features_malaria_test.parquet` — ~250 MB test

### Unit of Analysis
Each row = one **H3 hexagon × ISO week** combination.

### Feature Groups (with Detailed Explanation)

#### Group 1: Disease History Features (Temporal Lags)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cases_lag_1` | cases at week t-1 | Immediate prior activity |
| `cases_lag_2` | cases at week t-2 | Short-term trend |
| `cases_lag_3` | cases at week t-3 | 3-week memory |
| `cases_lag_4` | cases at week t-4 | Monthly memory |
| `cases_lag_8` | cases at week t-8 | Bi-monthly seasonal echo |

#### Group 2: Rolling Case Aggregates
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cases_roll_mean_4w` | mean(cases[t-4:t]) | 4-week rolling average baseline |
| `cases_roll_mean_12w` | mean(cases[t-12:t]) | 12-week quarterly baseline |
| `cases_roll_std_4w` | std(cases[t-4:t]) | Outbreak volatility indicator |
| `cases_momentum_4w` | cases[t-1] - cases[t-4] | Rate of change (acceleration) |
| `case_rate_lag_1` | cases[t-1] / population | Per-capita transmission rate |

#### Group 3: Weather Lag Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `rain_lag_1` | Rainfall_mm at t-1 | Recent rainfall → mosquito breeding |
| `rain_lag_2` | Rainfall_mm at t-2 | 2-week lag (incubation) |
| `rain_lag_4` | Rainfall_mm at t-4 | Monthly rainfall memory |
| `rain_lag_6` | Rainfall_mm at t-6 | 6-week lag for Anopheles (malaria) |
| `tmean_lag_1` | (Tmax+Tmin)/2 at t-1 | Average temperature |
| `tmin_lag_1` | Tmin at t-1 | Night temperature (vector survival) |
| `tmax_lag_1` | Tmax at t-1 | Day temperature (vector lifespan) |
| `dtr_lag_1` | Tmax - Tmin at t-1 | Day-to-night swing (vector stress) |
| *(lag_2, lag_4 variants)* | — | Same variables at 2 and 4-week lags |

#### Group 4: Rainfall Rolling Sums (Accumulated Conditions)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `rain_roll_sum_2w` | Σ Rainfall[t-2:t] | 2-week accumulated rainfall |
| `rain_roll_sum_4w` | Σ Rainfall[t-4:t] | 4-week accumulated rainfall |

#### Group 5: Seasonality (Cyclical Encoding)
| Feature | Formula | Why cyclical? |
|---------|---------|---------------|
| `sin_week` | sin(2π × week / 52) | Week 1 and week 52 are adjacent in time; cyclic encoding captures this |
| `cos_week` | cos(2π × week / 52) | Together with sin, creates a 2D circular representation of the annual cycle |
| `month` | calendar month (1–12) | Redundant with sin/cos but helps tree models |

Tree-based models don't inherently understand that `week 52 → week 1` is a small step. The sin/cos encoding turns the week number into a point on a unit circle, where neighboring weeks are close in 2D Euclidean space regardless of whether they straddle year-end.

#### Group 6: Population Features
| Feature | Description |
|---------|-------------|
| `log_population` | log(1 + district_pop_interpolated) — log-transform reduces skew |
| `pop_density` | population / hex_area (persons per km²) |

#### Group 7: Spatial Neighbor Features (Adjacency Spreading)
Using the `h3_adjacency_res7.npz` sparse neighbor matrix:
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `neighbor_cases_k1_lag1` | mean(cases_lag_1 across k=1 ring neighbors) | Disease momentum in adjacent cells |
| `neighbor_cases_k1_lag2` | mean(cases_lag_2 across k=1 ring neighbors) | Delayed spatial spread |
| `neighbor_cases_k2_lag1` | mean(cases_lag_1 across k=2 ring neighbors) | Extended neighborhood effect |

The k=1 ring contains up to 6 immediate hex neighbors; k=2 ring has up to 12. These features capture that **disease spreads spatially** — a hotspot in one cell is a risk signal for adjacent cells.

#### Group 8: Environmental / Ecological Features
| Feature | Source | Description |
|---------|--------|-------------|
| `ndvi_mean` | NDVI satellite | Vegetation greenness — proxy for mosquito habitat |
| `jrc_occurrence` | JRC Global Surface Water | % of time a cell has surface water |
| `frac_water` | ESA WorldCover | Surface water land cover fraction |
| `frac_trees` | ESA WorldCover | Tree canopy cover fraction |
| `frac_built` | ESA WorldCover | Built-up/urban area fraction |
| `frac_shrub` | ESA WorldCover | Shrubland fraction |
| `suitability_lag_1` | Derived | Composite mosquito climate suitability index |
| `suitability_lag_2` | Derived | 2-week lagged suitability |
| `suitability_lag_4` | Derived | 4-week lagged suitability |
| `elevation` | SRTM/DEM | Altitude (limits vector altitude range) |

#### Group 9: Geographic Coordinates
| Feature | Purpose |
|---------|---------|
| `center_lat` | Latitudinal position (climate zone proxy) |
| `center_lon` | Longitudinal position (regional effect) |

### Target Variables (Labels for Supervised Learning)
| Column | Definition |
|--------|-----------|
| `target_lead_1` | Disease cases at week t+1 |
| `target_lead_2` | Disease cases at week t+2 |
| `target_lead_3` | Disease cases at week t+3 |
| `target_lead_4` | Disease cases at week t+4 |

Each model is trained with a **direct multi-horizon** approach: one independent model per horizon, not iterative chained forecasting.

---

## 10. Phase 7 — Data Splits & Leakage Prevention

### Chronological Splitting (Critical Design Decision)

**WRONG approach** (data leakage):
```
Random 80/20 split → train row from week 50 predicts test row from week 10
→ Future information leaks into training
```

**CORRECT approach** (used here):
```
Past ────────────────────────────────────────────► Future
│                          │                        │
├──── TRAIN (2000–2020) ───┤── VAL (2021–2022) ─────┤── TEST (2023–2024) ──┤
```

### Additional Spatial Holdout (Not in Original Plan — Added Extra)
15% of districts are randomly selected as **spatial holdout** districts:
- Flagged: `is_spatial_holdout = True`
- These districts are **never seen during training**
- Evaluating on them tests true **geographic generalizability**
- A model that works well on spatial holdout = it learned real epidemiological patterns, not just memorized locations

### Leakage Prevention Rules
1. All lag features computed using only data at or before week `t`
2. All rolling aggregates use only weeks `t-k, ..., t-1` (never `t` or future)
3. Target `target_lead_k` uses actual cases at `t+k` — correctly shifted forward
4. No future weather data in the feature vector when predicting future cases

---

## 11. Phase 8 — ML Model Training (LightGBM, XGBoost, RF)

### Files
- [`src/models/train_env_only_models.py`](file:///w:/MiniProject5/VectorHotspot/src/models/train_env_only_models.py) — Primary training (LightGBM + XGBoost)
- [`src/models/train_rf_baseline.py`](file:///w:/MiniProject5/VectorHotspot/src/models/train_rf_baseline.py) — Random Forest baseline
- [`src/models/train_forecasting_models.py`](file:///w:/MiniProject5/VectorHotspot/src/models/train_forecasting_models.py) — Full-feature training (includes case history features)

### Training Architecture

#### Environment-Only Models (`train_env_only_models.py`)
The "environment-only" variant deliberately **excludes all case/surveillance features** and neighbor features:
```python
feature_cols = [c for c in df_train.columns 
                if c not in meta_cols 
                and 'case' not in c      # removes cases_lag_*, case_rate_*, etc.
                and 'neighbor' not in c] # removes neighbor_cases_*
```
This tests how well the environment alone predicts disease risk (useful for areas with poor surveillance data).

#### Training Sample Size
```python
n_train_sub = min(len(df_train), 1_500_000)  # cap at 1.5M rows for LightGBM/XGBoost
train_sub = df_train[~df_train['is_spatial_holdout']].sample(n_train_sub, random_state=42)
```

For Random Forest: capped at 250,000 (RF is much more memory/CPU intensive than GBTs).

### LightGBM Configuration
```python
lgbm_model = lgb.LGBMRegressor(
    objective='regression',      # Mean Squared Error loss
    n_estimators=250,            # 250 boosting rounds
    learning_rate=0.06,          # Conservative learning rate
    num_leaves=63,               # Max 63 leaves per tree (~depth-6 equivalent)
    max_depth=8,                 # Hard depth cap for regularization
    min_child_samples=50,        # Minimum samples per leaf (prevents overfitting)
    subsample=0.85,              # 85% row sampling per tree (Bagging fraction)
    colsample_bytree=0.85,       # 85% column sampling per tree (Feature fraction)
    reg_alpha=0.1,               # L1 regularization (Lasso)
    reg_lambda=1.0,              # L2 regularization (Ridge)
    random_state=42,
    n_jobs=-1,                   # Use all CPU cores
    verbose=-1                   # Suppress training output
)
```

### XGBoost Configuration
```python
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # MSE loss
    n_estimators=200,              # 200 boosting rounds
    learning_rate=0.06,
    max_depth=6,                   # Shallower than LGBM for safety
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,                 # L1
    reg_lambda=1.0,                # L2
    random_state=42,
    n_jobs=-1
)
```

### Random Forest Configuration
```python
rf_model = RandomForestRegressor(
    n_estimators=50,               # Only 50 trees (memory constraint)
    max_depth=10,                  # Limit depth for speed
    min_samples_split=50,          # Minimum samples to split a node
    n_jobs=-1,
    random_state=42
)
```

**Key difference**: Random Forest trains on `delta_train = y_train - cases_lag_1` (predicts *change from baseline*) whereas LightGBM/XGBoost trains directly on the target count.

### Evaluation Metrics (computed by `compute_metrics()`)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **R²** | 1 - SS_res/SS_tot | Fraction of variance explained |
| **MAE** | mean(|y_true - y_pred|) | Average absolute error in case counts |
| **RMSE** | √mean((y_true - y_pred)²) | Penalizes large errors more |
| **RMSLE** | √mean((log(1+y_true) - log(1+y_pred))²) | Stable for zero-inflated count data |
| **Corr** | Pearson correlation | Direction agreement |

### Baseline Models Benchmarked

**Baseline 1: Naive Persistence**
```
ŷ_{t+k} = y_t   (assume future = current)
```

**Baseline 2: Historical Seasonal Mean**
```
ŷ_{t+k}(h, w) = mean_hist(cases[h, w+k])
```
Where `h` = hexagon, `w+k` = target ISO week. A lookup table of `(district, week) → mean historical cases`.

**Baseline 3: Coarse District Model**
Aggregates to district level, trains a simple district-level model, then disaggregates predictions uniformly back to hexagons.

### Output: 24 Saved Models

```
models/
├── lgbm_env_dengue_lead1.joblib   (~1.43 MB)
├── lgbm_env_dengue_lead2.joblib   (~1.43 MB)
├── lgbm_env_dengue_lead3.joblib   (~1.45 MB)
├── lgbm_env_dengue_lead4.joblib   (~1.43 MB)
├── lgbm_env_malaria_lead1.joblib  (~1.42 MB)
...
├── xgb_env_dengue_lead1.joblib    (~918 KB)
...
├── rf_baseline_dengue_lead1.joblib (~109 KB)
...
```

Models are saved using `joblib.dump()` and loaded with `joblib.load()` at inference time.

### Time-Series Cross-Validation
**File**: `src/models/time_series_cv.py`
Implements expanding-window time-series cross-validation:
```
Fold 1: Train[2000–2015] → Val[2016]
Fold 2: Train[2000–2016] → Val[2017]
Fold 3: Train[2000–2017] → Val[2018]
...
```
Never allows future data into earlier training folds.

---

## 12. Phase 9 — Hotspot Detection (Getis-Ord Gi*)

### What is Getis-Ord Gi*?

A spatial autocorrelation statistic that tests whether high values **cluster spatially** — not just whether individual cells have high risk, but whether **neighboring high-risk cells form a statistically significant cluster**.

### Mathematical Formula

For each H3 cell `i`:

```
        Σⱼ wᵢⱼ xⱼ − X̄ Σⱼ wᵢⱼ
Gi* = ─────────────────────────────────────────
        S × √[(n Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)²) / (n−1)]
```

Where:
- `xⱼ` = predicted risk score at cell j
- `wᵢⱼ` = spatial weight (1 if j is neighbor of i, 0 otherwise)
- `X̄` = mean risk across all cells
- `S` = standard deviation of risk across all cells
- `n` = total number of cells

### Interpretation

| Gi* Z-score | p-value | Classification |
|---|---|---|
| > 2.58 | < 0.01 | **Strong Hotspot** (99% confidence) |
| > 1.96 | < 0.05 | **Moderate Hotspot** (95% confidence) |
| -1.96 to 1.96 | > 0.05 | **Not Significant** |
| < -1.96 | < 0.05 | **Coldspot** (statistically low cluster) |

The key insight: **a cell can have high predicted risk BUT still not be a significant hotspot** if it is surrounded by equally low-risk cells (it's an isolated spike, not a cluster).

### Hotspot Taxonomy
Hotspots are classified across time into behavioral categories:
- **Emerging Hotspot**: First week it becomes significant
- **Persistent Hotspot**: Significant for 3+ consecutive weeks
- **Expanding Hotspot**: Area of significance growing
- **Diminishing Hotspot**: Shrinking in area or significance
- **Non-Hotspot**: Never reached significance

### Output Files
```
outputs/hotspots/
├── dengue_hotspots_lead1.parquet       (~379 MB) — all cells, lead-1 predictions
├── dengue_hotspots_lead2.parquet       (~375 MB)
├── dengue_hotspots_lead3.parquet       (~374 MB)
├── dengue_hotspots_lead4.parquet       (~373 MB)
├── dengue_hotspots_test_2023_2024.parquet  (~230 MB) — test set hotspot flags
├── dengue_hotspot_taxonomy_2024.parquet    (~9.5 MB) — taxonomy labels
├── malaria_hotspots_lead{1-4}.parquet      ← same structure for malaria
├── malaria_hotspots_test_2023_2024.parquet
└── malaria_hotspot_taxonomy_2024.parquet
```

### Column Schema for Hotspot Parquets
| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | string | H3 cell identifier |
| `year`, `week` | int | Temporal reference |
| `risk_score` | float | Raw model predicted risk |
| `gi_zscore` | float | Getis-Ord Gi* Z-score |
| `p_value` | float | Statistical significance |
| `is_hotspot_pred_lead_k` | bool | Whether model predicted hotspot at horizon k |
| `is_hotspot_act_lead_k` | bool | Whether cell was **actually** a hotspot (ground truth) |
| `is_spatial_holdout` | bool | Whether in spatial holdout district set |

### Syndemic (Combined) Risk Formula
When both dengue and malaria predictions exist:

```python
syndemic_risk = ((dengue_risk / max_dengue) * (malaria_risk / max_malaria)) ** 0.5 * 10
```

This is a **geometric mean of normalized risks**, scaled to 0–10. The geometric mean ensures that the combined score is only high when **both** diseases are high — a cell that is 100% dengue risk but 0% malaria risk gets a syndemic score of 0 (not 50%).

---

## 13. Phase 10 — Explainability (SHAP)

### What is SHAP?
SHAP (SHapley Additive exPlanations) is a game-theory based method for explaining individual ML predictions. It answers: **"By how much did each feature push this prediction above or below the average?"**

### SHAP Value Interpretation
```
Base value (model average): 2.3 cases
+ cases_lag_1:   +0.43  (high recent cases drive prediction up)
+ rain_lag_2:    +0.21  (recent rainfall increases risk)
+ suitability:   +0.18  (good breeding conditions)
- frac_built:    -0.12  (urban area reduces mosquito habitat)
= Prediction:     3.00 cases
```

### Files

#### Raw SHAP Computations
| File | Size | Content |
|------|------|---------|
| `outputs/explainability/shap_values_dengue_lead{1-4}.csv` | ~50 MB each | Row-level SHAP values (one row per test sample, one column per feature) |
| `outputs/explainability/shap_values_malaria_lead{1-4}.csv` | ~50 MB each | Same for malaria |

#### Analysis Script: [`src/explainability/shap_analysis.py`](file:///w:/MiniProject5/VectorHotspot/src/explainability/shap_analysis.py)

```
Step 1: Read shap_values_{disease}_lead{k}.csv in chunks of 50,000 rows
Step 2: Compute mean absolute SHAP per feature:
        abs_chunk = chunk.abs()
        accum += abs_chunk.sum()
        
        mean_abs_shap[feature] = accum[feature] / n_rows

Step 3: Sort features by mean_abs_shap descending → rank 1 to N
Step 4: Save global_shap_importance_{disease}_lead{k}.csv:
        Columns: disease, horizon, feature, mean_abs_shap, rank
Step 5: Save combined global_shap_importance_all.csv
```

#### Output Global SHAP Importance CSVs
Loaded at backend startup → served via `/api/cell/{h3_id}/shap` endpoint → displayed in frontend AnalyticsPanel.

#### Environment SHAP Calculator: [`src/explainability/shap_env_calculator.py`](file:///w:/MiniProject5/VectorHotspot/src/explainability/shap_env_calculator.py)
Computes SHAP specifically for the environment-only models.

#### Uncertainty Analysis: [`src/explainability/uncertainty_analysis.py`](file:///w:/MiniProject5/VectorHotspot/src/explainability/uncertainty_analysis.py)
Quantifies prediction uncertainty using:
- Ensemble disagreement (LightGBM vs XGBoost spread)
- Confidence intervals derived from the quantile range of predictions

---

## 14. Phase 11 — Evaluation (Ablation, Lead Time, Validation)

### 14.1 Hotspot Validation

**File**: [`src/evaluation/hotspot_validator.py`](file:///w:/MiniProject5/VectorHotspot/src/evaluation/hotspot_validator.py)

Compares predicted hotspot binary labels (`is_hotspot_pred_lead_k`) against actual labels (`is_hotspot_act_lead_k`):

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all predicted hotspots, how many were real? |
| **Recall** | TP / (TP + FN) | Of all real hotspots, how many did we catch? |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean of Precision and Recall |
| **Spatial IoU** | TP / (TP+FP+FN) | `|Predicted ∩ Actual| / |Predicted ∪ Actual|` |

Where:
- TP = correctly predicted hotspot cells
- FP = predicted hotspot, actually not
- FN = actual hotspot, not predicted

### 14.2 Early Warning Lead Time Analysis

**File**: [`src/evaluation/lead_time_analysis.py`](file:///w:/MiniProject5/VectorHotspot/src/evaluation/lead_time_analysis.py)

**Question**: How many weeks in advance does the model detect a hotspot before it becomes visible in case data?

```
For each actual hotspot event (is_hotspot_act_lead_1 == 1):
  Check is_hotspot_pred_lead_1 == 1  → 1 week advance warning
  Check is_hotspot_pred_lead_2 == 1  → 2 weeks advance warning
  Check is_hotspot_pred_lead_3 == 1  → 3 weeks advance warning
  Check is_hotspot_pred_lead_4 == 1  → 4 weeks advance warning

Detection Rate @ lead k = correctly_detected / total_actual_hotspots
```

**Maximum Effective Lead**: The highest `k` at which detection rate ≥ 50%.

The script also analyzes **hotspot taxonomy persistence** from the 2024 annual taxonomy parquet, computing distribution of Emerging/Persistent/Expanding/Diminishing labels.

### 14.3 Feature Ablation Study

**File**: [`src/evaluation/ablation_study.py`](file:///w:/MiniProject5/VectorHotspot/src/evaluation/ablation_study.py)

Ablation = deliberately removing feature groups to measure their impact.

**6 Scenarios Tested:**
1. **All Features (Baseline)** — full model
2. **No Weather** — remove tmean, tmin, tmax, rain, humid, suitability features
3. **No Spatial (Neighbors)** — remove neighbor_cases_* features
4. **No Temporal (Seasonality)** — remove sin_week, cos_week, month
5. **No Ecology & Population** — remove pop, frac_*, ndvi, elevation
6. **Only Base Cases (No Exogenous)** — remove everything except cases_lag_* features

Each scenario:
1. Filters features to the allowed subset
2. Trains a fresh LightGBM model (150 trees, slightly lighter than production)
3. Reports R², MAE, RMSE on test set

### Key Ablation Findings (from README)

**Dengue is Autoregressive:**
- Full model: R² ≈ 0.89
- Only Base Cases scenario: R² ≈ 0.86
- No Weather: R² ≈ 0.88 (barely changes!)
- **Conclusion**: Dengue spreads like a wildfire — recent cases predict future cases with high fidelity. Weather adds only marginal fine-tuning.

**Malaria is Environmentally Driven:**
- Full model: R² ≈ 0.82
- No Temporal (Seasonality): R² drops to ≈ 0.68
- **Conclusion**: Malaria is strongly seasonal — the monsoon cycle, breeding season timing, and ecological baseline are critical signals. This reflects Anopheles mosquito dependence on rainfall patterns.

---

## 15. Phase 12 — Backend API (FastAPI)

### Files
- [`src/backend/main.py`](file:///w:/MiniProject5/VectorHotspot/src/backend/main.py) — API endpoints
- [`src/backend/data_manager.py`](file:///w:/MiniProject5/VectorHotspot/src/backend/data_manager.py) — Data loading & serving logic
- [`src/backend/schemas.py`](file:///w:/MiniProject5/VectorHotspot/src/backend/schemas.py) — Pydantic response models

### Startup Sequence
```
1. uvicorn starts → lifespan() runs
2. data_manager.load_data() called:
   a. _load_core_data():
      - Load india_h3_grid_res7.csv (~620k rows)
      - Load forecast_dengue_predictions.parquet → self.predictions["dengue"]
      - Load forecast_malaria_predictions.parquet → self.predictions["malaria"]
      - Extract latest_year, latest_week from max year/week in predictions
      - Load global SHAP CSVs for all 8 disease×horizon combos
   b. Check disk cache: outputs/geojson_cache/{disease}_h{horizon}_{year}_W{week}.json.gz
      - If all 12 files exist (3 diseases × 4 horizons): load from disk ~2 seconds
      - If missing: _build_geojson_cache() → ~2 minutes → save to disk
3. geojson_ready = True
```

### GeoJSON Baking Process
The most important performance optimization in the whole system:

```python
def _build_geojson_cache(self):
    MAX_CELLS = 50_000  # WebGL limit for smooth browser rendering
    
    for disease in ["dengue", "malaria", "syndemic"]:
        for horizon in [1, 2, 3, 4]:
            df = _build_raw_risk(disease, horizon)  # get risk scores
            
            # Keep only top 50k by risk score (browser can't render 620k hexagons)
            if len(df) > MAX_CELLS:
                df = df.nlargest(MAX_CELLS, 'risk_score')
            
            # Batch reverse geocode all cell centers
            lat_lons = [h3.cell_to_latlng(h) for h in h3_list]
            rg_results = reverse_geocoder.search(lat_lons)  # ultrafast batch
            
            # Build GeoJSON Feature for each cell
            for row in df.itertuples():
                coords = _h3_to_polygon(row.h3_index)  # h3.cell_to_boundary → [lng,lat] pairs
                pct = (risk_score / max_risk) * 99.9   # normalize to 0–99.9%
                
                feature = {
                    'type': 'Feature',
                    'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                    'properties': {
                        'h3_index': ...,
                        'risk_score': ...,
                        'risk_percent': pct,
                        'district': precise_name_from_reverse_geocoder,
                        'state': ...
                    }
                }
            
            # Serialize entire FeatureCollection to JSON string
            geojson_cache[disease][horizon] = {
                'geojson': json.dumps(feature_collection),
                'maxRisk': max_risk
            }
    
    # Save to gzip-compressed JSON files (one per disease × horizon)
    # Saved as: outputs/geojson_cache/dengue_h1_2024_W52.json.gz
```

**Why Python-side baking?** Python's h3 library is ~15× faster than h3-js (JavaScript) at computing hexagon boundaries. Pre-computing this server-side means the browser receives paint-ready GeoJSON — no computation needed client-side.

### API Endpoints

| Endpoint | Method | Parameters | Returns |
|----------|--------|-----------|---------|
| `/api/health` | GET | — | `{status, version}` |
| `/api/ready` | GET | — | `{core_ready, map_ready}` |
| `/api/metadata` | GET | — | `{latest_year, latest_week, diseases, models}` |
| `/api/analytics` | GET | — | `{total_cells, total_districts, latest_predictions}` |
| `/api/risk` | GET | `?disease=&horizon=` | Lightweight `[[h3_index, risk_score],...]` array |
| `/api/risk/all` | GET | `?disease=` | All 4 horizons in one response |
| `/api/geojson` | GET | `?disease=&horizon=` | Pre-baked GeoJSON FeatureCollection |
| `/api/geojson/all` | GET | `?disease=` | All 4 horizons' GeoJSON in one response |
| `/api/hotspots` | GET | `?disease=&horizon=&limit=` | Top N risk cells with metadata |
| `/api/forecast` | GET | `?disease=&horizon=&district=` | Time series history |
| `/api/cell/{h3_id}` | GET | `?disease=&horizon=` | Single cell time series |
| `/api/cell/{h3_id}/shap` | GET | `?disease=&horizon=` | Global SHAP importance for that disease/horizon |

### Middleware
```python
GZipMiddleware(minimum_size=1000)  # Compress any response > 1KB — cuts map payload by ~70%
CORSMiddleware(allow_origins=["*"])  # Allows frontend dev server (localhost:5173) to call API
```

### Syndemic Risk Computation
```python
# Geometric mean of normalized dengue and malaria risks
syndemic_risk = ((dengue_risk / max_dengue) * (malaria_risk / max_malaria)) ** 0.5 * 10
```

### Forecast History Backfill (Demo Mode)
If only one prediction point exists (real-time demo scenario), 11 synthetic historical data points are generated:
```python
fake_val = max(0, base['pred_lgbm'] + random.uniform(-0.5, 0.2) * base['pred_lgbm'])
```
This prevents the time-series chart from showing only a single point.

---

## 16. Phase 13 — Frontend Dashboard (React + MapLibre)

### Tech Stack
| Component | Technology |
|-----------|-----------|
| Build tool | Vite |
| UI framework | React 18 |
| Map rendering | MapLibre GL JS |
| H3 bindings | h3-js (JavaScript H3 library) |
| Charts | Recharts (LineChart) |
| Icons | Lucide React |
| Persistent cache | IndexedDB (via custom `geoJsonStore.js`) |
| Session cache | sessionStorage (via `apiCache.js`) |

### File: [`src/frontend/src/App.jsx`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/App.jsx)

**Root state management:**
```javascript
const [disease, setDisease] = useState('dengue');  // 'dengue' | 'malaria' | 'syndemic'
const [horizon, setHorizon] = useState(1);          // 1 | 2 | 3 | 4
const [selectedCell, setSelectedCell] = useState(null);  // Clicked hexagon properties
const [theme, setTheme] = useState('dark');         // 'dark' | 'light'
const [metadata, setMetadata] = useState(null);     // API metadata
const [dataVersion, setDataVersion] = useState(null); // "2024_W52" — IndexedDB cache key
```

**Startup initialization:**
1. `cachedFetch('/api/metadata')` → gets `latest_year`, `latest_week`
2. Sets `dataVersion = "2024_W52"` (used as IndexedDB key for GeoJSON)
3. **Background cache priming**: fires `primeCache()` for all `disease × horizon` hotspot combos
4. Renders `<MapComponent>`, `<HotspotSidebar>`, optionally `<AnalyticsPanel>`

**Debounced horizon slider:**
```javascript
const setHorizonDebounced = useCallback((val) => {
    clearTimeout(horizonDebounce.current);
    horizonDebounce.current = setTimeout(() => setHorizon(val), 80);
}, []);
```
80ms debounce prevents re-rendering on every slider tick during drag.

### File: [`src/frontend/src/components/MapComponent.jsx`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/components/MapComponent.jsx)

#### Zero-Latency Architecture (8-Layer System)

**Layer 1: MapLibre Initialization (once)**
```javascript
map.current = new maplibregl.Map({
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [78.9629, 20.5937],  // India's geographic center
    zoom: 4,
    pitch: 40,                    // Subtle 3D tilt for depth effect
});
```

**Layer 2: GeoJSON Paint (in-memory)**
```javascript
const paintFromCache = (dis, hor) => {
    const entry = geoCache.current[dis]?.[String(hor)];  // O(1) lookup
    map.current.getSource('hex-grid').setData(entry.geojson);  // Instant repaint
};
```

**Layer 3: IDB-First Loading (prefetchDisease)**
```
Disease change → Check geoCache (in-memory)
              → If miss: check IndexedDB (idbGetGeo)
              → If miss: fetch /api/geojson/all?disease=X
                         → store in IndexedDB
                         → store in geoCache
                         → paint map
```

**Layer 4: Stale IndexedDB Cleanup**
```javascript
useEffect(() => { idbPrune(dataVersion); }, [dataVersion]);
// Deletes all entries whose key doesn't match current version
```

**Layer 5: Disease change → IDB-first load**
**Layer 6: Horizon change → instant in-memory read (O(1))**
**Layer 7: Cell highlight** — when user clicks a hex, draws outline ring using h3-js's `cellToBoundary()` and MapLibre's `hex-highlight-line` layer
**Layer 8: Theme toggle** — swaps between dark (dark-matter) and light (positron) CartoBaseMaps styles

#### Risk Color Scale
```javascript
const COLOR_STOPS = [
    'interpolate', ['linear'], ['get', 'risk_percent'],
    0,   'rgba(59, 130, 246, 0.18)',   // 0%   → Blue (Safe)
    25,  'rgba(34, 197, 94, 0.45)',    // 25%  → Green (Low)
    50,  'rgba(249, 115, 22, 0.65)',   // 50%  → Orange (Moderate)
    75,  'rgba(239, 68, 68, 0.85)',    // 75%  → Red (High)
    100, 'rgba(153, 27, 27, 1.0)',     // 100% → Dark Red (Critical)
];
```

### File: [`src/frontend/src/components/HotspotSidebar.jsx`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/components/HotspotSidebar.jsx)

- Displays top 20 highest-risk hotspots for current `disease × horizon`
- Uses **Stale-While-Revalidate** (SWR) pattern via `cachedFetch`:
  - Returns cached data instantly
  - Simultaneously re-fetches in background
  - Shows spinning `RefreshCw` icon while revalidating
- Color-coded badges: Red (≥75%), Orange (≥40%), Green (<40%)
- Clicking a hotspot card → `onHotspotClick` → triggers map `flyTo()` to center on that hex

### File: [`src/frontend/src/components/AnalyticsPanel.jsx`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/components/AnalyticsPanel.jsx)

**Content (when a cell is selected):**
1. **Trend Chart** (Recharts LineChart): `Past Cases` vs `Forecast` over the last 12 weeks
2. **Drivers Section**: SHAP feature importance bars (top 10 features)

**Feature Dictionary** (`FEATURE_DICTIONARY`):
Maps 45+ raw ML feature names to human-friendly labels:
```javascript
'cases_lag_1'       → { label: 'History',   desc: 'Past case history (1 wk ago)' }
'rain_lag_2'        → { label: 'Rainfall',  desc: 'Rainfall 2 weeks ago' }
'suitability_lag_1' → { label: 'Breeding',  desc: 'Mosquito climate breeding suitability' }
'sin_week'          → { label: 'Season',    desc: 'Seasonal monsoon timing' }
```

**SHAP Impact Calculation:**
```javascript
const sumShap = top10.reduce((a, c) => a + c.mean_abs_shap, 0) || 1;
const impactPct = (val / sumShap) * 100;  // each feature's % share of total SHAP
const barPct = (val / maxVal) * 100;       // bar width relative to top feature
```

### File: [`src/frontend/src/apiCache.js`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/apiCache.js)

**Stale-While-Revalidate (SWR) Cache:**
- Storage: `sessionStorage` (clears on browser close)
- TTL: 10 minutes
- Key format: `vh:/api/hotspots?disease=dengue&horizon=1&limit=20`

```javascript
async function cachedFetch(url, onUpdate = null) {
    const cached = readCache(url);
    
    if (cached !== null) {
        // Return cached IMMEDIATELY; kick off background revalidation
        if (onUpdate) fetchWithRetry(url)
            .then(r => r.json())
            .then(fresh => { writeCache(url, fresh); onUpdate(fresh); });
        return cached;  // → instant return
    }
    
    // Cache miss: wait for network
    const r = await fetchWithRetry(url, maxAttempts=5, baseDelay=2000);
    writeCache(url, await r.json());
}
```

**Retry Logic**: Exponential backoff starting at 2 seconds, factor 1.5×, max 5 attempts. Handles IMD server flakiness and cold-start backend delays.

### File: [`src/frontend/src/geoJsonStore.js`](file:///w:/MiniProject5/VectorHotspot/src/frontend/src/geoJsonStore.js)

**IndexedDB persistent GeoJSON cache:**
- Database: `vh_geojson_v2`, Object Store: `geojson`
- Key format: `geojson_v2_dengue_2024_W52`
- Stores all 4 horizons' GeoJSON together per disease entry

```javascript
// On cache hit: map renders in ~10ms (vs ~2min compute)
const cached = await idbGetGeo(disease, dataVersion);
if (cached) { geoCache.current[disease] = cached; paintFromCache(disease, horizon); }

// After network fetch: persist for future loads
await idbSetGeo(disease, dataVersion, horizons);

// When version changes: prune stale entries
await idbPrune(dataVersion);  // deletes all keys not containing current version
```

---

## 17. Saved Model Artifacts

### 24 Total Trained Models

| File | Size | Model Type | Disease | Horizon |
|------|------|-----------|---------|---------|
| `lgbm_env_dengue_lead1.joblib` | 1.43 MB | LightGBM | Dengue | t+1 |
| `lgbm_env_dengue_lead2.joblib` | 1.43 MB | LightGBM | Dengue | t+2 |
| `lgbm_env_dengue_lead3.joblib` | 1.45 MB | LightGBM | Dengue | t+3 |
| `lgbm_env_dengue_lead4.joblib` | 1.43 MB | LightGBM | Dengue | t+4 |
| `lgbm_env_malaria_lead1.joblib` | 1.42 MB | LightGBM | Malaria | t+1 |
| `lgbm_env_malaria_lead2.joblib` | 1.40 MB | LightGBM | Malaria | t+2 |
| `lgbm_env_malaria_lead3.joblib` | 1.42 MB | LightGBM | Malaria | t+3 |
| `lgbm_env_malaria_lead4.joblib` | 1.38 MB | LightGBM | Malaria | t+4 |
| `xgb_env_dengue_lead{1-4}.joblib` | ~920 KB each | XGBoost | Dengue | t+1–4 |
| `xgb_env_malaria_lead{1-4}.joblib` | ~925 KB each | XGBoost | Malaria | t+1–4 |
| `rf_baseline_dengue_lead{1-4}.joblib` | ~109–123 KB each | Random Forest | Dengue | t+1–4 |
| `rf_baseline_malaria_lead{1-4}.joblib` | ~118–165 KB each | Random Forest | Malaria | t+1–4 |

**Naming convention**: `{model_type}_{feature_set}_{disease}_lead{k}.joblib`

The models in `models/` are all **environment-only** (no case history features). The full-feature models (which include `cases_lag_*` features) were trained in `train_forecasting_models.py` — their predictions are saved to `forecast_{disease}_predictions.parquet`, which the backend uses directly.

---

## 18. Output Directory Structure

```
outputs/
├── hotspots/
│   ├── dengue_hotspots_lead{1-4}.parquet          ← All hexagons, full year predictions
│   ├── dengue_hotspots_test_2023_2024.parquet     ← Test set with both pred + actual flags
│   ├── dengue_hotspot_taxonomy_2024.parquet        ← Emerging/Persistent/Expanding/etc labels
│   ├── malaria_hotspots_lead{1-4}.parquet
│   ├── malaria_hotspots_test_2023_2024.parquet
│   └── malaria_hotspot_taxonomy_2024.parquet
│
├── explainability/
│   ├── shap_values_dengue_lead{1-4}.csv           ← Per-sample SHAP values (~50MB each)
│   ├── shap_values_malaria_lead{1-4}.csv
│   ├── global_shap_importance_dengue_lead{1-4}.csv← Top features sorted by mean|SHAP|
│   ├── global_shap_importance_malaria_lead{1-4}.csv
│   └── global_shap_importance_all.csv             ← Combined across all diseases+horizons
│
├── geojson_cache/
│   └── {disease}_h{1-4}_{year}_W{week}.json.gz   ← Server-side baked GeoJSON, gzipped
│
├── metrics/
│   ├── hotspot_validation_metrics.csv             ← P/R/F1/IoU per disease×horizon×split
│   ├── lead_time_analysis.csv                     ← Detection rate per lead week
│   ├── lead_time_summary.csv                      ← Summary with max effective lead
│   └── hotspot_taxonomy_distribution.csv          ← % of cells in each taxonomy bucket
│
├── tables/
│   ├── model_evaluation_metrics.csv               ← R²/MAE/RMSE for all model comparisons
│   ├── rf_baseline_metrics.csv                    ← Random Forest specific metrics
│   └── ablation_study_results.csv                 ← Ablation study R²/MAE by scenario
│
├── reports/                                        ← Human-readable summary reports
└── alerts/                                         ← Generated alert outputs
```

---

## 19. End-to-End Data Flow Diagram

```
                    RAW DATA SOURCES
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      IMD Weather    Disease CSVs    WorldPop TIFFs
      (imdlib API)   (NVBDCP)        (1km rasters)
          │               │               │
          ▼               ▼               ▼
   fetch_imd_weather  (pre-existing)  compute_district_
       .py              data          population.py
          │               │               │
          ▼               ▼               ▼
   imd_district_    dengue/malaria    district_pop_
   weekly_weather    _district_       2000_2020.csv
   _2000_2024.csv    2000_2024.csv
          │               │               │
          └───────────────┼───────────────┘
                          ▼
               generate_h3_grid.py
               (H3 Resolution 7)
                          │
                          ▼
               india_h3_grid_res7.csv
               h3_adjacency_res7.npz
                          │
                          ▼
               disaggregation_prototype.py
               (District → H3 dasymetric)
                          │
                          ▼
         dengue_hex_annual.csv / malaria_hex_annual.csv
                          │
                          ▼
              FEATURE ENGINEERING
              (in training scripts)
              45+ features per H3 × Week:
              • Cases lags (1,2,3,4,8 weeks)
              • Rolling stats (4w, 12w mean/std)
              • Weather lags (1,2,4,6 weeks)
              • Rainfall rolling sums (2w, 4w)
              • sin_week / cos_week
              • Population + density
              • Neighbor cases (k1, k2 rings)
              • NDVI, JRC water, WorldCover
              • Suitability index
                          │
                          ▼
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    TRAIN parquet    VAL parquet     TEST parquet
    (~2000–2020)    (~2021–2022)    (~2023–2024)
         │
         ▼
    ML TRAINING
    (train_env_only_models.py)
         │
    ┌────┴──────────────────┐
    ▼                        ▼
 LightGBM × 8           XGBoost × 8
 (2 diseases × 4 horiz)  (same)
         │
    ┌────┴────────────────┐
    ▼                      ▼
 models/*.joblib    forecast_*_predictions.parquet
                          │
                          ▼
              GETIS-ORD Gi* COMPUTATION
              (per cell per week)
                          │
                          ▼
              hotspot_*_lead{1-4}.parquet
              hotspot_taxonomy_2024.parquet
                          │
                          ▼
              SHAP COMPUTATION
              (shap_analysis.py)
                          │
                          ▼
              global_shap_importance_*.csv
                          │
              EVALUATION
              ┌───────────┴──────────┐
              ▼                      ▼
         hotspot_validator    lead_time_analysis
         .py                  .py
              │                      │
              ▼                      ▼
         P/R/F1/IoU          Detection rates
         metrics.csv          at t+1..t+4
                          │
                          ▼
                     FASTAPI BACKEND
                    (uvicorn server)
                          │
                    ┌─────┴─────────────┐
                    ▼                   ▼
             Bake GeoJSON         Load SHAP CSVs
             (50k top cells)      (global importance)
             Disk cache           Memory cache
                    │
                    ▼
             REST API Endpoints
             /api/geojson/all
             /api/hotspots
             /api/cell/{id}/shap
                    │
                    ▼
            REACT + MAPLIBRE
            FRONTEND DASHBOARD
            ┌───────────────────────────────────┐
            │  Header: Disease selector + slider │
            │  MapComponent:                     │
            │  • H3 hexagons colored by risk     │
            │  • IndexedDB GeoJSON cache         │
            │  • SWR data loading pattern        │
            │  HotspotSidebar:                   │
            │  • Top 20 hotspot cards            │
            │  • Fly-to on click                 │
            │  AnalyticsPanel (on cell click):   │
            │  • 12-week trend chart             │
            │  • SHAP driver bars (top 10)       │
            └───────────────────────────────────┘
```

---

## 20. Key Scientific Findings

### Finding 1: Dengue vs Malaria Epidemiological Dynamics
| Property | Dengue | Malaria |
|----------|--------|---------|
| **Dominant driver** | Historical case momentum | Environment + Seasonality |
| **R² (full model)** | ~0.89 | ~0.82 |
| **R² (only weather)** | ~0.55 | ~0.72 |
| **R² (only cases)** | ~0.86 | ~0.58 |
| **Key analogy** | Localized wildfire | Seasonal monsoon pattern |
| **Policy implication** | Track recent case counts | Track breeding conditions |

### Finding 2: Spatial vs Temporal Generalizability
- Models trained on seen districts generalize well to unseen spatial holdout districts
- Confirms models learned real epidemiological patterns, not location-specific memorization

### Finding 3: Early Warning Capability
- At t+1 (1-week ahead): high detection rate (model mostly correct)
- At t+4 (4-weeks ahead): lower detection rate but still better than baseline
- Malaria tends to be predictable further ahead (seasonal structure)
- Dengue is more volatile (emergence harder to predict weeks out)

### Finding 4: Hotspot Spatial Clustering
- Gi* statistics reveal that high predicted cells do form significant spatial clusters
- IoU metric validates that predicted hotspot geography substantially overlaps observed geography

---

## 21. Plan vs Implementation Gap Analysis

Based on [`plan_vs_implementation.md`](file:///w:/MiniProject5/VectorHotspot/plan_vs_implementation.md):

### Fully Implemented ✅
- H3 Resolution 7 grid generation
- IMD weather data pipeline (2000–2024)
- WorldPop population data
- Disease disaggregation to H3
- All 45 feature groups (lags, rolling, seasonality, spatial, environmental)
- Target variables for t+1 through t+4
- Chronological train/val/test split (no leakage)
- LightGBM × 8 models (both diseases × all horizons)
- XGBoost × 8 models
- Naive Persistence baseline
- Historical Seasonal Mean baseline
- Global SHAP analysis
- Hotspot detection (Getis-Ord Gi*)
- Hotspot taxonomy (Emerging/Persistent/etc.)
- Hotspot validation (P/R/F1/IoU)
- Early warning lead time analysis
- Feature ablation study
- FastAPI backend with GeoJSON baking
- React + MapLibre GL dashboard
- IndexedDB + sessionStorage caching
- Dark/Light theme
- Syndemic (combined) risk mode

### Partially Implemented 🟡
- CHIRPS gridded rainfall (used IMD rainfall instead — similar quality)
- Environmental variables (folders exist; integration into final feature set unclear)
- Prediction uncertainty intervals (analysis exists; not fully exposed in frontend)
- Data quality report (scripts exist; formal report not generated)

### Not Implemented ❌
- Formal time-series cross-validation (expanding window folds)
- Formal leakage audit document
- MAPE metric (intentionally skipped for zero-heavy count data)
- Feature metadata documentation file
- Calibration curves / Brier scores

### Extra Additions ➕ (Beyond Original Plan)
- **Spatial holdout** (15% unseen districts) — rigorous geographic generalizability test
- **Coarse District Aggregation** baseline (Baseline 3)
- **Random Forest** baseline (24 additional models)
- **Syndemic risk** (geometric mean fusion of dengue + malaria)
- **Server-side GeoJSON baking** with gzip disk cache
- **IndexedDB persistent GeoJSON cache** (zero-latency map loads after first visit)
- **Stale-While-Revalidate (SWR)** API caching pattern
- **Reverse geocoder** for precise district naming in GeoJSON
- **Feature Dictionary** (human-readable SHAP labels in frontend)
- **RMSLE** metric (Root Mean Squared Log Error — better for zero-inflated count data)

---

*Analysis generated: 2026-10-01*
*Project: VectorHotspot | Workspace: w:\MiniProject5\VectorHotspot*
