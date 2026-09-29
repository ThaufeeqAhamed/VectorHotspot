# VectorHotspot: Comprehensive Project Roadmap & Remaining Phases
**Dengue-Malaria Dual-Disease Spatiotemporal Early Warning System (India)**

---

## 1. Executive Summary & Project Context

### 1.1 Project Identity & Objective
* **Repository:** `https://github.com/ThaufeeqAhamed/VectorHotspot` (Private), branch: `switch`
* **Core Problem:** India's public epidemiological surveillance data for vector-borne diseases is available only at coarse spatial resolution (State-level for Dengue, District-level for Malaria) and annual granularity. Coarse data is inadequate for hyper-local public health action and early warning.
* **Core Solution:** A fully reproducible, real-data machine learning pipeline that:
  1. Statistically disaggregates coarse annual case counts to fine-grained spatial units (**Uber H3 Hexagons at Resolution 7, ~5.2 km² / ~2.3 km across, 620,742 cells over India**) using multi-source demographic and environmental covariates while **strictly preserving mass** ($\sum \text{hex\_cases} = \text{unit\_cases}$).
  2. Temporally disaggregates annual hexagon totals to weekly series driven by biophysical temperature/rainfall suitability.
  3. Trains dual-disease spatiotemporal forecasting models (LightGBM/XGBoost) for 1–4 week-ahead prediction.
  4. Applies **Getis-Ord $G_i^*$ hotspot detection directly on the PREDICTED future risk surface** ("Forecast-to-Hotspot Fusion").
  5. Validates predicted future hotspots against subsequently observed real outcomes, comparing dual-disease vector ecologies.

---

## 2. Completed Phases (Phases 1 to 10) — Audit & Exact Status

| Phase | Description | Key Deliverables / Results | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1–3** | **Data Collection & Cleaning** | 6 Tier-1 datasets cleaned: OpenDengue (State/Annual, 2010–2024), NCVBDC Malaria (District/Annual, 2000–2024), IMD Weather (District/Weekly, 2000–2024, Tmax/Tmin/Rain), WorldPop Population (District & Hex 1km, 2000–2020), 724 clean districts GeoJSON. | **COMPLETE** |
| **Phase 4** | **H3 Hexagonal Spatial Grid** | Generated 620,742 H3 Res-7 hexagons tagged with `(state, district)`. Exact coverage of all 724 districts and 36 states/UTs with zero coordinate duplicates. `data/processed/india_h3_grid_res7.csv`. | **COMPLETE** |
| **Phase 5** | **Spatial Disaggregation (Tier 1: Population)** | Poisson regression with aggregation constraint. Dengue $\beta_1 = +0.7662$ (urban), Malaria $\beta_1 = -1.2485$ (rural). **0.000000 max error (Exact mass preservation)**. `wire_disaggregation_model.py`. | **COMPLETE** |
| **Phase 6** | **Tier 2 Covariates & Enhanced Model Refit** | Downloaded & processed: MODIS NDVI (AppEEARS 2018–2020 mean), ESA WorldCover 2021 (water, tree, built, shrub fractions), JRC Global Surface Water occurrence. `data/processed/hex_tier2_covariates.csv` (28.7MB). Refit multi-covariate Poisson model. **0.000000 max error maintained**. 2024 comparative map in `outputs/figures/disaggregation_real_results_2024.png`. | **COMPLETE** |
| **Phase 7** | **Biophysical Temporal Disaggregation** | Annual hexagon case totals → weekly time-series using Brière thermal curves (Dengue optimal 26–32°C, Malaria 18–32°C) + lagged rainfall hydrology. **747,903,370 weekly rows** generated across 2000–2024. **0.000000 max error**. Year-by-year float64 accumulation to prevent float32 rounding. `src/disaggregation/temporal_disaggregation.py`. | **COMPLETE** |
| **Phase 8** | **Spatiotemporal Feature Engineering** | Streaming sliding-window year-by-year feature assembly. **61,642,778 records × 52 features**. Zero temporal or spatial data leakage. 15% spatial holdout. H3 k-ring 1 (W1) and k-ring 2 (W2) neighbor spillover. Train set: 2010–2022 (Dengue), 2000–2022 (Malaria); Test set: 2023–2024. `src/features/build_feature_store.py`. | **COMPLETE** |
| **Phase 9** | **Dual-Disease Multi-Horizon Forecasting** | LightGBM (primary) + XGBoost (secondary) per disease per horizon. Delta formulation: predict (Y_future − Y_current). **R² 0.87–0.95** on unseen 2023–2024 test set. Beats naive persistence and seasonal baselines. 16 `.joblib` model files saved to `models/`. Thread-capped: OMP/MKL/OPENBLAS = 4. `src/models/train_forecasting_models.py`. | **COMPLETE** |
| **Phase 10** | **Forecast-to-Hotspot Fusion (Getis-Ord Gi*)** | Vectorized sparse CSR Gi* on PREDICTED risk surfaces (t+1→t+4). W_star = h3.grid_disk(h,1) with self-loops. **2.97s** for 35,818 hexes × 104 weeks. 4-Tier Taxonomy (Persistent overrides others). Dengue t+1 IoU=**0.8145**, Malaria t+1 IoU=**0.7787**. Outputs: `outputs/hotspots/`, `outputs/tables/hotspot_fusion_evaluation_metrics.csv`. `src/hotspots/forecast_hotspot_fusion.py`. | **COMPLETE** |

### Critical Verified Decisions & Bugs Fixed (DO NOT RE-LITIGATE OR BREAK):
1. **H3 Resolution 7:** ~5.2 km² per cell (620,742 hexagons). Do not downgrade to resolution 6 (too coarse) or resolution 8 (computationally bloated without covariate support).
2. **Join Key Convention:** Always join administrative boundaries on `(state, district)` composite tuples, never `dt_code` (non-unique nationally).
3. **Malaria Data Correction:** `data/processed/malaria_district_2000_2024.csv` contains the canonical district lookup fix that repaired the 52% PDF state-mislabeling bug.
4. **Mass Preservation:** $\sum_{h \in U} \hat{Y}_{h, t} \equiv Y_{U, t}$ must hold exactly ($0.000000$ deviation) by construction via within-group scaling.
5. **No Synthetic Data:** Final pipeline uses strictly real data. `disaggregation_prototype.py` was proof-of-concept only.
6. **Path Resolution:** All scripts resolve paths via `Path(__file__).resolve().parent.parent...` relative to project root.
7. **Large Output Files:** Hex-annual and future hex-weekly CSVs/Parquets are gitignored.

---

## 3. End-to-End Pipeline Architecture

```
[ Tier 1 & 2 Surveillance & Remote Sensing Data ]
   ├── OpenDengue (State/Annual)
   ├── NCVBDC Malaria (District/Annual)
   ├── WorldPop 1km Population (2000-2020)
   ├── MODIS MOD13A2 NDVI (2018-2020)
   ├── ESA WorldCover 2021 (Trees, Water, Built, Shrub)
   └── JRC Surface Water Occurrence (1984-2021)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5-6: Mass-Preserving Spatial Disaggregation  ✅ DONE │
│  (H3 Resolution 7, 620,742 Hexagons, Annual Grid)           │
│  Outputs: dengue_hex_annual.csv, malaria_hex_annual.csv    │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 7: Biophysical Temporal Disaggregation    ✅ DONE    │
│  (Annual Hexagon Totals ──► Weekly Hexagon Time-Series)     │
│  • IMD Weekly Weather (Rainfall, Tmax, Tmin, DTR)           │
│  • Temperature-dependent EIP & R0 thermal suitability curves│
│  • Mass-preserving weekly allocation: sum(weeks) == annual  │
│  • 747,903,370 weekly rows, 0.000000 mass error             │
│  Outputs: dengue_hex_weekly.parquet, malaria_hex_weekly.pq  │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 8: Spatiotemporal Feature Engineering Engine ✅ DONE │
│  • Temporal Lags (t-1 to t-8 weeks cases & weather)         │
│  • Rolling Window Statistics (4w, 12w moving mean/variance) │
│  • H3 Spatial Neighbor Features (k-ring 1 & 2 spillover)    │
│  • Seasonality Encodings (sin/cos week-of-year)             │
│  • 61,642,778 records × 52 features, zero leakage           │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 9: Dual-Disease Multi-Horizon Forecasting  ✅ DONE   │
│  • Independent LightGBM & XGBoost per disease               │
│  • Horizons: t+1, t+2, t+3, t+4 weeks ahead                 │
│  • Delta formulation: predict (Y_future − Y_current)        │
│  • R² 0.87–0.95; beats naive & seasonal baselines           │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 10: Forecast-to-Hotspot Fusion (Getis-Ord Gi*)✅DONE │
│  • Gi* Local Spatial Autocorrelation on PREDICTED risk      │
│  • Sparse CSR W_star (k-disk=1 with self-loops)             │
│  • 4-Tier Taxonomy: Emerging, Intensifying, Persistent,     │
│    Diminishing (Persistent overrides all others)            │
│  • Dengue t+1 IoU=0.8145 | Malaria t+1 IoU=0.7787          │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 11: Future-Hotspot Validation & Research Evaluation  │
│  • Predicted Hotspots vs Subsequently Observed Hotspots     │
│  • Metrics: Spatial IoU, Precision, Recall, F1, Lead Time   │
│  • National Spatial Holdout Validation (108 districts)      │
│  • Ablation Studies (Spatial Method, Feature Eng, Weather)  │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 12: Dual-Disease Explainability (SHAP) & Ecology     │
│  • TreeSHAP global & local feature importance attribution   │
│  • Vector ecology comparison: Urban Aedes vs Forest Anopheles│
│  • Conformal Prediction Intervals & Uncertainty Calibration │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 13: Early Warning Engine & Operational Dashboard     │
│  • Alert Matrix: Risk Level × Hotspot Significance × Lead   │
│  • Interactive Map Dashboard (Streamlit / Deck.gl / Leaflet)│
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Completed Phase Details & Remaining Phase Specification

---

### ✅ PHASE 7: Biophysical Temporal Disaggregation — COMPLETE

**Delivered:** Annual hexagon case totals → weekly time-series for 620,742 hexes × 52 weeks, 2000–2024.
- Brière thermal performance curves: Dengue optimal 26–32°C, Malaria 18–32°C
- Lagged rainfall hydrology (2–6 week cumulative lag)
- **747,903,370 weekly rows** generated; **0.000000 max mass error**
- Critical fix: year-by-year float64 accumulation prevents float32 rounding artifacts
- Outputs: `data/processed/dengue_hex_weekly.parquet`, `malaria_hex_weekly.parquet` (ZSTD compressed)
- Script: `src/disaggregation/temporal_disaggregation.py`

---

### ✅ PHASE 8: Spatiotemporal Feature Engineering — COMPLETE

**Delivered:** 61,642,778 records × 52 features; zero temporal or spatial data leakage.
- Case lags t-1 to t-8, rolling mean/std (4w, 12w), momentum
- IMD weather lags (Tmax, Tmin, DTR, rainfall) at t-1 to t-6
- H3 k-ring 1 (W1) and k-ring 2 (W2) spatial neighbor spillover (sparse CSR adjacency)
- Static covariates: log(Population), NDVI, land cover fractions, JRC water occurrence
- Cyclical encodings: sin/cos(2π·week/52)
- 15% spatial holdout (geographically contiguous district clusters excluded from training)
- Train: 2010–2022 (Dengue), 2000–2022 (Malaria); Test: 2023–2024
- Outputs: `features_dengue_train.parquet`, `features_dengue_test.parquet` (and malaria equivalents)
- Script: `src/features/build_feature_store.py`

---

### ✅ PHASE 9: Dual-Disease Multi-Horizon Forecasting — COMPLETE

**Delivered:** LightGBM (primary) + XGBoost (secondary) per disease per horizon; 16 `.joblib` model files.
- Delta formulation: models predict (Y_future − Y_current); base cases added back for non-negative predictions
- Hyperparameters: n_estimators=250, learning_rate=0.06, num_leaves=63, max_depth=8, n_jobs=4
- Thread-capped: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4
- **R² 0.87–0.95** on unseen 2023–2024 test set; beats naive persistence and seasonal baselines
- Test set: 35,818 spatial holdout hexes × 104 weeks = 3,725,072 rows
- Outputs: `models/lgbm_dengue_lead{1-4}.joblib`, `models/xgb_malaria_lead{1-4}.joblib`, etc.
- Script: `src/models/train_forecasting_models.py`

---

### ✅ PHASE 10: Forecast-to-Hotspot Fusion (Getis-Ord $G_i^*$) — COMPLETE

**Delivered:** Vectorized sparse Gi* on predicted risk surfaces; 4-tier hotspot taxonomy validated against ground truth.
- W_star built via `h3.grid_disk(h, 1)` with self-loops; 58,022 edges for 35,818 hexes (1.62 avg neighbors/hex)
- All 104 time slices computed in **2.97s** (batch sparse matrix operations)
- 4-Tier Taxonomy (Persistent overrides all others):
  - **Persistent:** z-scores z1–z4 all ≥ 2.58 (highest certainty, most actionable)
  - **Emerging:** current z < 1.96, lead-4 z ≥ 1.96 (newly developing)
  - **Intensifying:** z4 > z3 > z2 > z1, z4 ≥ 1.96 (growing severity)
  - **Diminishing:** current z ≥ 1.96, lead-4 z < 1.96 (waning)
- Hotspot detection results (2023–2024 test set, Spatial IoU vs. actual observed hotspots):

| Disease | Horizon | IoU | Precision | Recall | F1 |
|---------|---------|-----|-----------|--------|-----|
| Dengue | t+1 | **0.8145** | 0.8899 | 0.9058 | 0.8978 |
| Dengue | t+2 | 0.7716 | 0.8617 | 0.8806 | 0.8711 |
| Dengue | t+3 | 0.7277 | 0.8332 | 0.8518 | 0.8424 |
| Dengue | t+4 | 0.6907 | 0.8141 | 0.8201 | 0.8171 |
| Malaria | t+1 | **0.7787** | 0.8665 | 0.8849 | 0.8756 |
| Malaria | t+2 | 0.6369 | 0.7851 | 0.7714 | 0.7782 |
| Malaria | t+3 | 0.5080 | 0.7345 | 0.6223 | 0.6738 |
| Malaria | t+4 | 0.4487 | 0.6803 | 0.5685 | 0.6194 |

- Outputs: `outputs/hotspots/dengue_hotspots_test_2023_2024.parquet` (219.4 MB), `malaria_hotspots_test_2023_2024.parquet` (116.7 MB), `outputs/tables/hotspot_fusion_evaluation_metrics.csv`, `outputs/figures/hotspot_fusion_diagnostics.png`
- Script: `src/hotspots/forecast_hotspot_fusion.py`

**DO NOT BREAK (Phase 10 invariants):**
- Never re-threshold `is_hotspot_act_lead_k` from raw cases (e.g. 90th percentile). These columns are computed FROM Gi* on actual case data inside phase 10 and must be read directly from parquet.
- W_star must include self-loops (`h3.grid_disk(h, 1)`, not `h3.grid_ring(h, 1)`).
- Thread caps must remain (OMP/MKL/OPENBLAS = 4) — removing them locks the 12-core CPU.

---

### PHASE 11: Rigorous Evaluation, Ablation Studies & Ground-Truth Validation
* **Goal:** Scientifically defend the pipeline against naive approaches and prove future predictive utility.

#### Detailed Tasks:
1. **Future Hotspot Verification:**
   * Compare Predicted Hotspot Polygons at week $t$ for horizon $t+k$ against the Observed Hotspots computed from actual data at week $t+k$.
   * Calculate **Spatial Intersection over Union (Spatial IoU)**:
     $$\text{Spatial IoU} = \frac{\text{Area}(\text{Predicted Hotspot} \cap \text{Observed Hotspot})}{\text{Area}(\text{Predicted Hotspot} \cup \text{Observed Hotspot})}$$
   * Classification metrics: Precision, Recall, F1-Score, and Detection Lead Time (weeks in advance).
2. **National Spatial Holdout Validation:**
   * Filter 108 completely withheld test-set districts across 28 states.
   * Aggregate H3 hexagon disaggregated estimates up to the district level.
   * Compare predicted spatial risk ranking across holdout districts against actual spatial ranking of reported cases (Spearman rank correlation).
3. **Formal Ablation Suite:**
   * **Ablation 1 (Spatial Methodology):** Full Tier-1+Tier-2 Covariate Disaggregation vs. Naive Uniform Disaggregation vs. Population-Only Share Disaggregation.
   * **Ablation 2 (Feature Engineering):** Model with H3 $k$-ring spatial neighbors vs. Model with administrative district mean cases vs. Model with no spatial features.
   * **Ablation 3 (Environmental Signals):** With vs. without IMD weather lagged suitability features (temp, rainfall, thermal suitability).

---

### PHASE 12: Dual-Disease Comparative Ecology, Explainability (SHAP) & Calibration
* **Goal:** Unpack the biological and geographic divergence between Dengue and Malaria transmission and provide local model interpretability.

#### Detailed Tasks:
1. **SHAP (SHapley Additive exPlanations):**
   * Run TreeSHAP across validation and test sets.
   * Global feature importance plots comparing top drivers for Dengue vs. Malaria.
   * Local waterfall/force plots for individual emerging hotspot hexagons (e.g. explaining why a specific hexagon in urban Delhi triggered an alert vs. a rural forest hexagon in Odisha).
2. **Dual-Disease Ecological Divergence Analysis:**
   * Co-hotspot index: Map regions where Dengue and Malaria co-occur vs. where they strictly diverge.
   * Quantify relationship between urbanization index (`frac_built`), forest cover (`frac_trees`), surface water (`jrc_occurrence`), and disease dominance.
3. **Uncertainty Calibration:**
   * Conformal prediction intervals ($90\%$ and $95\%$ prediction bounds) per hexagon.
   * Reliability diagrams and calibration curves.

---

### PHASE 13: Operational Early Warning Engine & Interactive Dashboard
* **Goal:** Deliver an intuitive, interactive visualization tool and structured alert engine.

#### Detailed Tasks:
1. **Alert Generation Engine:**
   * Threshold-based rule engine:
     * **Level 1 (Yellow / Watch):** Predicted case growth $>20\%$, $G_i^* > 1.65$.
     * **Level 2 (Orange / Warning):** Emerging Hotspot ($G_i^* \ge 1.96$, $p < 0.05$), Lead time $\ge 2$ weeks.
     * **Level 3 (Red / High Alert):** Intensifying / Persistent Hotspot ($G_i^* \ge 2.58$, $p < 0.01$).
2. **Interactive Web Dashboard:**
   * Streamlit application with PyDeck / Mapbox / Leaflet map rendering.
   * Interactive H3 hexagon layer with color-coded risk and hotspot layers.
   * Disease toggle (Dengue vs. Malaria vs. Dual Comparison).
   * Time-slider for 1–4 week forecast horizon.
   * Drill-down panel: Click any hexagon to see time-series forecast curve, top SHAP explanatory factors, and parent administrative totals.

---

## 5. Summary Table: Remaining Phases & Work Breakdown

| Phase | Phase Name | Primary Input Data | Primary Outputs / Artifacts | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Phase 7** | **Temporal Disaggregation** | Annual hex CSVs, IMD Weekly Weather | `dengue_hex_weekly.parquet`, `malaria_hex_weekly.parquet` | ✅ DONE |
| **Phase 8** | **Feature Engineering** | Weekly Parquet, IMD Weather, H3 Grid | `features_dengue_train/test.parquet`, `features_malaria_train/test.parquet` | ✅ DONE |
| **Phase 9** | **Forecasting Models** | Feature Store | 16 `.joblib` model files; forecast risk surfaces t+1…t+4 | ✅ DONE |
| **Phase 10** | **Hotspot Fusion ($G_i^*$)** | Forecast Risk Surfaces | Hotspot parquets, IoU metrics table, diagnostic figure | ✅ DONE |
| **Phase 11** | **Evaluation & Ablations** | Predicted vs Observed Hotspots, Spatial Holdout Districts | Ablation tables, holdout validation stats, lead-time metrics | ⏳ NEXT |
| **Phase 12** | **SHAP & Dual Ecology** | Trained Models, Feature Store | SHAP summary plots, co-hotspot ecology analysis, calibration curves | 📋 PLANNED |
| **Phase 13** | **Dashboard & Alerts** | All predictions, hotspots, SHAP values | Streamlit interactive dashboard, `alerts_summary.json` | 📋 PLANNED |

---

*This document represents the official master specification for VectorHotspot moving forward.*
