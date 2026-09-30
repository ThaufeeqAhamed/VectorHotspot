# VectorHotspot — Plan vs Implementation Comparison

> Comparison of `proj_plan.md` (100 sections) against the actual codebase as of 2026-09-30.

---

## Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented |
| 🟡 | Partially implemented |
| ❌ | Not implemented / Missing |
| ➕ | Extra / Not in plan |

---

## 1. Project Layer Overview

| Layer (from Plan) | Status | Notes |
|---|---|---|
| Layer 1 – Data Pipeline | ✅ | H3 grid, weather, population, disease data all processed |
| Layer 2 – Spatial-Temporal Processing | ✅ | H3 × Week dataset built, adjacency matrix computed |
| Layer 3 – ML Prediction | ✅ | LightGBM + XGBoost, all 4 horizons, both diseases |
| Layer 4 – Explainability & Uncertainty | 🟡 | SHAP outputs exist; uncertainty intervals exist — but no dedicated src module |
| Layer 5 – Spatial Hotspot & Validation | 🟡 | Hotspot outputs exist; validation partially done; `src/evaluation/` is empty |
| Layer 6 – Frontend / Visualization | 🟡 | Streamlit dashboard exists — React + MapLibre is **not built** |

---

## 2. Data Layer (Sections 7–12)

### 2.1 Disease Data
| Planned | Status | Details |
|---|---|---|
| Dengue (date, location, district, state, cases) | ✅ | `dengue_hex_weekly.parquet`, `dengue_hex_annual.csv` |
| Malaria (date, location, district, state, cases) | ✅ | `malaria_district_2000_2024.csv`, `malaria_hex_*.parquet` |

### 2.2 Weather Data (Section 8 – IMD-based)
| Planned | Status | Details |
|---|---|---|
| Temperature, Humidity, Rainfall | ✅ | `imd_district_weekly_weather_2000_2024_FIXED.csv` |
| 7/14/28-day rolling derived variables | ✅ | Present in feature engineering (confirmed via feature columns in training script) |

### 2.3 Rainfall (Section 9 – CHIRPS)
| Planned | Status | Details |
|---|---|---|
| CHIRPS gridded rainfall → H3 × Week | 🟡 | Rainfall in IMD weather data, but no standalone CHIRPS pipeline visible |

### 2.4 Population Data (Section 10 – WorldPop)
| Planned | Status | Details |
|---|---|---|
| Population, Population density | ✅ | `district_population_2000_2020.csv`, `compute_district_population.py` |
| WorldPop gridded source | 🟡 | `data/raw/population/` folder exists; WorldPop scripts present |

### 2.5 Environmental Data (Section 11 – Optional)
| Planned | Status | Details |
|---|---|---|
| NDVI | 🟡 | `data/raw/ndvi/` folder exists — integration status unclear |
| Land Cover | 🟡 | `data/raw/worldcover/` folder exists |
| Water Bodies | 🟡 | `data/raw/jrc_water/` folder exists |
| Land Surface Temperature, Elevation, Urbanization | ❌ | No evidence of these in processed features |

### 2.6 Geographic Data (Section 12)
| Planned | Status | Details |
|---|---|---|
| Country/State/District boundaries | ✅ | `india_districts_clean.geojson`, `bhopal_wards.geojson` |

---

## 3. Spatial Representation — H3 (Sections 13–14)

| Planned | Status | Notes |
|---|---|---|
| H3 hexagonal grid as spatial unit | ✅ | H3 Resolution 7 (~5.2 km²) confirmed |
| H3 × Week dataset | ✅ | `india_h3_grid_res7.csv`, `h3_adjacency_res7.npz` |
| Resolution choice documented | ✅ | Documented in `generate_h3_grid.py` docstring with reasoning |
| `H3 Cell × Week` modelling grid | ✅ | Features datasets (`features_dengue_*.parquet`) use this structure |

---

## 4. Data Processing Pipeline (Sections 15–16)

| Step | Status | Notes |
|---|---|---|
| Download / Import | ✅ | `fetch_imd_weather.py`, population scripts |
| Schema / Date / Geographic standardization | ✅ | Done across data_prep scripts |
| Missing value handling | 🟡 | Addressed in scripts but no standalone QA report |
| Outlier checks | 🟡 | Partial (`recompute_temperature_fast.py` indicates fixes were needed) |
| Spatial alignment → H3 aggregation | ✅ | Done in `generate_h3_grid.py` and disaggregation scripts |
| **Data Quality Report** (Section 16) | ❌ | Planned to produce a formal report — **none exists** |

---

## 5. Target Definition & Data Leakage (Sections 17–18)

| Planned | Status | Notes |
|---|---|---|
| +1/+2/+3/+4 week targets defined | ✅ | `target_lead_1` through `target_lead_4` columns in feature sets |
| Chronological (no future leakage) | ✅ | Training script uses time-based train/val/test split |
| **Leakage audit documented** (Section 79) | ❌ | No formal leakage audit document exists |

---

## 6. Feature Engineering (Sections 19–25)

| Feature Group | Planned | Status |
|---|---|---|
| Disease lag features (lag 1–4) | ✅ | `cases_lag_1`, `rain_lag_1`, etc. visible in training script |
| Rolling disease features (2w/4w/8w) | ✅ | Confirmed in feature columns |
| Weather lag (rainfall/humidity/temperature) | ✅ | Present |
| Weather rolling (7d/14d/28d) | ✅ | Present |
| Seasonality (week_of_year, month, sin/cos) | ✅ | Confirmed |
| Population features | ✅ | Joined to H3 grid |
| Spatial neighbor features (neighbor_case_mean, etc.) | ✅ | `h3_adjacency_res7.npz` used for spatial neighbors |
| Environmental features (NDVI, LST, Land Cover) | 🟡 | Raw folders exist; unclear if fed into final feature set |
| **Feature metadata / documentation** | ❌ | No `feature_metadata` file found |

---

## 7. Baseline Models (Sections 26, 34)

| Planned | Status | Notes |
|---|---|---|
| Persistence Baseline | ✅ | "Baseline 1: Naive Persistence" in `train_forecasting_models.py` |
| Seasonal Baseline | ✅ | "Baseline 2: Historical Seasonal Mean" |
| Coarse District Model (Baseline 3) | ✅ ➕ | "Baseline 3: Coarse District Aggregation" — not in plan, added as extra |
| Random Forest | ❌ | Plan mentions RF; only LightGBM + XGBoost trained |

---

## 8. Primary ML Models (Sections 27–33)

| Planned | Status | Notes |
|---|---|---|
| LightGBM | ✅ | Trained for all 4 horizons × 2 diseases = 8 models |
| XGBoost | ✅ | Same — 8 models |
| Temporal train/val/test split | ✅ | Chronological split used |
| **Spatial holdout** (unseen districts) | ✅ ➕ | Not in plan — extra validation dimension added |
| Cross-validation (time-aware folds) | ❌ | Plan recommends it; not visible in implementation |
| Metrics: MAE, RMSE, R², RMSLE, Corr | ✅ | All computed in `compute_metrics()` |
| MAPE | ❌ | Excluded (mentioned but appropriately skipped for zero-heavy data) |

---

## 9. Explainability — SHAP (Sections 35–37)

| Planned | Status | Notes |
|---|---|---|
| Global SHAP (feature importance across dataset) | ✅ | `shap_values_dengue_lead*.csv`, `shap_values_malaria_lead*.csv` in outputs |
| Local SHAP (per cell explanation) | ✅ | Same files contain per-row SHAP values |
| SHAP summary plot | 🟡 | `plot_forecasting_results.py` exists; SHAP plots not confirmed |
| **Dedicated `src/explainability/` module** | ❌ | Folder is **empty** — only a `__pycache__/` in there |

---

## 10. Uncertainty (Sections 38–39)

| Planned | Status | Notes |
|---|---|---|
| Prediction intervals (lower/upper bounds) | ✅ | `prediction_intervals_dengue_lead*.csv` and malaria equivalents exist |
| All 4 horizons × 2 diseases | ✅ | 8 interval files present |
| Uncertainty method documented | ❌ | No documentation on which method was used (quantile regression, conformal, bootstrap?) |
| Uncertainty evaluation / coverage metrics | ❌ | No coverage evaluation found |

---

## 11. Risk Calibration (Section 40)

| Planned | Status |
|---|---|
| Calibration curve / Brier score / Reliability diagram | ❌ Not implemented |

---

## 12. Risk Classification (Section 41)

| Planned | Status | Notes |
|---|---|---|
| Risk categories (Low/Moderate/High/Very High) | 🟡 | Alert levels (0–3) used in dashboard instead of the planned 0.00–1.00 scale thresholds |

---

## 13. Spatial Hotspot Detection (Sections 42–51)

| Planned | Status | Notes |
|---|---|---|
| Getis-Ord Gi* Z-score & p-value per cell | ✅ | `gi_zscore_pred_lead_*` columns in hotspot parquet files |
| Hotspot classification (strong/moderate/none/coldspot) | ✅ | `hotspot_taxonomy_2024.parquet` files exist for both diseases |
| Hotspot output with h3_id, risk, gi_zscore, p_value, hotspot_status | ✅ | Present in hotspot parquets |
| All 4 horizons × both diseases | ✅ | `dengue_hotspots_lead1–4.parquet`, same for malaria |
| Hotspot persistence tracking (emerging/expanding/persistent/declining) | 🟡 | Taxonomy files suggest categories exist, but logic not confirmed |
| Hotspot evolution fields (first_detection_week, last_detection_week, duration) | 🟡 | Unclear from file names alone |
| Hotspot Validation (Precision/Recall/F1/IoU) | 🟡 | `dengue_hotspots_test_2023_2024.parquet` exists — but `src/evaluation/` is empty (no validation runner) |
| Spatial IoU calculation | ❌ | No IoU script found |
| Early Warning Lead Time measurement | ❌ | No lead time measurement script found |

---

## 14. Dengue/Malaria Fusion (Sections 52–53)

| Planned | Status | Notes |
|---|---|---|
| Separate dengue & malaria hotspot outputs | ✅ | Both produced separately |
| Combined Risk Score / Fusion Layer | ❌ | No fusion score / overlay implemented |
| Combined state categories (low/high dengue × low/high malaria) | ❌ | Not implemented |

---

## 15. Alert System (not explicitly in plan as section but implied)

| Item | Status | Notes |
|---|---|---|
| Alert generation script | ✅ ➕ | `generate_alerts.py` — extra feature beyond plan |
| Per-hexagon alert levels (0–3) | ✅ ➕ | `alerts_dengue_latest.json`, `alerts_malaria_latest.json` |
| Alert summary by state | ✅ ➕ | `alerts_dengue_by_state.csv`, `alert_summary.json` |

---

## 16. Backend — FastAPI (Sections 54–59)

| Planned | Status | Notes |
|---|---|---|
| FastAPI backend | ❌ | **No FastAPI backend built at all** |
| `GET /api/health` | ❌ | |
| `GET /api/risk` | ❌ | |
| `GET /api/forecast` | ❌ | |
| `GET /api/hotspots` | ❌ | |
| `GET /api/cell/{h3_id}` | ❌ | |
| `GET /api/cell/{h3_id}/shap` | ❌ | |
| `GET /api/analytics` | ❌ | |
| `GET /api/metadata` | ❌ | |

> [!CAUTION]
> The entire FastAPI backend is missing. This is a core deliverable of Phase 11.

---

## 17. Frontend — React + MapLibre (Sections 60–71)

| Planned | Status | Notes |
|---|---|---|
| React Application | ❌ | **No React frontend exists** |
| MapLibre GL JS | ❌ | |
| H3 hexagon visualization | ❌ | |
| Forecast page (+1/+2/+3/+4 horizon selector) | ❌ | |
| Hotspot page with Gi* stats | ❌ | |
| Explainability page with SHAP chart | ❌ | |
| Analytics page | ❌ | |
| Temporal animation / timeline slider | ❌ | |
| Cell detail panel (on click) | ❌ | |
| Layer toggles (risk/hotspot/rainfall/population/boundaries) | ❌ | |
| Tailwind CSS styling | ❌ | |

> [!CAUTION]
> The entire React + MapLibre frontend (Phase 12) is not built. This is the primary final product interface.

---

## 18. Streamlit Dashboard (Section 72 — "Research Tool")

| Planned | Status | Notes |
|---|---|---|
| Streamlit as research/dev tool | ✅ | Explicitly planned as secondary tool |
| PyDeck map (H3 hexagons) | ✅ | `app.py` — PolygonLayer with H3 cells |
| Disease toggle (Dengue/Malaria) | ✅ | Sidebar selectbox |
| Forecast horizon slider (1–4 weeks) | ✅ | Present in app.py |
| Alert level view | ✅ | Color-coded by alert level |
| Risk heatmap view (Gi* z-score) | ✅ | Risk heatmap mode |
| Simplified Plotly version | ✅ ➕ | `app_simple.py` — extra fallback dashboard |
| Overview, By State, Top Alerts, Data Table tabs | ✅ ➕ | In `app_simple.py` |
| Cell detail panel (click-to-inspect) | ❌ | Not implemented in dashboard |

---

## 19. Repository Structure (Section 73)

| Planned | Status | Notes |
|---|---|---|
| `data/raw/`, `data/processed/` | ✅ | Present (no `data/interim/` though) |
| `src/data/`, `src/features/`, `src/models/`, `src/spatial/`, `src/explainability/`, `src/evaluation/`, `src/utils/` | 🟡 | Structure is `src/data_prep/`, `src/models/`, `src/dashboard/`, `src/disaggregation/`, `src/explainability/`(empty), `src/evaluation/`(empty) |
| `outputs/predictions/`, `outputs/hotspots/`, `outputs/explainability/`, `outputs/metrics/`, `outputs/figures/` | 🟡 | Has `outputs/hotspots/`, `outputs/explainability/`, `outputs/alerts/`; missing `outputs/metrics/` and `outputs/figures/` |
| `backend/` directory (FastAPI) | ❌ | Does not exist |
| `frontend/` directory (React) | ❌ | Does not exist |
| `notebooks/` directory | ❌ | Not present |
| `tests/` directory | ❌ | Not present |
| `docs/` directory | ❌ | Not present |
| `requirements.txt` | ❌ | Not found |
| `README.md` | ❌ | Not found |

---

## 20. Experiment Tracking & Studies (Sections 74–78)

| Planned | Status |
|---|---|
| Experiment log (EXP-001, etc.) | ❌ |
| Feature ablation study (A–E feature groups) | ❌ |
| Spatial resolution experiment | ❌ |
| Forecast horizon comparison | ✅ (implicitly done via 4 horizons) |
| Model explainability evaluation (SHAP reasonableness check) | 🟡 (SHAP computed, not formally reviewed) |

---

## 21. Testing Plan (Section 81)

| Test Category | Status |
|---|---|
| Data tests (schema, missing, date order, H3 validity) | ❌ |
| ML tests (prediction shape, NaN check, leakage, model loading) | 🟡 `verify_forecasting_models.py` exists |
| Spatial tests (H3 polygon validity, Gi* calc, neighbors, hotspot) | ❌ |
| API tests (health, risk, forecast, hotspots, cell, shap) | ❌ (API doesn't exist) |
| Frontend tests (map, cell selection, filters, forecast, SHAP panel) | ❌ (Frontend doesn't exist) |

---

## 22. Non-Functional Requirements

| Requirement | Planned Section | Status |
|---|---|---|
| Error handling in frontend | §82 | ❌ (No frontend) |
| Loading states / skeletons | §83 | ❌ |
| Responsive design (desktop/tablet/mobile) | §84 | ❌ |
| Accessibility (contrast, keyboard nav, labels) | §85 | ❌ |
| Security (env vars, no exposed keys) | §86 | ❌ |
| Map performance / vector tiles | §87 | ❌ |
| Backend caching | §88 | ❌ |

---

## 23. Deployment (Section 89)

| Planned | Status |
|---|---|
| Deployment architecture (React static + FastAPI) | ❌ |

---

## 24. MVP Checklist (Section 90)

| MVP Item | Status |
|---|---|
| ✓ H3 spatial grid | ✅ |
| ✓ Processed disease data | ✅ |
| ✓ Weather features | ✅ |
| ✓ Population features | ✅ |
| ✓ +1 week model | ✅ |
| ✓ Risk prediction | ✅ |
| ✓ Prediction interval | ✅ |
| ✓ SHAP | ✅ |
| ✓ Getis-Ord Gi* | ✅ |
| ✓ Hotspot detection | ✅ |
| ✓ **FastAPI** | ❌ |
| ✓ **React** | ❌ |
| ✓ **MapLibre** | ❌ |
| ✓ **H3 map visualization** | ❌ (only Streamlit/PyDeck) |
| ✓ Cell detail panel | ❌ |
| ✓ Basic forecast page | ❌ (Streamlit has it, React doesn't) |

> MVP is **~60% complete** — the ML/data science pipeline is solid, but the API + React frontend are entirely missing.

---

## 25. Version 2 Features (Section 91) Status

| Feature | Status |
|---|---|
| +2/+3/+4 week forecast | ✅ |
| Temporal animation | ❌ |
| Malaria model | ✅ |
| Dengue/Malaria fusion | ❌ |
| Hotspot evolution | 🟡 (taxonomy parquets exist) |
| Hotspot validation (formal) | ❌ |
| Advanced uncertainty | ❌ |
| Calibration | ❌ |
| Analytics page | ❌ (React) |
| Alert center | ✅ ➕ (Streamlit-based alerts) |

---

## 26. Extra Items (Not in Plan)

| Item | Location |
|---|---|
| Disaggregation prototype (synthetic proof-of-concept) | `src/disaggregation/disaggregation_prototype.py` |
| `app_simple.py` — Plotly fallback dashboard | `src/dashboard/app_simple.py` |
| Baseline 3 (Coarse District Model) | `train_forecasting_models.py` |
| Spatial holdout test set (unseen districts) | Feature datasets |
| `hex_grid_cache.gpkg` (spatial cache) | `data/raw/` |
| Bhopal wards GeoJSON (city-level detail) | `data/raw/bhopal_wards.geojson` |
| Alert generation pipeline | `src/dashboard/generate_alerts.py` |

---

## 27. Summary Table by Development Phase

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Data Foundation | ✅ Done |
| Phase 2 | H3 Spatial Dataset | ✅ Done |
| Phase 3 | Feature Engineering | ✅ Done |
| Phase 4 | Baselines | ✅ Done |
| Phase 5 | Main ML Model | ✅ Done |
| Phase 6 | Forecasting (all horizons) | ✅ Done |
| Phase 7 | SHAP | ✅ Output exists; `src/explainability/` empty |
| Phase 8 | Uncertainty | ✅ Output exists; method undocumented |
| Phase 9 | Spatial Hotspots | ✅ Done |
| Phase 10 | Validation | 🟡 Test data exists; formal validation script missing |
| Phase 11 | FastAPI Backend | ❌ **Not Started** |
| Phase 12 | React + MapLibre Frontend | ❌ **Not Started** |
| Phase 13 | Integration (React ↔ FastAPI ↔ ML) | ❌ **Not Started** |
| Phase 14 | Finalization (tests, docs, polish) | ❌ **Not Started** |

---

## 28. What is Pending (Prioritized)

> [!IMPORTANT]
> The following are the critical gaps between plan and implementation:

### 🔴 Critical — Core Deliverables Missing
1. **FastAPI backend** (Phase 11) — all 8+ planned endpoints missing
2. **React + MapLibre frontend** (Phase 12) — entire frontend not built
3. **Integration testing** (Phase 13) — nothing to integrate yet

### 🟠 Important — Scientific Completeness
4. **Hotspot validation runner** — `src/evaluation/` is empty; no Precision/Recall/F1/IoU scripts
5. **Early warning lead time measurement** — no script exists
6. **Dengue/Malaria fusion layer** — combined risk score not implemented
7. **Spatial IoU calculation** — not implemented
8. **Risk calibration** (Brier score, calibration curve) — not implemented

### 🟡 Moderate — Documentation & Quality
9. **Data quality report** — no formal QA report generated
10. **Uncertainty method documentation** — what method was used for intervals is unknown
11. **Leakage audit document** — not written
12. **Feature metadata file** — no feature dictionary
13. **Experiment tracking log** — no EXP-001 style records
14. **Feature ablation study** — not run

### 🟢 Nice to Have
15. **`requirements.txt`** — missing
16. **`README.md`** — missing
17. **`notebooks/`** directory
18. **`tests/`** directory with unit/integration tests
19. **Temporal animation** in frontend (when built)
20. **Random Forest baseline** (was in plan section 34)
21. **Cross-validation** (time-aware folds)
22. **`src/explainability/`** module code (currently empty folder)
23. **Environmental variables** (NDVI, LST, elevation) integration clarity

---

*Generated: 2026-09-30*
