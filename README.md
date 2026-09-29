# VectorHotspot: Dengue-Malaria Dual-Disease Spatiotemporal Early Warning System

A fully reproducible machine learning pipeline that forecasts dengue and malaria risk across India at fine spatial resolution (H3 hexagons, ~5.2 km² each) using statistical disaggregation, biophysical temporal modelling, gradient-boosted forecasting, and Getis-Ord Gi* hotspot fusion.

## Project Status

**Current Phase:** Phase 11 (Ablation Studies & Ground-Truth Validation) — in progress

**Completed (Phases 1–10):**
- ✅ Phase 1–3: All Tier 1 data collected & cleaned (dengue, malaria, weather, boundaries, population)
- ✅ Phase 4: H3 Resolution-7 hexagonal grid (620,742 hexagons, all 724 districts covered)
- ✅ Phase 5: Spatial disaggregation — population-only Poisson model, 0.000000 mass error
- ✅ Phase 6: Tier 2 covariates (MODIS NDVI, ESA WorldCover, JRC Water) + multi-covariate refit
- ✅ Phase 7: Biophysical temporal disaggregation — annual → 747,903,370 weekly rows, Brière thermal curves, 0.000000 mass error
- ✅ Phase 8: Spatiotemporal feature engineering — 61.6M records, 52 features, zero leakage, 15% spatial holdout
- ✅ Phase 9: Dual-disease multi-horizon forecasting — LightGBM/XGBoost, R² 0.87–0.95, beats naive & seasonal baselines
- ✅ Phase 10: Forecast-to-Hotspot Fusion — vectorized Getis-Ord Gi* on predicted risk, 4-tier taxonomy, validated against ground truth

**Remaining:**
- Phase 11: Ablation studies, lead-time analysis, Bhopal ward validation
- Phase 12: SHAP explainability & dual-disease ecology
- Phase 13: Early warning engine & interactive dashboard

## Phase 10 Key Results (2023–2024 Test Set)

| Disease | Horizon | Spatial IoU | Precision | Recall | F1 |
|---------|---------|-------------|-----------|--------|----|
| Dengue | t+1 | **0.8145** | 0.8899 | 0.9058 | 0.8978 |
| Dengue | t+2 | 0.7716 | 0.8617 | 0.8806 | 0.8711 |
| Dengue | t+3 | 0.7277 | 0.8332 | 0.8518 | 0.8424 |
| Dengue | t+4 | 0.6907 | 0.8141 | 0.8201 | 0.8171 |
| Malaria | t+1 | **0.7787** | 0.8665 | 0.8849 | 0.8756 |
| Malaria | t+2 | 0.6369 | 0.7851 | 0.7714 | 0.7782 |
| Malaria | t+3 | 0.5080 | 0.7345 | 0.6223 | 0.6738 |
| Malaria | t+4 | 0.4487 | 0.6803 | 0.5685 | 0.6194 |

**4-Tier Hotspot Taxonomy (2023–2024 test set flagged hotspots):**
- Dengue: Persistent 67.7% | Diminishing 16.4% | Emerging 8.8% | Intensifying 7.1%
- Malaria: Persistent 48.3% | Diminishing 32.3% | Emerging 16.1% | Intensifying 3.3%

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scipy geopandas shapely h3 rasterio requests tqdm lightgbm xgboost pyarrow scikit-learn joblib matplotlib seaborn
```

### Run the Full Pipeline (Phases 5–10)

```bash
# Phase 5: Spatial disaggregation
python src/disaggregation/wire_disaggregation_model.py

# Phase 6: Tier 2 covariates already incorporated — rerun if covariates change
# python src/disaggregation/wire_disaggregation_with_tier2.py

# Phase 7: Biophysical temporal disaggregation
python src/disaggregation/temporal_disaggregation.py

# Phase 8: Feature engineering
python src/features/build_feature_store.py

# Phase 9: Train forecasting models
python src/models/train_forecasting_models.py

# Phase 10: Hotspot fusion
python src/hotspots/forecast_hotspot_fusion.py

# Phase 10: Generate diagnostic figure
python src/hotspots/plot_hotspots.py
```

## Repository Structure

```
VectorHotspot/
├── data/
│   ├── boundaries/
│   │   └── india_districts_clean.geojson       # 724 districts, 36 states
│   ├── raw/
│   │   └── population/                          # WorldPop .tif files (user-placed)
│   └── processed/
│       ├── dengue_state_2010_2024.csv           # State-level dengue (OpenDengue)
│       ├── malaria_district_2000_2024.csv       # District-level malaria (NCVBDC, corrected)
│       ├── india_h3_grid_res7.csv               # 620,742 hexagons
│       ├── hex_population_2000_2020.csv         # Per-hexagon WorldPop population
│       ├── hex_tier2_covariates.csv             # NDVI, land cover, water occurrence
│       ├── imd_district_weekly_weather_2000_2024_FIXED.csv
│       ├── features_dengue_train.parquet        # Feature store — training set
│       ├── features_dengue_test.parquet         # Feature store — 2023-2024 test set
│       ├── features_malaria_train.parquet
│       └── features_malaria_test.parquet
├── src/
│   ├── data_prep/
│   │   ├── download_worldpop.py
│   │   ├── generate_h3_grid.py
│   │   ├── generate_hex_population.py
│   │   ├── fetch_imd_weather.py
│   │   └── recompute_temperature_fast.py
│   ├── disaggregation/
│   │   ├── disaggregation_prototype.py          # Synthetic proof-of-concept only
│   │   └── wire_disaggregation_model.py         # Real Poisson disaggregation
│   ├── features/
│   │   └── build_feature_store.py               # Phase 8 feature engineering
│   ├── models/
│   │   ├── train_forecasting_models.py          # Phase 9 LightGBM/XGBoost training
│   │   ├── plot_forecasting_results.py
│   │   └── verify_forecasting_models.py
│   └── hotspots/
│       ├── forecast_hotspot_fusion.py           # Phase 10 Gi* engine
│       └── plot_hotspots.py                     # Phase 10 diagnostic figure
├── models/
│   └── lgbm_dengue_lead{1-4}.joblib            # Trained LightGBM models (16 files)
├── outputs/
│   ├── figures/
│   │   ├── hotspot_fusion_diagnostics.png       # Phase 10 4-panel diagnostic
│   │   └── forecasting_models_evaluation.png   # Phase 9 evaluation figure
│   ├── hotspots/
│   │   ├── dengue_hotspots_test_2023_2024.parquet   # 219.4 MB
│   │   └── malaria_hotspots_test_2023_2024.parquet  # 116.7 MB
│   └── tables/
│       ├── hotspot_fusion_evaluation_metrics.csv
│       └── model_evaluation_metrics.csv
├── PROJECT_HANDOVER.md                          # Full detailed project history
├── Remaining_Phases.md                          # Master roadmap (Phases 11-13 remaining)
└── README.md                                    # This file
```

## Methodology Summary

### End-to-End Pipeline Architecture

```
Coarse Case Data (State/District, Annual)
  + Environmental Covariates (Population, NDVI, Land Cover, Water)
  + IMD Weekly Weather (2000-2024)
          │
          ▼
 [Phase 5-6] Mass-Preserving Spatial Disaggregation
   → 620,742 H3 hexagons, exact annual totals preserved
          │
          ▼
 [Phase 7] Biophysical Temporal Disaggregation
   → Brière thermal curves + lagged hydrology
   → Annual → 52 weekly time-steps, exact totals preserved
          │
          ▼
 [Phase 8] Spatiotemporal Feature Engineering
   → Case lags, rolling stats, H3 neighbor spillover
   → Weather lags, seasonality encodings
   → 52 features, 61.6M records
          │
          ▼
 [Phase 9] Multi-Horizon Forecasting (LightGBM/XGBoost)
   → Delta formulation: predict (Y_future − Y_current)
   → Horizons t+1, t+2, t+3, t+4
          │
          ▼
 [Phase 10] Forecast-to-Hotspot Fusion (Getis-Ord Gi*)
   → Gi* computed on PREDICTED risk surfaces
   → 4-tier taxonomy: Persistent / Diminishing / Emerging / Intensifying
   → Validated vs. Gi* on actual outcomes (Spatial IoU)
```

### Core Design Decisions

1. **Dual-stage disaggregation:** Spatial (Poisson regression, population + Tier 2 covariates) then temporal (biophysical thermal curves + lagged rainfall)
2. **Mass preservation:** Exact by construction — hexagon predictions sum to known coarse totals at every stage (0.000000 max error)
3. **Delta forecasting:** Models predict `Y_future − Y_current`; adding the non-negative base ensures physically valid predictions
4. **Forecast-to-Hotspot Fusion:** Gi* is applied to *predicted* future risk, not historical case data — the key novelty
5. **Thread capping:** `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` to prevent CPU lockup on inference
6. **Sparse Gi* computation:** SciPy CSR adjacency matrix (W*) with self-loops (k-disk=1); vectorized across all 104 time slices in 2.97s

### H3 Resolution Decision

**Resolution 7** (~5.2 km² per hexagon, 620,742 hexagons over India)
- ~877× finer than district-level resolution
- Close to the practical ceiling of what real covariates (1km population, ~25km rainfall, ~100km temperature) can support
- Finer than resolution 6 (~36 km²); not so fine as to fabricate spatial precision

## Important Design Rules (Do Not Break)

1. **Always join on `(state, district)` tuples** — `dt_code` is not unique nationally
2. **`malaria_district_2000_2024.csv` is the state-mislabeling-corrected version** — never reintroduce older copies
3. **Mass preservation must be exact (0.000000 deviation)** at all disaggregation stages
4. **Large intermediate files are gitignored** — hex-annual CSVs and hex-weekly Parquets are regenerable
5. **No synthetic data in the production pipeline** — `disaggregation_prototype.py` is proof-of-concept only
6. **Path resolution via `Path(__file__).resolve().parent`** — all scripts use this pattern

## Known Limitations

- Case data is annual-only from real surveillance sources (temporal disaggregation covers this)
- 14.2% of malaria rows have unresolved (state, district) pairs — documented, not a bug
- GPU acceleration (GTX 1650, CUDA 12.7, CuPy installed) is not currently used due to missing `nvJitLink` DLL — all inference runs on CPU
- Test set (2023–2024) represents 35,818 spatial holdout hexagons (15% spatial holdout), not full India

## Data Sources

| Dataset | Source | Spatial Level | Temporal Coverage |
|---------|--------|---------------|-------------------|
| Dengue cases | OpenDengue | State | 2010–2024 |
| Malaria cases | NCVBDC PDF | District | 2000–2024 |
| Weather | IMD via imdlib | District, Weekly | 2000–2024 |
| Population | WorldPop 1km | Hexagon | 2000–2020 |
| NDVI | MODIS MOD13A2 AppEEARS | Hexagon | 2018–2020 mean |
| Land cover | ESA WorldCover 2021 | Hexagon | 2021 |
| Surface water | JRC Global Surface Water | Hexagon | 1984–2021 |
| Boundaries | udit-001/india-maps-data | District polygons | Current |

## Validation Results

### Mass Preservation
- ✅ Phase 5 (Spatial): 0.000000 max error — 451 dengue state-years, 14,402 malaria district-years
- ✅ Phase 6 (Tier 2 refit): 0.000000 max error maintained
- ✅ Phase 7 (Temporal): 0.000000 max error — 747,903,370 weekly rows

### Forecasting Models (Phase 9)
- R² 0.87–0.95 on unseen 2023–2024 test set
- Beats naive persistence and historical seasonal baselines across all horizons and both diseases

### Hotspot Detection (Phase 10)
- Dengue t+1: IoU=0.8145, Precision=0.8899, Recall=0.9058, F1=0.8978
- Malaria t+1: IoU=0.7787, Precision=0.8665, Recall=0.8849, F1=0.8756
- Gi* computed across 35,818 hexes × 104 weeks in 2.97s (vectorized sparse matrix)

### Epidemiological Plausibility
- ✅ Dengue: positive population coefficient (urban/high-density concentration)
- ✅ Malaria: negative population coefficient (rural/low-density concentration)
- ✅ Visual: dengue concentrated in metros; malaria in Odisha-Chhattisgarh-Jharkhand belt

## References & Acknowledgments

**Methodology inspired by:**
- Malaria Atlas Project (MAP) — model-based geostatistical disaggregation
- Brière et al. / Mordecai et al. — thermal performance curves for vector-borne disease

**Data sources:**
- OpenDengue project
- NCVBDC (National Centre for Vector Borne Diseases Control), India
- IMD (India Meteorological Department)
- WorldPop (University of Southampton)
- NASA MODIS (LP DAAC / AppEEARS)
- ESA WorldCover 2021
- JRC Global Surface Water
- Community-maintained India boundary data (GitHub: udit-001/india-maps-data)

## Contact & Collaboration

**GitHub:** [ThaufeeqAhamed/VectorHotspot](https://github.com/ThaufeeqAhamed/VectorHotspot) (private)

For detailed project history, methodology decisions, and debugging narratives, see [`PROJECT_HANDOVER.md`](PROJECT_HANDOVER.md).

---

*Last Updated: 2026-09-29 (Phase 10 complete)*
