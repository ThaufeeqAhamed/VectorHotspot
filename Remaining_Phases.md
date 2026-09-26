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

## 2. Completed Phases (Phases 1 to 6) — Audit & Exact Status

| Phase | Description | Key Deliverables / Results | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1–3** | **Data Collection & Cleaning** | 6 Tier-1 datasets cleaned: OpenDengue (State/Annual, 2010–2024), NCVBDC Malaria (District/Annual, 2000–2024), IMD Weather (District/Weekly, 2000–2024, Tmax/Tmin/Rain), WorldPop Population (District & Hex 1km, 2000–2020), 724 clean districts GeoJSON. | **COMPLETE** |
| **Phase 4** | **H3 Hexagonal Spatial Grid** | Generated 620,742 H3 Res-7 hexagons tagged with `(state, district)`. Exact coverage of all 724 districts and 36 states/UTs with zero coordinate duplicates. `data/processed/india_h3_grid_res7.csv`. | **COMPLETE** |
| **Phase 5** | **Spatial Disaggregation (Tier 1: Population)** | Poisson regression with aggregation constraint. Dengue $\beta_1 = +0.7662$ (urban), Malaria $\beta_1 = -1.2485$ (rural). **0.000000 max error (Exact mass preservation)**. `wire_disaggregation_model.py`. | **COMPLETE** |
| **Phase 6** | **Tier 2 Covariates & Enhanced Model Refit** | Downloaded & processed: MODIS NDVI (AppEEARS 2018–2020 mean), ESA WorldCover 2021 (water, tree, built, shrub fractions), JRC Global Surface Water occurrence. `data/processed/hex_tier2_covariates.csv` (28.7MB). Refit multi-covariate Poisson model. **0.000000 max error maintained**. 2024 comparative map generated in `outputs/figures/disaggregation_real_results_2024.png`. | **COMPLETE** |

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
│  PHASE 5-6: Mass-Preserving Spatial Disaggregation          │
│  (H3 Resolution 7, 620,742 Hexagons, Annual Grid)           │
│  Outputs: dengue_hex_annual.csv, malaria_hex_annual.csv    │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 7: Biophysical Temporal Disaggregation               │
│  (Annual Hexagon Totals ──► Weekly Hexagon Time-Series)     │
│  • IMD Weekly Weather (Rainfall, Tmax, Tmin, DTR)           │
│  • Temperature-dependent EIP & R0 thermal suitability curves│
│  • Mass-preserving weekly allocation: sum(weeks) == annual  │
│  Outputs: dengue_hex_weekly.parquet, malaria_hex_weekly.pq  │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 8: Spatiotemporal Feature Engineering Engine         │
│  • Temporal Lags (t-1 to t-8 weeks cases & weather)         │
│  • Rolling Window Statistics (4w, 12w moving mean/variance) │
│  • H3 Spatial Neighbor Features (k-ring 1 & 2 spillover)    │
│  • Seasonality Encodings (sin/cos week-of-year)             │
│  • Spatial & Temporal Leakage-Proof Train/Val/Test Split    │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 9: Dual-Disease Multi-Horizon Forecasting Models     │
│  • Independent LightGBM & XGBoost per disease               │
│  • Horizons: t+1, t+2, t+3, t+4 weeks ahead                 │
│  • Objective: Zero-inflated Tweedie / Poisson / NegBinomial │
│  • Baselines: Historical Seasonal, Naive Lag, Coarse-Admin  │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 10: Forecast-to-Hotspot Fusion (Getis-Ord Gi*)       │
│  • Gi* Local Spatial Autocorrelation on PREDICTED risk      │
│  • H3 Distance-decay Spatial Weight Matrix W                │
│  • Hotspot Taxonomy: Emerging, Intensifying, Persistent     │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 11: Future-Hotspot Validation & Research Evaluation  │
│  • Predicted Hotspots vs Subsequently Observed Hotspots     │
│  • Metrics: Spatial IoU, Precision, Recall, F1, Lead Time   │
│  • Bhopal Ward-Level (86 wards) Ground-Truth Cross-Check    │
│  • Ablation Studies (Tier 2 effect, H3 vs Admin neighbor)   │
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

## 4. Detailed Specification of Remaining Phases

---

### PHASE 7: Biophysical Temporal Disaggregation (Annual $\to$ Weekly)
* **Goal:** Convert annual hexagon case totals into weekly hexagon case time-series ($620,742 \text{ cells} \times 52 \text{ weeks}$) driven by real meteorological conditions, while strictly preserving annual totals ($\sum_{w=1}^{52} \hat{C}_{h, y, w} = C_{h, y}$).
* **Why this is required:** Surveillance case data is annual, but early warning requires weekly forecasts ($t+1$ to $t+4$ weeks). Naive even distribution (dividing by 52) destroys the monsoon outbreak cycle.

#### Detailed Tasks:
1. **Develop Biophysical Suitability Index ($S_{h, w}$):**
   * **Temperature-Dependent Suitability ($f_T(T)$):** Implement Briët / Mordecai thermal performance curves for vector competence:
     * *Dengue (Aedes aegypti):* Optimal transmission $26^\circ\text{C} - 32^\circ\text{C}$; extrinsic incubation period (EIP) drops sharply above $24^\circ\text{C}$; lethal thermal ceiling $>38^\circ\text{C}$.
     * *Malaria (Anopheles culicifacies / stephensi):* Transmission window $18^\circ\text{C} - 32^\circ\text{C}$, optimal $\sim 25^\circ\text{C} - 28^\circ\text{C}$.
   * **Rainfall & Lagged Hydrology ($f_R(R)$):** 2-to-6 week lagged cumulative rainfall driving larval habitat expansion.
   * **Combined Weekly Weight:**
     $$W_{h, y, w} = f_T(T_{h, y, w}) \cdot \left(1 + \gamma \cdot \log(1 + R_{h, y, w-\text{lag}})\right) \cdot \text{SeasonalPrior}(w)$$
2. **Mass-Preserving Weekly Allocation:**
   $$C_{h, y, w} = C_{h, y} \cdot \frac{W_{h, y, w}}{\sum_{w'=1}^{52} W_{h, y, w'}}$$
   If annual cases $C_{h,y} = 0$, all weeks remain 0.
3. **High-Performance Data Storage:**
   * A full weekly table across 15–25 years is $32\text{M} - 50\text{M}$ rows.
   * Store as partitioned Apache Parquet (`data/processed/dengue_hex_weekly.parquet` and `data/processed/malaria_hex_weekly.parquet`), partitioned by `year` or `state`, with `snappy` or `zstd` compression.
4. **Validation:** Check that sum over weeks equals annual totals across all 620k hexagons ($0.000000$ deviation). Verify that peak transmission weeks match known state-level peak monsoon/post-monsoon epidemiological records (e.g., Delhi dengue peak in Weeks 38–44, Kerala monsoon peaks in June/July).

---

### PHASE 8: Spatiotemporal Feature Engineering & Data Preparation
* **Goal:** Build a feature store for each hexagon $h$ at week $w$ without temporal or spatial data leakage.

#### Detailed Feature Taxonomy:
1. **Target Formulations:**
   * Primary: Case count $\hat{y}_{h, w+k}$ (for horizons $k \in \{1, 2, 3, 4\}$).
   * Secondary / Classification: Outbreak indicator ($\mathbb{I}[\text{cases} > \text{Threshold}_{h, 90\text{th}}]$) and Log-rate per 10k population.
2. **Autoregressive Case Lags & Moving Statistics:**
   * Lags: $C_{h, w-1}, C_{h, w-2}, C_{h, w-3}, C_{h, w-4}, C_{h, w-8}$.
   * Rolling means & standard deviations: $\text{mean}_{4w}, \text{mean}_{12w}, \text{std}_{4w}$.
   * Momentum / Rate of Acceleration: $\frac{C_{h, w-1} - C_{h, w-4}}{C_{h, w-4} + \epsilon}$.
3. **Lagged Meteorological Covariates (IMD Weather):**
   * Temperature: $T_{\text{max}}, T_{\text{min}}$, Diurnal Temperature Range ($\text{DTR} = T_{\text{max}} - T_{\text{min}}$) at lags $t-1, t-2, t-3, t-4$.
   * Rainfall: Weekly rainfall, 2-week rolling sum, 4-week rolling cumulative rainfall at lags $t-1, t-2, t-4, t-6$.
   * Extreme weather anomalies: $\Delta T = T_{h, w} - \bar{T}_{h, \text{historical}(w)}$.
4. **Spatial Neighbor Spillover Features (H3 $k$-ring):**
   * $k=1$ ring (6 contiguous neighbors): Mean neighbor case count at $w-1, w-2$.
   * $k=2$ ring (18 surrounding neighbors): Mean neighbor case count at $w-1$.
   * Distance-decayed spatial lag: $S_{h, w-1} = \sum_{j \in \mathcal{N}(h)} \frac{C_{j, w-1}}{d(h, j)}$.
   * **Strict Leakage Guard:** Spatial neighbor aggregations *must only use historical weeks ($w-1, w-2$)*, never concurrent week $w$.
5. **Static Environmental & Demographic Features (from Phase 5 & 6):**
   * $\log(\text{Population})$, Population Density.
   * MODIS NDVI mean.
   * ESA WorldCover: `frac_trees`, `frac_water`, `frac_built`, `frac_shrub`.
   * JRC Global Surface Water Occurrence.
6. **Cyclical Seasonality Encodings:**
   * $\sin\left(\frac{2\pi \cdot w}{52}\right), \cos\left(\frac{2\pi \cdot w}{52}\right)$.
7. **Leakage-Safe Train / Validation / Test Splitting Strategy:**
   * **Temporal Holdout:**
     * Training Set: 2010–2020 (Dengue), 2000–2020 (Malaria).
     * Validation Set (Hyperparameter tuning): 2021–2022.
     * Test Set (Unseen future evaluation): 2023–2024.
   * **Spatial Holdout (Generalizability):** 15% of geographically contiguous district clusters held out completely from training to test spatial transferability.

---

### PHASE 9: Dual-Disease Forecasting Models (LightGBM & XGBoost)
* **Goal:** Train separate gradient boosted tree models for Dengue and Malaria across 4 forecast horizons ($t+1, t+2, t+3, t+4$ weeks).

#### Detailed Tasks:
1. **Model Architecture & Loss Functions:**
   * Implement **LightGBM Regressor** and **XGBoost Regressor**.
   * Use **Tweedie Loss** ($1 < p < 2$) or **Negative Binomial / Poisson deviance** to properly model zero-inflated, right-skewed count data.
   * Direct Multi-Horizon Forecasting: Train independent models $M_{d, k}$ for disease $d \in \{\text{Dengue}, \text{Malaria}\}$ and horizon $k \in \{1, 2, 3, 4\}$.
2. **Hyperparameter Optimization:**
   * Automated tuning via Optuna (learning rate, tree depth, num_leaves, feature_fraction, min_child_samples, reg_alpha, reg_lambda).
3. **Rigorous Baseline Benchmarks:**
   * Baseline 1: *Naive Persistence Model* ($\hat{y}_{t+k} = y_t$).
   * Baseline 2: *Historical Seasonal Mean Model* ($\hat{y}_{h, w+k} = \bar{C}_{h, w+k}$).
   * Baseline 3: *Coarse-scale (District) Model with Naive Disaggregation* (train at district level, divide evenly to hexagons).
4. **Evaluation Metrics:**
   * RMSE, MAE, Root Mean Squared Logarithmic Error (RMSLE).
   * Continuous Ranked Probability Score (CRPS) or Quantile Loss.

---

### PHASE 10: Forecast-to-Hotspot Fusion (Getis-Ord $G_i^*$)
* **Goal:** Run spatial cluster and hotspot detection on the **PREDICTED future risk surface** ($\hat{Y}_{t+k}$), identifying statistically significant clusters of high risk *before* they manifest.

#### Detailed Tasks:
1. **Mathematical Formulation:**
   $$G_i^*(k) = \frac{\sum_{j=1}^N w_{ij} \hat{y}_{j, t+k} - \bar{Y} \sum_{j=1}^N w_{ij}}{S \sqrt{\frac{N \sum_{j=1}^N w_{ij}^2 - (\sum_{j=1}^N w_{ij})^2}{N - 1}}}$$
   Where $w_{ij}$ is the H3 spatial adjacency matrix ($w_{ij} = 1$ if $j \in \text{k-ring}(i, 1)$, else $0$), $\bar{Y}$ is global mean predicted risk, and $S$ is the standard deviation.
2. **Hotspot Classification Taxonomy:**
   * **Emerging Hotspot:** $G_i^* \ge 1.96$ ($p < 0.05$) at $t+k$, but was not significant at $t$.
   * **Intensifying Hotspot:** $G_i^*$ $z$-score increasing over consecutive forecast horizons $t+1 \to t+4$.
   * **Persistent Hotspot:** $G_i^* \ge 2.58$ ($p < 0.01$) across all recent historical and predicted weeks.
   * **Diminishing Hotspot:** Historical hotspot showing decreasing predicted $z$-scores.

---

### PHASE 11: Rigorous Evaluation, Ablation Studies & Ground-Truth Validation
* **Goal:** Scientifically defend the pipeline against naive approaches and prove future predictive utility.

#### Detailed Tasks:
1. **Future Hotspot Verification:**
   * Compare Predicted Hotspot Polygons at week $t$ for horizon $t+k$ against the Observed Hotspots computed from actual data at week $t+k$.
   * Calculate **Spatial Intersection over Union (Spatial IoU)**:
     $$\text{Spatial IoU} = \frac{\text{Area}(\text{Predicted Hotspot} \cap \text{Observed Hotspot})}{\text{Area}(\text{Predicted Hotspot} \cup \text{Observed Hotspot})}$$
   * Classification metrics: Precision, Recall, F1-Score, and Detection Lead Time (weeks in advance).
2. **Bhopal Ward-Level (86 Wards) Ground-Truth Validation:**
   * Aggregate H3 hexagon disaggregated estimates up to the 86 municipal wards of Bhopal (`bhopal_wards.geojson`).
   * Compare predicted intra-district spatial ranking against actual ward-level municipal population/dengue vulnerability distributions.
3. **Formal Ablation Suite:**
   * **Ablation 1 (Spatial Methodology):** Mass-Preserving Poisson vs. Naive Uniform Disaggregation vs. Population-Only Disaggregation vs. Full Tier-1+Tier-2 Covariate Disaggregation.
   * **Ablation 2 (Feature Engineering):** Model with H3 $k$-ring spatial neighbors vs. Model with administrative district neighbors vs. Model with no spatial features.
   * **Ablation 3 (Environmental Signals):** With vs. without IMD weather lagged suitability features.

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

| Phase | Phase Name | Primary Input Data | Primary Outputs / Artifacts | Estimated Tasks |
| :---: | :--- | :--- | :--- | :---: |
| **Phase 7** | **Temporal Disaggregation** | Disaggregated Annual CSVs, IMD Weekly Weather | `dengue_hex_weekly.parquet`, `malaria_hex_weekly.parquet` | 4 tasks |
| **Phase 8** | **Feature Engineering** | Weekly Parquet tables, IMD Weather, H3 Grid | Feature Store (`features_dengue.parquet`, `features_malaria.parquet`) | 5 tasks |
| **Phase 9** | **Forecasting Models** | Engineered Feature Store | Trained LightGBM/XGBoost models, Forecast Risk Surfaces ($t+1 \dots t+4$) | 4 tasks |
| **Phase 10** | **Hotspot Fusion ($G_i^*$)** | Forecast Risk Surfaces, H3 Neighbor Matrix | Hotspot classifications ($z$-scores, $p$-values, hotspot categories) | 3 tasks |
| **Phase 11** | **Evaluation & Ablations** | Predicted vs Observed Hotspots, Bhopal Wards | Precision/Recall/F1/IoU metrics, Ablation tables, Ward validation plots | 4 tasks |
| **Phase 12** | **SHAP & Dual Ecology** | Trained Models, Feature Store | SHAP summary plots, Co-hotspot ecology analysis, Calibration curves | 3 tasks |
| **Phase 13** | **Dashboard & Alerts** | All predictions, hotspots, SHAP values | Streamlit Interactive Dashboard, `alerts_summary.json` | 3 tasks |

---

*This document represents the official master specification for VectorHotspot moving forward.*
