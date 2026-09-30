# VectorHotspot --- Complete Project Plan

## AI-Based Dengue & Malaria Spatio-Temporal Hotspot Prediction and Early Warning System

------------------------------------------------------------------------

# 1. Project Overview

## Project Name

**VectorHotspot**

## Proposed Title

**AI-Based Dengue/Malaria Hotspot Prediction: Grid-Based Spatio-Temporal
Risk Surface and Hotspot Fusion**

## One-Line Description

VectorHotspot is a spatio-temporal machine-learning system that predicts
future dengue and malaria risk at fine geographic resolution, identifies
statistically significant spatial hotspots, explains model predictions
using SHAP, estimates prediction uncertainty, and presents the results
through an interactive React + MapLibre early-warning dashboard.

------------------------------------------------------------------------

# 2. Problem Statement

Dengue and malaria outbreaks are influenced by multiple interacting
factors:

-   Previous disease activity
-   Rainfall
-   Temperature
-   Humidity
-   Population density
-   Environmental conditions
-   Spatial proximity to other affected areas
-   Seasonal patterns

Traditional disease maps generally show **where cases have already
occurred**.

VectorHotspot aims to move from:

``` text
Past Cases
    ↓
Static Map
```

to:

``` text
Historical Disease + Weather + Environment + Population
                         ↓
                Machine Learning
                         ↓
                 Future Risk
                         ↓
              Spatial Hotspot Detection
                         ↓
               Early Warning
```

The goal is not simply to predict a number of cases. The goal is to
identify **where future risk is likely to concentrate and how early that
risk can be detected**.

------------------------------------------------------------------------

# 3. Core Research Question

> Can a spatio-temporal machine-learning model use historical disease,
> meteorological, environmental, population, and spatial information to
> forecast future dengue/malaria risk and identify statistically
> significant hotspots before they become apparent in observed disease
> data?

------------------------------------------------------------------------

# 4. Main Objectives

## Objective 1 --- Build a spatio-temporal dataset

Combine:

-   Disease cases
-   Weather
-   Rainfall
-   Humidity
-   Temperature
-   Population
-   Environmental variables
-   Geographic boundaries

into a common:

``` text
H3 Cell × Week
```

dataset.

------------------------------------------------------------------------

## Objective 2 --- Forecast future disease risk

Predict risk for:

``` text
+1 week
+2 weeks
+3 weeks
+4 weeks
```

The initial implementation should prioritize **+1 week** and then expand
to longer horizons.

------------------------------------------------------------------------

## Objective 3 --- Detect spatial hotspots

Use predicted risk values with spatial statistics such as:

**Getis-Ord Gi\***

to identify statistically significant clusters of high predicted risk.

------------------------------------------------------------------------

## Objective 4 --- Explain predictions

Use:

**SHAP**

to answer:

> Why does the model think this location is high risk?

------------------------------------------------------------------------

## Objective 5 --- Quantify uncertainty

Generate prediction intervals so the system communicates:

``` text
Predicted risk
+
Expected uncertainty
```

rather than presenting predictions as exact values.

------------------------------------------------------------------------

## Objective 6 --- Validate early-warning capability

Compare:

``` text
Predicted hotspot
        VS
Observed future hotspot
```

using spatial and temporal evaluation metrics.

------------------------------------------------------------------------

## Objective 7 --- Build an interactive frontend

Build a production-style web dashboard using:

-   React
-   MapLibre GL JS
-   H3
-   Plotly/Recharts
-   FastAPI

------------------------------------------------------------------------

# 5. Final System Architecture

``` text
                           VECTORHOTSPOT
                                |
          +---------------------+----------------------+
          |                                            |
          v                                            v
    DATA PIPELINE                               USER INTERFACE
          |                                            |
          v                                            v
 Disease Data                                  React Frontend
 Weather Data                                  MapLibre
 Population                                    H3 Visualization
 Environment                                   Charts
 Geography                                     Filters
          |                                    Explainability
          v                                            |
    DATA PROCESSING                                     |
          |                                              |
          v                                              |
      H3 × WEEK                                         |
          |                                              |
          v                                              |
  FEATURE ENGINEERING                                   |
          |                                              |
          v                                              |
 LightGBM / XGBoost                                     |
          |                                              |
          +-------------------+--------------------------+
                              |
                    Future Risk Prediction
                              |
              +---------------+---------------+
              |                               |
              v                               v
            SHAP                      Prediction Interval
              |                               |
              +---------------+---------------+
                              |
                              v
                       Risk Surface
                              |
                              v
                       Getis-Ord Gi*
                              |
                              v
                       Hotspot Detection
                              |
                              v
                       Hotspot Validation
                              |
                              v
                     Dengue/Malaria Fusion
                              |
                              v
                            FastAPI
                              |
                              v
                        React Dashboard
```

------------------------------------------------------------------------

# 6. System Layers

The complete system has six major layers.

``` text
1. Data Layer
2. Spatial-Temporal Processing Layer
3. ML Prediction Layer
4. Explainability & Uncertainty Layer
5. Spatial Hotspot & Validation Layer
6. Frontend / Visualization Layer
```

------------------------------------------------------------------------

# 7. Data Layer

## 7.1 Disease Data

Primary target data:

### Dengue

Required fields where available:

``` text
Date / Week
Location
District
State
Dengue Cases
```

### Malaria

``` text
Date / Week
Location
District
State
Malaria Cases
```

The exact geographic level depends on the available source.

------------------------------------------------------------------------

# 8. Weather Data

Potential variables:

``` text
Temperature
Humidity
Rainfall
```

Recommended derived variables:

``` text
7-day rainfall
14-day rainfall
28-day rainfall

7-day average temperature
14-day average temperature
28-day average temperature

7-day humidity
14-day humidity
28-day humidity
```

The project should preserve the original values as well as derived
features where useful.

------------------------------------------------------------------------

# 9. Rainfall Data

Rainfall is particularly important because mosquito breeding conditions
are strongly affected by water accumulation.

Possible source:

**CHIRPS**

Use rainfall at the appropriate spatial and temporal resolution and
aggregate it to:

``` text
H3 Cell × Week
```

------------------------------------------------------------------------

# 10. Population Data

Population can be obtained from a gridded source such as:

**WorldPop**

Useful variables:

``` text
Population
Population density
```

Population should be spatially joined to the H3 grid.

The project has already used processed population data in the working
pipeline.

------------------------------------------------------------------------

# 11. Environmental Data

Optional but valuable variables:

``` text
NDVI
Land Surface Temperature
Land Cover
Water Bodies
Elevation
Urbanization
```

These should be introduced only if data quality and project scope allow.

Do not add environmental variables merely for complexity.

------------------------------------------------------------------------

# 12. Geographic Data

Required:

``` text
Country boundaries
State boundaries
District boundaries
```

These are used for:

-   Map context
-   Aggregation
-   Filtering
-   Geographic interpretation

The primary modelling unit remains the H3 cell.

------------------------------------------------------------------------

# 13. Spatial Representation --- H3

The project should use **H3 hexagonal cells** as the common spatial
representation.

Instead of:

``` text
District × Month
```

the system uses:

``` text
H3 Cell × Week
```

Example:

``` text
Cell A | Week 1
Cell A | Week 2
Cell A | Week 3

Cell B | Week 1
Cell B | Week 2
Cell B | Week 3
```

This creates a consistent spatio-temporal modelling grid.

------------------------------------------------------------------------

# 14. Choosing H3 Resolution

The H3 resolution should be selected based on:

-   Disease data spatial resolution
-   Weather data resolution
-   Population data resolution
-   Computational cost
-   Number of observations
-   Geographic study area

Do not select a very fine resolution if the underlying disease data
cannot support it.

The chosen resolution should be documented as part of the experiment.

------------------------------------------------------------------------

# 15. Data Processing Pipeline

``` text
Raw Sources
    ↓
Download / Import
    ↓
Schema Standardization
    ↓
Date Standardization
    ↓
Geographic Standardization
    ↓
Missing Value Handling
    ↓
Outlier Checks
    ↓
Spatial Alignment
    ↓
Temporal Alignment
    ↓
H3 Aggregation
    ↓
H3 × Week Dataset
```

------------------------------------------------------------------------

# 16. Data Quality Checks

Every dataset should be checked for:

-   Missing values
-   Duplicate records
-   Invalid dates
-   Invalid coordinates
-   Negative case counts
-   Impossible weather values
-   Geographic mismatches
-   Missing districts
-   Temporal gaps

Create a data-quality report.

Example:

``` text
Dataset
Rows
Columns
Missing %
Duplicate %
Date range
Spatial coverage
```

------------------------------------------------------------------------

# 17. Target Definition

The target must represent **future disease risk**.

For a +1 week model:

``` text
Features at week t
        ↓
Target at week t+1
```

For example:

``` text
Rainfall(t)
Cases(t)
Humidity(t)
Population(t)
        ↓
Predict Cases(t+1)
```

For longer horizons:

``` text
t → t+2
t → t+3
t → t+4
```

Each forecast horizon should be clearly defined.

------------------------------------------------------------------------

# 18. Avoiding Data Leakage

This is one of the most important technical requirements.

The model must never receive information from the future.

For example, when predicting:

``` text
Week 40
```

features must not contain:

``` text
Week 41
Week 42
```

or any data derived from them.

All rolling and lag features must be calculated using information
available at prediction time.

------------------------------------------------------------------------

# 19. Feature Engineering

## 19.1 Disease History

Important features:

``` text
cases_lag_1
cases_lag_2
cases_lag_3
cases_lag_4
```

These represent recent disease activity.

------------------------------------------------------------------------

## 19.2 Rolling Disease Features

Examples:

``` text
cases_rolling_2w
cases_rolling_4w
cases_rolling_8w
```

Possible statistics:

``` text
rolling_mean
rolling_max
rolling_sum
rolling_std
```

------------------------------------------------------------------------

# 20. Weather Lag Features

For rainfall:

``` text
rainfall_lag_1
rainfall_lag_2
rainfall_lag_3
```

For humidity:

``` text
humidity_lag_1
humidity_lag_2
```

For temperature:

``` text
temperature_lag_1
temperature_lag_2
```

The exact lag structure should be selected through experiments.

------------------------------------------------------------------------

# 21. Weather Rolling Features

Examples:

``` text
rainfall_7d
rainfall_14d
rainfall_28d

humidity_7d
humidity_14d
humidity_28d

temperature_7d
temperature_14d
temperature_28d
```

These capture accumulated environmental conditions rather than only the
latest observation.

------------------------------------------------------------------------

# 22. Seasonality Features

Include:

``` text
week_of_year
month
season
```

Potential cyclic encoding:

``` text
sin_week
cos_week
```

This can help tree or other models capture recurring seasonal patterns.

------------------------------------------------------------------------

# 23. Population Features

Potential features:

``` text
population
population_density
```

These provide exposure-related context.

------------------------------------------------------------------------

# 24. Spatial Features

Disease risk is not independent between neighboring locations.

Potential spatial features:

``` text
neighbor_case_mean
neighbor_risk_mean
neighbor_case_sum
neighbor_population
```

These must be calculated using only information available at prediction
time.

------------------------------------------------------------------------

# 25. Final Feature Groups

The model can be organized conceptually into:

``` text
Disease History
       +
Weather
       +
Rainfall
       +
Population
       +
Environment
       +
Seasonality
       +
Spatial Features
```

------------------------------------------------------------------------

# 26. Baseline Models

Before the final model, establish simple baselines.

## Persistence Baseline

``` text
Future cases ≈ Current cases
```

## Seasonal Baseline

``` text
Future risk ≈ Historical seasonal average
```

These provide reference points.

------------------------------------------------------------------------

# 27. Primary ML Model

The primary model family should be:

**LightGBM and/or XGBoost**

These are suitable because they handle:

-   Nonlinear relationships
-   Mixed feature types
-   Feature interactions
-   Missing values
-   Large tabular datasets
-   Feature importance
-   SHAP explanations

Start with one primary model to keep the project manageable.

------------------------------------------------------------------------

# 28. Model Training

The model receives:

``` text
H3 Cell
Week
Disease history
Weather
Population
Environment
Seasonality
Spatial features
```

and predicts:

``` text
Future disease risk
```

------------------------------------------------------------------------

# 29. Temporal Train/Validation/Test Split

Do not randomly split spatio-temporal data.

Use chronological splitting.

Example:

``` text
Past ----------------------------> Future

TRAIN          VALIDATION          TEST
|--------------|-------------------|
```

Example percentages can be:

``` text
70% Train
15% Validation
15% Test
```

The exact dates should be based on the available dataset.

------------------------------------------------------------------------

# 30. Cross-Validation

If computationally feasible, use time-aware validation.

Possible strategy:

``` text
Fold 1:
Train → Validate

Fold 2:
Train --------> Validate

Fold 3:
Train ----------------> Validate
```

Never allow future observations to enter earlier training folds.

------------------------------------------------------------------------

# 31. Prediction Horizons

The recommended progression:

``` text
Phase 1:
+1 week

Phase 2:
+2 weeks

Phase 3:
+3 weeks

Phase 4:
+4 weeks
```

The +1 week model should be fully validated before expanding to all
horizons.

------------------------------------------------------------------------

# 32. Model Evaluation

## Regression Metrics

If predicting continuous cases/risk:

``` text
MAE
RMSE
R²
```

Potentially:

``` text
MAPE
```

only where appropriate, because MAPE can behave poorly around zero.

------------------------------------------------------------------------

# 33. Classification Metrics

If predictions are converted to high-risk vs non-high-risk:

``` text
Precision
Recall
F1
ROC-AUC
PR-AUC
```

For rare high-risk events, PR-AUC can be particularly informative.

------------------------------------------------------------------------

# 34. Model Comparison

Compare:

``` text
Persistence baseline
Seasonal baseline
Random Forest
XGBoost
LightGBM
```

The objective is to demonstrate whether the proposed approach improves
upon simpler alternatives.

Do not select a model only because it produces a visually attractive
map.

------------------------------------------------------------------------

# 35. SHAP Explainability

SHAP answers:

> Why did the model make this prediction?

For each prediction:

``` text
Prediction
    |
    +-- Previous cases     +0.23
    +-- Rainfall           +0.17
    +-- Humidity           +0.11
    +-- Population         +0.06
    +-- Temperature        +0.03
```

------------------------------------------------------------------------

# 36. Global SHAP

Global SHAP should show:

> Which features influence the model most across the dataset?

Outputs:

-   SHAP summary plot
-   Feature importance
-   Feature distribution

------------------------------------------------------------------------

# 37. Local SHAP

For a selected H3 cell:

> Why is this particular cell high risk?

Show:

``` text
Prediction = 0.84

Previous cases       +0.23
Rainfall 14d         +0.17
Humidity             +0.11
Population density   +0.06
Temperature          +0.03
```

------------------------------------------------------------------------

# 38. Prediction Uncertainty

The model should provide uncertainty in addition to its point
prediction.

Example:

``` text
Prediction: 0.84
Lower bound: 0.73
Upper bound: 0.91
```

The frontend should communicate:

``` text
84% predicted
Expected interval: 73–91%
```

------------------------------------------------------------------------

# 39. Prediction Intervals

Potential approaches include:

-   Quantile regression
-   Conformal prediction
-   Bootstrap-based uncertainty
-   Model ensembles

The exact method should be chosen based on the existing modelling
pipeline and validation requirements.

For the project, the important requirement is:

> The uncertainty method must be clearly documented and evaluated.

------------------------------------------------------------------------

# 40. Risk Calibration

If the output is interpreted as probability, evaluate whether predicted
probabilities are calibrated.

Possible tools:

``` text
Calibration curve
Brier score
Reliability diagram
```

Calibration should be treated separately from ordinary prediction
accuracy.

------------------------------------------------------------------------

# 41. Risk Classification

Convert continuous risk into understandable categories.

Example project-defined categories:

``` text
0.00–0.30   Low
0.30–0.60   Moderate
0.60–0.80   High
0.80–1.00   Very High
```

The final thresholds should be validated or clearly documented rather
than presented as medically established thresholds.

------------------------------------------------------------------------

# 42. Spatial Hotspot Detection

After prediction:

``` text
Predicted Risk
      ↓
Risk Surface
      ↓
Getis-Ord Gi*
      ↓
Hotspot / Coldspot Detection
```

This answers:

> Are high predicted values spatially clustered?

------------------------------------------------------------------------

# 43. Getis-Ord Gi\*

For each H3 cell, calculate:

``` text
Gi* Z-score
p-value
```

Interpretation:

``` text
High positive Z-score
        ↓
High-value spatial cluster

Low negative Z-score
        ↓
Low-value spatial cluster
```

Statistical significance should be evaluated using an appropriate
spatial-statistics methodology.

------------------------------------------------------------------------

# 44. Hotspot Output

Each cell can contain:

``` text
h3_id
risk
gi_zscore
p_value
hotspot_status
forecast_week
```

Example:

``` text
Cell A
Risk = 0.91
Gi* = 4.2
p < 0.01
Hotspot = YES
```

------------------------------------------------------------------------

# 45. Risk vs Hotspot

The system must distinguish:

``` text
HIGH RISK
```

from:

``` text
STATISTICALLY SIGNIFICANT HOTSPOT
```

A cell can have high predicted risk but not belong to a statistically
significant spatial cluster.

This distinction is central to the project.

------------------------------------------------------------------------

# 46. Hotspot Categories

Possible frontend classifications:

``` text
Strong hotspot
Moderate hotspot
Not significant
Coldspot
```

The actual thresholds should follow the chosen statistical significance
criteria.

------------------------------------------------------------------------

# 47. Hotspot Validation

Compare predicted future hotspots with observed disease hotspots.

``` text
Predicted Hotspot
        VS
Observed Hotspot
```

Evaluate:

``` text
Precision
Recall
F1
Spatial IoU
```

------------------------------------------------------------------------

# 48. Spatial IoU

For predicted hotspot area P and observed hotspot area O:

``` text
IoU = |P ∩ O| / |P ∪ O|
```

This measures spatial overlap.

------------------------------------------------------------------------

# 49. Early Warning Lead Time

Measure how early the model detects a hotspot.

Example:

``` text
Prediction
Week 20
    |
    |------ 3 weeks ------|
    |
Observed hotspot
Week 23
```

Lead time:

``` text
3 weeks
```

Report:

``` text
Mean lead time
Median lead time
Distribution of lead time
```

------------------------------------------------------------------------

# 50. Hotspot Persistence

Track whether a hotspot remains significant.

Example:

``` text
Week 1   Emerging
Week 2   Strong
Week 3   Strong
Week 4   Persistent
```

Possible outputs:

``` text
hotspot_duration
first_detection_week
last_detection_week
```

------------------------------------------------------------------------

# 51. Hotspot Evolution

The system should support:

``` text
Emerging
Expanding
Persistent
Declining
```

These categories should be based on clearly defined temporal rules.

------------------------------------------------------------------------

# 52. Dengue/Malaria Fusion

For each disease:

``` text
Dengue Prediction
      ↓
Dengue Hotspots

Malaria Prediction
      ↓
Malaria Hotspots
```

Then overlay the results.

Possible combined states:

``` text
Low Dengue / Low Malaria
High Dengue / Low Malaria
Low Dengue / High Malaria
High Dengue / High Malaria
```

------------------------------------------------------------------------

# 53. Combined Risk

A combined risk score should be explicitly defined rather than simply
adding arbitrary values.

Possible methods can be investigated:

``` text
Weighted combination
Normalized joint risk
Joint hotspot intersection
```

The project should document whichever method is selected.

------------------------------------------------------------------------

# 54. Backend Architecture

Use **FastAPI** as the interface between the ML pipeline and React.

``` text
React
  |
  | HTTP / JSON / GeoJSON
  v
FastAPI
  |
  +--> Risk data
  +--> Forecast data
  +--> Hotspots
  +--> SHAP
  +--> Analytics
```

------------------------------------------------------------------------

# 55. API Endpoints

Recommended endpoints:

``` text
GET /api/health

GET /api/risk
GET /api/forecast
GET /api/hotspots
GET /api/cell/{h3_id}
GET /api/cell/{h3_id}/shap
GET /api/analytics
GET /api/metadata
```

------------------------------------------------------------------------

# 56. Risk API

``` text
GET /api/risk
```

Parameters can include:

``` text
disease
week
state
district
forecast_horizon
```

Return:

``` text
GeoJSON FeatureCollection
```

------------------------------------------------------------------------

# 57. Forecast API

``` text
GET /api/forecast?horizon=1
```

Return:

``` text
forecast week
H3 cells
risk
lower bound
upper bound
```

------------------------------------------------------------------------

# 58. Cell API

``` text
GET /api/cell/{h3_id}
```

Return:

``` json
{
  "h3_id": "8928308280fffff",
  "risk": 0.84,
  "risk_level": "very_high",
  "forecast_week": "2026-W40",
  "lower_bound": 0.73,
  "upper_bound": 0.91,
  "hotspot": true,
  "gi_zscore": 3.82,
  "p_value": 0.001
}
```

------------------------------------------------------------------------

# 59. SHAP API

``` text
GET /api/cell/{h3_id}/shap
```

Return:

``` json
{
  "prediction": 0.84,
  "features": [
    {
      "name": "previous_cases",
      "shap_value": 0.23
    },
    {
      "name": "rainfall_14d",
      "shap_value": 0.17
    }
  ]
}
```

------------------------------------------------------------------------

# 60. Frontend Architecture

Use:

``` text
React
React Router
MapLibre GL JS
H3
Plotly / Recharts
Tailwind CSS
Fetch / Axios
```

------------------------------------------------------------------------

# 61. Frontend Pages

Recommended main pages:

``` text
1. Overview
2. Risk Map
3. Forecast
4. Hotspots
5. Explainability
```

Optional:

``` text
6. Analytics
7. Methodology
```

------------------------------------------------------------------------

# 62. Overview Page

The Overview page should contain:

``` text
KPI Cards
     ↓
Main Map
     ↓
Risk Trend
     ↓
Top Hotspots
     ↓
Alerts
```

Purpose:

> Give a quick understanding of the current situation.

------------------------------------------------------------------------

# 63. Risk Map Page

Main components:

``` text
Filters
MapLibre
H3 Risk Layer
Hotspot Layer
Legend
Cell Detail Panel
```

Purpose:

> Understand where risk is located.

------------------------------------------------------------------------

# 64. Forecast Page

Components:

``` text
+1 / +2 / +3 / +4 week selector
Forecast Map
Trend Chart
Prediction Interval
Temporal Slider
```

Purpose:

> Understand future risk.

------------------------------------------------------------------------

# 65. Hotspots Page

Components:

``` text
Hotspot Summary
Hotspot Map
Gi* Statistics
Hotspot Table
Evolution Timeline
Validation Metrics
```

Purpose:

> Understand spatial clustering and hotspot behavior.

------------------------------------------------------------------------

# 66. Explainability Page

Components:

``` text
Cell Selector
Prediction
SHAP Chart
Feature Explanation
Prediction Interval
```

Purpose:

> Understand why the model predicted risk.

------------------------------------------------------------------------

# 67. Analytics Page

Components:

``` text
Model Metrics
Baseline Comparison
Hotspot Metrics
Lead Time
Calibration
Ablation Results
```

Purpose:

> Research and project evaluation.

------------------------------------------------------------------------

# 68. MapLibre Design

The map should support:

``` text
Risk Layer
Hotspot Layer
Actual Cases Layer
Rainfall Layer
Population Layer
Administrative Boundaries
```

Users should be able to toggle layers.

------------------------------------------------------------------------

# 69. Map Interaction

Clicking a cell should open a panel containing:

``` text
H3 ID
Predicted Risk
Risk Level
Forecast Horizon
Prediction Interval
Hotspot Status
Gi* Z-score
p-value
SHAP Link
```

------------------------------------------------------------------------

# 70. Temporal Animation

Add a timeline:

``` text
W36 ----- W37 ----- W38 ----- W39 ----- W40
                  |
                PLAY
```

The map should update as the timeline changes.

This is particularly important because the project is spatio-temporal.

------------------------------------------------------------------------

# 71. Frontend Project Structure

``` text
frontend/
│
├── src/
│   ├── components/
│   │   ├── Map/
│   │   │   ├── RiskMap.jsx
│   │   │   ├── H3Layer.jsx
│   │   │   ├── HotspotLayer.jsx
│   │   │   └── MapControls.jsx
│   │   │
│   │   ├── Dashboard/
│   │   │   ├── KPI.jsx
│   │   │   ├── RiskCard.jsx
│   │   │   └── AlertCard.jsx
│   │   │
│   │   ├── Forecast/
│   │   │   ├── ForecastChart.jsx
│   │   │   └── ForecastSlider.jsx
│   │   │
│   │   ├── Explainability/
│   │   │   ├── ShapChart.jsx
│   │   │   └── ExplanationPanel.jsx
│   │   │
│   │   └── Hotspots/
│   │       ├── HotspotTable.jsx
│   │       └── HotspotStats.jsx
│   │
│   ├── pages/
│   │   ├── Overview.jsx
│   │   ├── RiskMap.jsx
│   │   ├── Forecast.jsx
│   │   ├── Hotspots.jsx
│   │   ├── Explainability.jsx
│   │   └── Analytics.jsx
│   │
│   ├── services/
│   │   └── api.js
│   │
│   ├── hooks/
│   │   └── useRiskData.js
│   │
│   ├── App.jsx
│   └── main.jsx
│
└── package.json
```

------------------------------------------------------------------------

# 72. Existing Streamlit Dashboard

Streamlit should not necessarily be discarded.

It can remain useful for:

``` text
ML experimentation
Data debugging
Model evaluation
Research analysis
Quick visualizations
```

React + MapLibre becomes the polished end-user interface.

``` text
Streamlit
    ↓
Research / Development

React + MapLibre
    ↓
Final Product
```

------------------------------------------------------------------------

# 73. Repository Structure

Recommended overall project organization:

``` text
VectorHotspot/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── spatial/
│   ├── explainability/
│   ├── evaluation/
│   └── utils/
│
├── outputs/
│   ├── predictions/
│   ├── hotspots/
│   ├── explainability/
│   ├── metrics/
│   └── figures/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── services/
│   │   └── schemas/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

------------------------------------------------------------------------

# 74. Experiment Tracking

Each experiment should record:

``` text
Experiment ID
Dataset version
Feature set
Model
Hyperparameters
Forecast horizon
Training period
Validation period
Test period
Metrics
Output location
```

Example:

``` text
EXP-001

Model: LightGBM
Horizon: +1 week
Features: Disease + Weather + Population
MAE: ...
RMSE: ...
F1: ...
```

------------------------------------------------------------------------

# 75. Feature Ablation Study

Perform experiments such as:

``` text
Experiment A
Disease history only

Experiment B
Disease + weather

Experiment C
Disease + weather + population

Experiment D
Disease + weather + population + spatial

Experiment E
All features
```

This demonstrates which feature groups actually contribute.

------------------------------------------------------------------------

# 76. Spatial Resolution Experiment

Compare different H3 resolutions where feasible.

Example:

``` text
Coarser H3
     VS
Medium H3
     VS
Finer H3
```

Evaluate:

-   Prediction quality
-   Hotspot quality
-   Computational cost
-   Data sparsity

------------------------------------------------------------------------

# 77. Forecast Horizon Experiment

Compare:

``` text
+1 week
+2 weeks
+3 weeks
+4 weeks
```

Expected result should be presented empirically rather than assumed.

------------------------------------------------------------------------

# 78. Model Explainability Evaluation

Check whether important model features are reasonable.

For example:

``` text
Disease history
Rainfall
Humidity
Temperature
Population
```

Do not claim causal relationships solely from SHAP.

SHAP indicates model contribution, not causation.

------------------------------------------------------------------------

# 79. Data Leakage Audit

Before final evaluation, verify:

``` text
No future cases in features
No future rainfall
No future target-derived variables
No random temporal split
No post-outbreak information leakage
```

Document this explicitly.

------------------------------------------------------------------------

# 80. Reproducibility

Maintain:

``` text
requirements.txt
environment configuration
dataset versions
random seeds
model configuration
experiment IDs
```

The final project should be reproducible from a clean environment as far
as data licensing and availability permit.

------------------------------------------------------------------------

# 81. Testing Plan

## Data Tests

Check:

``` text
Schema
Missing values
Date ordering
H3 validity
Duplicate rows
```

## ML Tests

Check:

``` text
Prediction shape
No NaN predictions
No future leakage
Model loading
Feature consistency
```

## Spatial Tests

Check:

``` text
H3 polygon validity
Gi* calculation
Neighbor relationships
Hotspot output
```

## API Tests

Check:

``` text
/health
/risk
/forecast
/hotspots
/cell
/shap
```

## Frontend Tests

Check:

``` text
Map rendering
Cell selection
Filters
Forecast switching
SHAP panel
Responsive layout
API failure states
Loading states
```

------------------------------------------------------------------------

# 82. Error Handling

The frontend should not fail silently.

Handle:

``` text
API unavailable
No data for selected week
No hotspot found
SHAP unavailable
Invalid H3 cell
Slow network
Missing forecast
```

Example:

``` text
No hotspot data available for this
forecast period.

Try another forecast horizon.
```

------------------------------------------------------------------------

# 83. Loading States

Use:

``` text
Map skeleton
Chart skeleton
KPI loading state
Cell panel loading state
```

Avoid freezing the entire page while one API request is running.

------------------------------------------------------------------------

# 84. Responsive Design

The dashboard should work on:

``` text
Desktop
Laptop
Tablet
Mobile
```

The main research/demo experience should prioritize desktop because the
map and analytics require screen space.

------------------------------------------------------------------------

# 85. Accessibility

Include:

-   Clear text labels
-   Keyboard navigation
-   Non-color-only indicators
-   Readable font sizes
-   Sufficient contrast
-   Accessible buttons
-   Tooltips where appropriate

Risk should not be communicated only through red/green colors.

------------------------------------------------------------------------

# 86. Security

The backend should not expose:

-   API keys
-   Database credentials
-   Internal filesystem paths
-   Model secrets
-   Sensitive raw data

Use environment variables for secrets.

------------------------------------------------------------------------

# 87. Performance

Map performance is important.

If the number of H3 cells becomes large:

``` text
GeoJSON
    ↓
Evaluate size
    ↓
If large:
Vector tiles / optimized GeoJSON
```

Avoid sending huge datasets to the browser unnecessarily.

Only load:

``` text
Current viewport
Selected week
Selected disease
Selected forecast horizon
```

when possible.

------------------------------------------------------------------------

# 88. Caching

Cache expensive backend outputs where appropriate.

Potential cached resources:

``` text
Forecast results
Risk GeoJSON
Hotspot results
SHAP results
```

SHAP for a cell should not be recalculated every time the user opens the
same cell if the result can safely be cached.

------------------------------------------------------------------------

# 89. Deployment Architecture

Potential final deployment:

``` text
                    Internet
                       |
            +----------+----------+
            |                     |
            v                     v
      React Frontend          FastAPI
      Static Hosting          Backend
            |                     |
            |                     v
            |                ML Outputs
            |                     |
            +----------+----------+
                       |
                       v
                  Data Storage
```

Possible deployment providers can be selected later based on project
requirements and free-tier availability.

------------------------------------------------------------------------

# 90. MVP Definition

The first complete working version should include:

``` text
✓ H3 spatial grid
✓ Processed disease data
✓ Weather features
✓ Population features
✓ +1 week model
✓ Risk prediction
✓ Prediction interval
✓ SHAP
✓ Getis-Ord Gi*
✓ Hotspot detection
✓ FastAPI
✓ React
✓ MapLibre
✓ H3 map visualization
✓ Cell detail panel
✓ Basic forecast page
```

Do not wait for every advanced feature before producing a working MVP.

------------------------------------------------------------------------

# 91. Version 2 Features

After MVP:

``` text
+2 week forecast
+3 week forecast
+4 week forecast

Temporal animation

Malaria model

Dengue/Malaria fusion

Hotspot evolution

Hotspot validation

Advanced uncertainty

Calibration

Analytics page

Alert center
```

------------------------------------------------------------------------

# 92. Suggested Development Phases

## Phase 1 --- Data Foundation

Tasks:

``` text
Collect disease data
Collect weather data
Collect population data
Collect geographic data
Standardize schemas
Perform data-quality checks
```

Deliverable:

``` text
Clean raw/intermediate datasets
```

------------------------------------------------------------------------

## Phase 2 --- H3 Spatial Dataset

Tasks:

``` text
Choose H3 resolution
Convert spatial data
Aggregate disease data
Aggregate weather data
Join population
Create H3 × Week dataset
```

Deliverable:

``` text
modeling_dataset.parquet/csv
```

------------------------------------------------------------------------

## Phase 3 --- Feature Engineering

Tasks:

``` text
Lag features
Rolling features
Seasonality
Population
Spatial features
Feature validation
Leakage audit
```

Deliverable:

``` text
feature_dataset
feature_metadata
```

------------------------------------------------------------------------

## Phase 4 --- Baselines

Tasks:

``` text
Persistence model
Seasonal baseline
Basic ML baseline
```

Deliverable:

``` text
baseline_metrics.csv
```

------------------------------------------------------------------------

## Phase 5 --- Main ML Model

Tasks:

``` text
Train LightGBM/XGBoost
Tune model
Time-based validation
Test evaluation
Save model
```

Deliverable:

``` text
model
predictions
metrics
```

------------------------------------------------------------------------

## Phase 6 --- Forecasting

Tasks:

``` text
+1 week
+2 weeks
+3 weeks
+4 weeks
```

Deliverable:

``` text
forecast outputs
```

Start with +1 week and expand only after the pipeline is stable.

------------------------------------------------------------------------

## Phase 7 --- SHAP

Tasks:

``` text
Global SHAP
Local SHAP
Feature importance
Explanation output files
```

Deliverable:

``` text
SHAP results
explainability plots
```

------------------------------------------------------------------------

## Phase 8 --- Uncertainty

Tasks:

``` text
Choose uncertainty method
Generate intervals
Evaluate coverage
Generate prediction interval files
```

Deliverable:

``` text
prediction_intervals
uncertainty metrics
```

------------------------------------------------------------------------

## Phase 9 --- Spatial Hotspots

Tasks:

``` text
Build spatial weights
Calculate Getis-Ord Gi*
Generate p-values
Classify hotspots
Generate hotspot GeoJSON
```

Deliverable:

``` text
hotspot dataset
hotspot GeoJSON
```

------------------------------------------------------------------------

## Phase 10 --- Validation

Tasks:

``` text
Predicted vs observed hotspots
Precision
Recall
F1
IoU
Lead time
Persistence
```

Deliverable:

``` text
validation report
```

------------------------------------------------------------------------

## Phase 11 --- FastAPI

Tasks:

``` text
Build API
Risk endpoint
Forecast endpoint
Hotspot endpoint
Cell endpoint
SHAP endpoint
Analytics endpoint
Health endpoint
```

Deliverable:

``` text
working backend API
```

------------------------------------------------------------------------

## Phase 12 --- React + MapLibre

Tasks:

``` text
React foundation
Routing
Dashboard layout
MapLibre
H3 layers
Filters
Cell panel
Charts
```

Deliverable:

``` text
working frontend
```

------------------------------------------------------------------------

## Phase 13 --- Integration

Connect:

``` text
React
    ↓
FastAPI
    ↓
ML outputs
```

Test complete user flow.

------------------------------------------------------------------------

## Phase 14 --- Finalization

Tasks:

``` text
Performance optimization
UI polishing
Error handling
Testing
Documentation
Screenshots
Demo preparation
Presentation
Final report
```

------------------------------------------------------------------------

# 93. Recommended Development Priority

The most important sequence is:

``` text
DATA
 ↓
H3 DATASET
 ↓
FEATURES
 ↓
+1 WEEK MODEL
 ↓
PREDICTIONS
 ↓
SHAP
 ↓
UNCERTAINTY
 ↓
GETIS-ORD Gi*
 ↓
HOTSPOT VALIDATION
 ↓
FASTAPI
 ↓
REACT + MAPLIBRE
 ↓
ADVANCED FEATURES
```

Do not build the full frontend before the prediction pipeline has stable
outputs.

------------------------------------------------------------------------

# 94. Final Project Deliverables

## Data

``` text
Processed datasets
Feature dataset
Data-quality report
```

## ML

``` text
Trained model
Baseline comparison
Predictions
Metrics
Forecast outputs
```

## Explainability

``` text
Global SHAP
Local SHAP
Feature importance
```

## Uncertainty

``` text
Prediction intervals
Coverage / uncertainty metrics
```

## Spatial Analysis

``` text
Getis-Ord Gi*
Hotspot GeoJSON
Hotspot statistics
```

## Validation

``` text
Precision
Recall
F1
Spatial IoU
Lead time
Persistence
```

## Backend

``` text
FastAPI service
API documentation
```

## Frontend

``` text
React application
MapLibre map
H3 visualization
Forecast interface
Hotspot interface
Explainability interface
```

## Documentation

``` text
README
PROJECT_PLAN.md
Methodology
Dataset documentation
API documentation
Experiment results
```

------------------------------------------------------------------------

# 95. Final User Experience

The final system should communicate the following sequence:

``` text
                    WHAT?
                      |
                Current Risk
                      |
                      v
                    WHERE?
                      |
                H3 Risk Map
                      |
                      v
                    WHEN?
                      |
              Future Forecast
                      |
                      v
                    WHY?
                      |
                 SHAP
                      |
                      v
                HOW SURE?
                      |
             Prediction Interval
                      |
                      v
              IS IT A HOTSPOT?
                      |
               Getis-Ord Gi*
                      |
                      v
              DID IT ACTUALLY
                 HAPPEN?
                      |
              Hotspot Validation
```

------------------------------------------------------------------------

# 96. Core Scientific Logic

The most important conceptual distinction in VectorHotspot is:

``` text
ML Model
    ↓
"What risk is predicted?"
```

``` text
SHAP
    ↓
"Why did the model predict it?"
```

``` text
Prediction Interval
    ↓
"How uncertain is the prediction?"
```

``` text
Getis-Ord Gi*
    ↓
"Is the predicted risk spatially clustered?"
```

``` text
Validation
    ↓
"Did the predicted hotspot correspond to future observations?"
```

The system should never treat these as the same thing.

------------------------------------------------------------------------

# 97. Final Architecture in One Diagram

``` text
                           RAW DATA
                              |
          +-------------------+-------------------+
          |                   |                   |
       Disease             Weather            Population
          |                   |                   |
          +-------------------+-------------------+
                              |
                       Environment
                              |
                              v
                   DATA CLEANING / QA
                              |
                              v
                       SPATIAL JOIN
                              |
                              v
                       H3 × WEEK
                              |
                              v
                    FEATURE ENGINEERING
                              |
             +----------------+----------------+
             |                |                |
       Disease Lags       Weather Lags     Spatial Lags
             |                |                |
             +----------------+----------------+
                              |
                              v
                    LIGHTGBM / XGBOOST
                              |
                              v
                    FUTURE PREDICTION
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
           SHAP                         UNCERTAINTY
             |                                 |
             +----------------+----------------+
                              |
                              v
                      PREDICTED RISK
                              |
                              v
                       GETIS-ORD Gi*
                              |
                              v
                       HOTSPOT MAP
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Hotspot Validation       Dengue/Malaria Fusion
                 |                         |
                 +------------+------------+
                              |
                              v
                            API
                         FastAPI
                              |
                              v
                     REACT FRONTEND
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          MapLibre          Charts          SHAP UI
             |
             v
        H3 Risk Surface
             |
             v
       EARLY WARNING SYSTEM
```

------------------------------------------------------------------------

# 98. Project Success Criteria

The project should be considered complete when it can demonstrate all of
the following:

### Data

-   Disease, weather, population, and spatial data are aligned.
-   Data quality checks are documented.
-   H3 × Week is the modelling structure.

### ML

-   Future disease risk can be predicted.
-   Time-based validation is used.
-   Baselines are compared.
-   Performance metrics are documented.

### Explainability

-   Global SHAP is available.
-   Local SHAP is available for selected cells.

### Uncertainty

-   Predictions have uncertainty estimates.
-   Prediction interval performance is evaluated.

### Spatial Analysis

-   Getis-Ord Gi\* is calculated.
-   Significant hotspots are identified.
-   Hotspot outputs can be visualized spatially.

### Early Warning

-   Predicted hotspots can be compared with future observed hotspots.
-   Lead time can be measured.

### Frontend

-   Users can explore the risk map.
-   Users can select H3 cells.
-   Users can inspect forecasts.
-   Users can inspect hotspots.
-   Users can inspect SHAP explanations.
-   Users can view uncertainty.

### Integration

``` text
ML → API → React → MapLibre
```

works end-to-end.

------------------------------------------------------------------------

# 99. Final Project Definition

**VectorHotspot is a grid-based spatio-temporal disease early-warning
system.**

Its core workflow is:

``` text
Historical Disease Data
        +
Weather
        +
Population
        +
Environmental Data
        +
Spatial Context
        ↓
       H3
        ↓
Feature Engineering
        ↓
LightGBM / XGBoost
        ↓
Future Risk Prediction
        ↓
   ┌────┴─────┐
   ↓          ↓
 SHAP     Uncertainty
   └────┬─────┘
        ↓
Predicted Risk Surface
        ↓
Getis-Ord Gi*
        ↓
Spatial Hotspots
        ↓
Hotspot Validation
        ↓
Dengue/Malaria Fusion
        ↓
FastAPI
        ↓
React + MapLibre
        ↓
Interactive Early-Warning Dashboard
```

The final system therefore combines:

**Machine Learning + Spatio-Temporal Modelling + H3 Spatial Grids +
Explainable AI + Uncertainty + Spatial Statistics + Interactive
Geospatial Visualization.**

------------------------------------------------------------------------

# 100. Immediate Next Steps

The recommended immediate implementation order is:

``` text
1. Freeze the dataset schema
2. Confirm the H3 resolution
3. Verify the H3 × Week dataset
4. Finish feature engineering
5. Complete the +1 week LightGBM/XGBoost model
6. Verify time-based evaluation
7. Finalize SHAP outputs
8. Finalize prediction intervals
9. Implement Getis-Ord Gi*
10. Generate hotspot GeoJSON
11. Build FastAPI
12. Create React + MapLibre frontend
13. Connect real API data
14. Add forecast interaction
15. Add hotspot interaction
16. Add SHAP UI
17. Add validation/analytics
18. Test the complete pipeline
19. Prepare final report and presentation
```

**Primary principle:** finish and validate the scientific pipeline
first, then expose stable outputs through the React + MapLibre
interface. This prevents frontend development from becoming dependent on
constantly changing ML outputs.
