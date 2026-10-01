# VectorHotspot
**Epidemiological Geospatial Early-Warning Intelligence System**

VectorHotspot is an enterprise-grade machine learning pipeline and interactive dashboard designed to forecast vector-borne disease transmission (specifically **Dengue** and **Malaria**) across India. It operates at a hyper-local spatial scale using Uber's H3 hexagonal grid system, predicting transmission risks 1 to 4 weeks into the future.

---

## 🌟 Key Features

*   **Hyper-Local Forecasting**: Predicts disease risk at **H3 Resolution 7** (approx. 5.16 km² per hexagon), covering all of India (over 620,000 hexagons).
*   **Multi-Horizon Predictions**: Generates risk scores for $t+1$, $t+2$, $t+3$, and $t+4$ weeks ahead.
*   **Dual Pathology Tracking**: Dedicated ML pipelines for both Dengue and Malaria, as well as a "Syndemic" (combined) risk visualization mode.
*   **Explainable AI**: Integrates SHAP (SHapley Additive exPlanations) to break down the exact environmental and historical drivers causing risk in specific areas.
*   **Zero-Latency Geospatial Dashboard**: A custom React + MapLibre GL dashboard optimized to render up to 50,000 active risk hexagons instantly using server-side GeoJSON baking and client-side IndexedDB caching.

---

## 🏗️ System Architecture

VectorHotspot is designed around a modular data engineering and machine learning architecture:

1.  **Data Processing Engine (`src/data_prep`)**: Ingests disparate spatial-temporal data (weather, satellite, census, clinical case records) and standardizes them into a unified H3 hexagonal grid.
2.  **Feature Engineering (`src/models/feature_engineering.py`)**: Computes 40+ advanced features including temporal lags, rolling averages, adjacency matrix spreading (disease momentum in neighboring cells), and seasonal sinusoidal encodings.
3.  **Predictive Modeling (`src/models`)**: Employs Gradient Boosted Trees (**LightGBM** and **XGBoost**) for direct multi-horizon count regression, trained using strict chronological holdout sets to prevent data leakage.
4.  **FastAPI Backend (`src/backend`)**: A highly optimized Python backend that computes the top national hotspots and serves compressed, pre-baked GeoJSON to the client.
5.  **React Frontend (`src/frontend`)**: A sleek, dark-glassmorphism dashboard using Vite, MapLibre GL, and React. Features dynamic Light/Dark modes, an automated hotspot feed, and cell-level deep dives.

---

## 📂 Repository Structure

```text
VectorHotspot/
├── data/
│   ├── raw/                 # Unprocessed IMD weather, population, and disease data
│   └── processed/           # Feature matrices, H3 grid mapping, and parquets
├── models/                  # Serialized .joblib LightGBM/XGBoost model artifacts
├── outputs/                 # Hotspot logs, generated GeoJSON caches, and evaluation tables
├── src/
│   ├── data_prep/           # Scripts to ingest, clean, and align raw data to H3 grid
│   ├── disaggregation/      # Downscaling district-level data to hexagonal patches
│   ├── evaluation/          # Cross-validation, lead-time analysis, and ablation studies
│   ├── explainability/      # SHAP value calculators for model interpretation
│   ├── models/              # Model training, baseline comparisons, and feature engineering
│   ├── backend/             # FastAPI REST endpoints and GeoJSON generation
│   └── frontend/            # React/Vite interactive dashboard UI
└── README.md
```

---

## 🔬 Scientific Findings (Ablation Study)

Extensive feature ablation studies on the models revealed significant differences in the epidemiological dynamics of the two pathogens:

*   **Dengue is Autoregressive**: Dengue transmission acts like a localized wildfire. The model achieves near-peak accuracy ($R^2$ ~ 0.89) relying almost entirely on recent historical case momentum within the same hexagon. Exogenous weather data only provides marginal fine-tuning.
*   **Malaria is Environmentally Driven**: Malaria forecasting is heavily dependent on the environment. Removing seasonal and cyclical temporal features causes accuracy to crash (from $R^2$ 0.82 to 0.68). Spatial neighbor spread and ecological baselines are critical to predicting Malaria effectively.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js (v18+)

### 1. Start the Backend API (FastAPI)
The backend generates the cached geographic coordinates and serves the API for the dashboard.
```bash
# Navigate to the project root
cd VectorHotspot

# Install Python requirements (if not already installed)
pip install -r src/backend/requirements.txt

# Start the uvicorn server on port 8000
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: The first time the backend boots, it will "bake" the GeoJSON coordinates from the model outputs, which may take ~10-15 seconds. Subsequent loads are instant.*

### 2. Start the Frontend Dashboard (React + Vite)
In a separate terminal window:
```bash
# Navigate to the frontend directory
cd VectorHotspot/src/frontend

# Install node modules
npm install

# Start the Vite development server
npm run dev
```

### 3. Usage
*   Open your browser to `http://localhost:5173/`.
*   Use the **Target Date Slider** at the top right to switch between $t+1$ and $t+4$ week forecasts.
*   Use the **Disease Dropdown** to toggle between Dengue, Malaria, and the Combined Syndemic view.
*   Click on any hexagon on the map, or on any Hotspot Card in the left sidebar, to open the **Analytics Panel** and view the specific environmental drivers (via SHAP) contributing to that area's risk.

---

## 🛡️ License & Acknowledgements
Built for epidemiological research and advanced vector-borne disease forecasting. Geographic data generated using [Uber's H3](https://h3geo.org/) spatial indexing. Map rendering powered by [MapLibre GL](https://maplibre.org/).
