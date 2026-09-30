from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import json

from .schemas import (
    HealthResponse, MetadataResponse, AnalyticsResponse,
    HotspotsResponse, ForecastResponse, GeoJSONResponse
)
from .data_manager import data_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data on startup
    data_manager.load_data()
    yield
    # Clean up on shutdown
    data_manager.grid_df = None
    data_manager.predictions = {}
    data_manager.latest_preds = {}

app = FastAPI(
    title="VectorHotspot API",
    description="Backend API for VectorHotspot predictions and analytics",
    version="1.0.0",
    lifespan=lifespan
)

# GZip compress responses > 1KB — cuts map payload by ~70%
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/api/ready")
def readiness_check():
    """Lightweight readiness probe the frontend polls while GeoJSON is baking."""
    return {
        "core_ready": data_manager.latest_preds != {},
        "map_ready":  data_manager.geojson_ready,
    }

@app.get("/api/metadata", response_model=MetadataResponse)
def get_metadata():
    return {
        "latest_year": data_manager.latest_year,
        "latest_week": data_manager.latest_week,
        "diseases": data_manager.api_diseases,
        "models": ["lgbm", "xgb", "rf_baseline", "naive"]
    }

@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics():
    dengue_preds = data_manager.latest_preds.get("dengue")
    total_cells = len(dengue_preds) if dengue_preds is not None else 0
    total_districts = dengue_preds['district'].nunique() if dengue_preds is not None else 0
    return {
        "total_cells": total_cells,
        "total_districts": total_districts,
        "latest_predictions": total_cells * 2 * 4  # cells * diseases * horizons
    }

@app.get("/api/risk")
def get_risk_map(
    disease: str = Query(...),
    horizon: int = Query(1, ge=1, le=4)
):
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    return data_manager.get_latest_risk_fast(disease, horizon)

@app.get("/api/geojson")
def get_risk_geojson(
    disease: str = Query(..., description="Disease name"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon (1-4 weeks)")
):
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    if not data_manager.geojson_ready:
        return Response(status_code=503, headers={"Retry-After": "10"},
                        content='{"detail":"GeoJSON cache is building, retry in ~10s"}',
                        media_type='application/json')
    entry = data_manager.get_risk_geojson(disease, horizon)
    return Response(
        content=json.dumps({'geojson': json.loads(entry['geojson']), 'maxRisk': entry['maxRisk']}),
        media_type='application/json'
    )

@app.get("/api/geojson/all")
def get_risk_geojson_all(
    disease: str = Query(..., description="Disease name")
):
    """
    Returns pre-built GeoJSON for ALL 4 horizons in one request.
    Returns 503 with Retry-After if background bake is still in progress.
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    if not data_manager.geojson_ready:
        return Response(status_code=503, headers={"Retry-After": "10"},
                        content='{"detail":"GeoJSON cache is building, please retry in ~10s"}',
                        media_type='application/json')
    result = {}
    for h in data_manager.horizons:
        entry = data_manager.get_risk_geojson(disease, h)
        result[str(h)] = {'geojson': json.loads(entry['geojson']), 'maxRisk': entry['maxRisk']}
    return Response(
        content=json.dumps({'disease': disease, 'horizons': result}),
        media_type='application/json'
    )

@app.get("/api/risk/all")
def get_risk_map_all(
    disease: str = Query(..., description="Disease name (e.g., dengue, malaria)"),
):
    """
    Returns all 4 forecast horizons in one response — enables
    zero-latency horizon switching on the frontend via pre-fetch caching.
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")

    return {
        "disease": disease,
        "horizons": {
            "1": data_manager.get_latest_risk_fast(disease, 1),
            "2": data_manager.get_latest_risk_fast(disease, 2),
            "3": data_manager.get_latest_risk_fast(disease, 3),
            "4": data_manager.get_latest_risk_fast(disease, 4),
        }
    }


@app.get("/api/hotspots", response_model=HotspotsResponse)
def get_hotspots(
    disease: str = Query(..., description="Disease name (e.g., dengue, malaria)"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon (1-4 weeks)"),
    limit: int = Query(100, ge=1, le=500)
):
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
        
    hotspots = data_manager.get_hotspots(disease, horizon, top_n=limit)
    return {
        "disease": disease,
        "horizon": horizon,
        "top_cells": hotspots
    }

@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(
    disease: str = Query(..., description="Disease name"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon (1-4 weeks)"),
    district: Optional[str] = Query(None, description="Filter by district name")
):
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
        
    history = data_manager.get_forecast_history(disease, horizon, district=district)
    return {
        "disease": disease,
        "horizon": horizon,
        "district": district,
        "history": history
    }

@app.get("/api/cell/{h3_id}", response_model=ForecastResponse)
def get_cell_forecast(
    h3_id: str,
    disease: str = Query(..., description="Disease name"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon")
):
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
        
    history = data_manager.get_forecast_history(disease, horizon, h3_index=h3_id)
    return {
        "disease": disease,
        "horizon": horizon,
        "h3_index": h3_id,
        "history": history
    }

@app.get("/api/cell/{h3_id}/shap")
def get_cell_shap(
    h3_id: str,
    disease: str = Query(..., description="Disease name"),
    horizon: int = Query(1, ge=1, le=4)
):
    """
    Returns global SHAP importance for the requested disease/horizon.
    (Local SHAP is too large to load in memory for the API currently).
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
        
    shap_records = data_manager.get_global_shap(disease, horizon)
    return {
        "h3_index": h3_id,
        "disease": disease,
        "horizon": horizon,
        "explanation": "Global SHAP importance",
        "top_features": shap_records
    }
