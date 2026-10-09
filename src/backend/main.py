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
        return Response(status_code=503, headers={"Retry-After": "5"},
                        content='{"detail":"GeoJSON cache is building, retry in ~5s"}',
                        media_type='application/json')
    entry = data_manager.get_risk_geojson(disease, horizon)
    is_agg = str(entry.get("is_aggregate", True)).lower()
    content = f'{{"geojson":{entry["geojson"]},"maxRisk":{entry["maxRisk"]},"is_aggregate":{is_agg}}}'
    return Response(
        content=content,
        media_type='application/json'
    )

@app.get("/api/geojson/all")
@app.get("/api/geojson/overview/all")
def get_risk_geojson_all(
    disease: str = Query(..., description="Disease name")
):
    """
    Returns pre-built National Overview GeoJSON for ALL 4 horizons in one request.
    Covers 100% of India with pre-aggregated H3 Res-4 hexagons.
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    if not data_manager.geojson_ready:
        return Response(status_code=503, headers={"Retry-After": "5"},
                        content='{"detail":"GeoJSON cache is building, please retry in ~5s"}',
                        media_type='application/json')
    horizons_parts = []
    for h in data_manager.horizons:
        entry = data_manager.get_risk_geojson(disease, h)
        is_agg = str(entry.get("is_aggregate", True)).lower()
        horizons_parts.append(f'"{h}":{{"geojson":{entry["geojson"]},"maxRisk":{entry["maxRisk"]},"is_aggregate":{is_agg}}}')
    content = f'{{"disease":"{disease}","is_aggregate":true,"horizons":{{{",".join(horizons_parts)}}}}}'
    return Response(
        content=content,
        media_type='application/json'
    )

@app.get("/api/map/viewport")
def get_map_viewport(
    min_lon: float = Query(..., description="West bounding coordinate"),
    min_lat: float = Query(..., description="South bounding coordinate"),
    max_lon: float = Query(..., description="East bounding coordinate"),
    max_lat: float = Query(..., description="North bounding coordinate"),
    disease: str = Query("dengue", description="Disease name (e.g., dengue, malaria)"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon (1-4 weeks)"),
    limit: int = Query(10000, ge=100, le=25000, description="Max cells to return"),
    state: Optional[str] = Query(None, description="Filter by state name")
):
    """
    Returns exact full-resolution H3 Res-7 cells intersecting the current viewport.
    Backed by persistent SQLite R*Tree index (sub-25ms latency).
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    result = data_manager.get_viewport_cells(
        min_lon=min_lon, min_lat=min_lat,
        max_lon=max_lon, max_lat=max_lat,
        disease=disease, horizon=horizon,
        limit=limit, state=state
    )
    return Response(
        content=json.dumps(result),
        media_type='application/json',
        headers={"Cache-Control": "public, max-age=1800"}
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
    risk_info = data_manager._get_single_cell_risk(h3_id, disease, horizon)
    return {
        "disease": disease,
        "horizon": horizon,
        "h3_index": h3_id,
        "history": history,
        "risk_score": risk_info.get("risk_score"),
        "risk_percent": risk_info.get("risk_percent")
    }

@app.get("/api/cell/{h3_id}/shap")
def get_cell_shap_endpoint(
    h3_id: str,
    disease: str = Query(..., description="Disease name"),
    horizon: int = Query(1, ge=1, le=4)
):
    """
    Returns per-cell SHAP feature importance for the specified H3 cell.
    Uses local (cell-specific) SHAP values when available, falls back to
    global importance if the cell is outside the local SHAP index.
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
        
    shap_records = data_manager.get_cell_shap(h3_id, disease, horizon)
    source = "local" if any(
        data_manager.shap_local.get((d, horizon)) is not None and
        h3_id in data_manager.shap_local[(d, horizon)].index
        for d in (["dengue", "malaria"] if disease == "syndemic" else [disease])
    ) else "global"
    return {
        "h3_index": h3_id,
        "disease": disease,
        "horizon": horizon,
        "explanation": f"{'Cell-specific' if source == 'local' else 'Global'} SHAP importance",
        "top_features": shap_records
    }

@app.get("/api/search")
def search_places_and_cells(
    q: str = Query(..., min_length=1, description="Place name, district, state or H3 cell"),
    disease: str = Query("dengue", description="Disease name"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon"),
    limit: int = Query(10, ge=1, le=25)
):
    """
    Search places (districts, states) and H3 cells.
    Returns matched districts with their peak risk cells, states, and individual H3 cells.
    """
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    return data_manager.search(q, disease=disease, horizon=horizon, limit=limit)

@app.get("/api/places/popular")
def get_popular_places(
    disease: str = Query("dengue", description="Disease name"),
    horizon: int = Query(1, ge=1, le=4, description="Forecast horizon")
):
    """Returns top popular metropolitan hubs and their current peak risk hotspots."""
    if disease not in data_manager.api_diseases:
        raise HTTPException(status_code=400, detail="Invalid disease")
    popular_names = [
        "Mumbai", "Bengaluru Urban", "Delhi", "Belagavi", 
        "Pune", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad", "Ernakulam"
    ]
    results = []
    for name in popular_names:
        matches = [d for d in data_manager.districts_list if d['district'].lower() == name.lower()]
        if matches:
            d = matches[0]
            top_cell = data_manager._get_district_top_cell(d['district'], disease, horizon)
            results.append({
                "district": d['district'],
                "state": d['state'],
                "center_lat": d['center_lat'],
                "center_lon": d['center_lon'],
                "bounds": [d['min_lon'], d['min_lat'], d['max_lon'], d['max_lat']],
                "cell_count": d['cell_count'],
                "top_cell": top_cell
            })
    return {"places": results}

