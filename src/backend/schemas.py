from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    version: str

class MetadataResponse(BaseModel):
    latest_year: int
    latest_week: int
    diseases: List[str]
    models: List[str]

class AnalyticsResponse(BaseModel):
    total_cells: int
    total_districts: int
    latest_predictions: int

class HotspotCell(BaseModel):
    h3_index: str
    district: str
    state: str
    risk_score: float

class HotspotsResponse(BaseModel):
    disease: str
    horizon: int
    top_cells: List[HotspotCell]

class ForecastPoint(BaseModel):
    year: int
    week: int
    actual: float
    pred_lgbm: float
    pred_xgb: float
    pred_naive: float

class ForecastResponse(BaseModel):
    disease: str
    h3_index: Optional[str] = None
    district: Optional[str] = None
    horizon: int
    history: List[ForecastPoint]

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]
