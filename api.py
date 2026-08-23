from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AQI_PROJECT_ROOT", str(ROOT))

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aqi_predictor.config import get_settings
from aqi_predictor.city_selection import load_selected_city_keys
from aqi_predictor.inference import forecast_city
from aqi_predictor.registry import LocalModelRegistry
from aqi_predictor.storage import LocalStore

settings = get_settings()
store = LocalStore(settings)
app = FastAPI(
    title="Pearls AQI Predictor API",
    version="6.7.0",
    description="Three-day daily AQI forecasts with a calibrated 72-hour dashboard curve.",
)
origins = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RefreshResponse(BaseModel):
    city: str
    generated_at: str


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/v1/health")
def health() -> dict:
    model_available = True
    try:
        LocalModelRegistry(settings).production_dir()
    except FileNotFoundError:
        model_available = False
    return {
        "status": "ok",
        "version": "6.7.0",
        "configured_cities": len(settings.cities),
        "active_training_cities": len(load_selected_city_keys(settings)),
        "forecast_days": settings.forecast_days,
        "model_available": model_available,
        "feature_store_backend": settings.feature_store_backend,
        "providers": {
            "open_meteo": True,
            "openaq": bool(settings.openaq_api_key),
            "openweather": bool(settings.openweather_api_key),
            "aqicn": bool(settings.aqicn_api_token),
            "hopsworks": bool(settings.hopsworks_project and settings.hopsworks_api_key),
        },
    }


@app.get("/api/v1/cities")
def cities() -> list[dict]:
    return [
        {
            "key": settings.city(key).key,
            "name": settings.city(key).name,
            "province": settings.city(key).province,
            "latitude": settings.city(key).latitude,
            "longitude": settings.city(key).longitude,
        }
        for key in load_selected_city_keys(settings)
    ]


@app.get("/api/v1/forecast/{city_key}")
def forecast(city_key: str, refresh: bool = Query(False)) -> dict:
    try:
        settings.city(city_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = store.load_prediction(city_key)
    if refresh or not payload:
        try:
            payload = forecast_city(settings, city_key)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return payload


@app.post("/api/v1/refresh/{city_key}", response_model=RefreshResponse)
def refresh(city_key: str, x_refresh_token: str | None = Header(None)) -> dict:
    configured_token = os.getenv("API_REFRESH_TOKEN", "").strip()
    if configured_token and x_refresh_token != configured_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    try:
        payload = forecast_city(settings, city_key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"city": city_key, "generated_at": payload["generated_at"]}


@app.get("/api/v1/model")
def model_status() -> dict:
    try:
        _, report, version_dir = LocalModelRegistry(settings).load_production()
    except FileNotFoundError as exc:
        return {"available": False, "detail": str(exc)}
    return {
        "available": True,
        "name": settings.model_name,
        "version_path": str(version_dir),
        "promotion": report.get("promotion"),
        "quality_gate": report.get("quality_gate"),
        "test_metrics": report.get("test_metrics"),
        "selected_algorithms": report.get("selected_algorithms"),
        "target_contract": report.get("target_contract"),
    }


@app.get("/api/v1/national")
def national() -> list[dict]:
    rows = []
    for key in load_selected_city_keys(settings):
        city = settings.city(key)
        payload = store.load_prediction(city.key)
        daily = payload.get("daily_forecast") or []
        if not daily:
            continue
        rows.append({
            "city": city.key,
            "name": city.name,
            "province": city.province,
            "latitude": city.latitude,
            "longitude": city.longitude,
            "day1_mean": daily[0].get("aqi_mean"),
            "day1_max": daily[0].get("aqi_max"),
            "category": daily[0].get("category"),
            "generated_at": payload.get("generated_at"),
        })
    return sorted(rows, key=lambda item: item.get("day1_mean") or -1, reverse=True)
