from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import Settings
from .providers import AQICNClient, OpenAQClient, OpenMeteoClient, OpenWeatherClient
from .storage import LocalStore


def _frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "columns": [], "non_null_ratio": 0.0, "flat_numeric_columns": []}
    numeric = frame.select_dtypes(include=[np.number])
    flat = [
        column for column in numeric
        if numeric[column].dropna().nunique() <= 1
    ]
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "non_null_ratio": float(frame.notna().mean().mean()),
        "flat_numeric_columns": flat,
        "first_timestamp": (
            pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").min().isoformat()
            if "timestamp" in frame else None
        ),
        "last_timestamp": (
            pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").max().isoformat()
            if "timestamp" in frame else None
        ),
    }


def probe_city(settings: Settings, city_key: str, *, days: int = 7) -> dict[str, Any]:
    city = settings.city(city_key)
    end = datetime.now(UTC).date() - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    sources: dict[str, Any] = {}

    open_meteo = OpenMeteoClient(settings)
    for name, callable_ in (
        ("open_meteo_historical_weather", lambda: open_meteo.fetch_historical_weather(city, start, end)),
        ("open_meteo_historical_air", lambda: open_meteo.fetch_historical_air_quality(city, start, end)),
        ("open_meteo_previous_runs", lambda: open_meteo.fetch_lead_weather(city, start, end)),
    ):
        try:
            frame, traces = callable_()
            sources[name] = {
                "configured": True,
                "success": not frame.empty,
                "stats": _frame_stats(frame),
                "traces": [trace.to_dict() for trace in traces],
            }
        except Exception as exc:
            sources[name] = {"configured": True, "success": False, "error": str(exc)}
    try:
        weather, air, traces = open_meteo.fetch_live(city)
        sources["open_meteo_live"] = {
            "configured": True,
            "success": not weather.empty and not air.empty,
            "weather": _frame_stats(weather),
            "air": _frame_stats(air),
            "traces": [trace.to_dict() for trace in traces],
        }
    except Exception as exc:
        sources["open_meteo_live"] = {"configured": True, "success": False, "error": str(exc)}

    if settings.openaq_api_key:
        try:
            current, traces = OpenAQClient(settings).latest(city)
            sources["openaq"] = {
                "configured": True,
                "success": bool(current),
                "current": current,
                "traces": [trace.to_dict() for trace in traces],
            }
        except Exception as exc:
            sources["openaq"] = {"configured": True, "success": False, "error": str(exc)}
    else:
        sources["openaq"] = {"configured": False, "success": False}

    if settings.openweather_api_key:
        try:
            weather, air, traces = OpenWeatherClient(settings).fetch_live(city)
            sources["openweather"] = {
                "configured": True,
                "success": not weather.empty and not air.empty,
                "weather": _frame_stats(weather),
                "air": _frame_stats(air),
                "traces": [trace.to_dict() for trace in traces],
            }
        except Exception as exc:
            sources["openweather"] = {"configured": True, "success": False, "error": str(exc)}
    else:
        sources["openweather"] = {"configured": False, "success": False}

    if settings.aqicn_api_token:
        try:
            current, forecast, traces = AQICNClient(settings).fetch_live(city)
            sources["aqicn"] = {
                "configured": True,
                "success": bool(current),
                "current": current,
                "forecast": _frame_stats(forecast),
                "traces": [trace.to_dict() for trace in traces],
            }
        except Exception as exc:
            sources["aqicn"] = {"configured": True, "success": False, "error": str(exc)}
    else:
        sources["aqicn"] = {"configured": False, "success": False}

    required = [
        "open_meteo_historical_weather",
        "open_meteo_historical_air",
        "open_meteo_previous_runs",
        "open_meteo_live",
    ]
    configured_optional = [
        name for name in ("openaq", "openweather", "aqicn")
        if sources[name]["configured"]
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "city": city.key,
        "required_sources_ok": all(sources[name]["success"] for name in required),
        "configured_optional_sources_ok": all(
            sources[name]["success"] for name in configured_optional
        ) if configured_optional else True,
        "sources": sources,
    }
    # Normalize NumPy/Pandas missing and scalar values before both saving and
    # returning the report. This keeps the file and terminal output valid JSON.
    payload = LocalStore._json_safe(payload)
    LocalStore(settings).save_report(f"provider_probe_{city.key}.json", payload)
    return payload
