from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import City, Settings
from .base import APITrace, HTTPClient, ProviderError

BASE_URL = "https://api.waqi.info/feed"


class AQICNClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HTTPClient(
            timeout_seconds=settings.api_timeout_seconds,
            attempts=settings.api_retry_attempts,
            pause_seconds=settings.request_pause_seconds,
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.aqicn_api_token)

    def fetch_live(self, city: City) -> tuple[dict[str, Any], pd.DataFrame, list[APITrace]]:
        if not self.configured:
            raise ProviderError("AQICN_API_TOKEN is not configured")
        url = f"{BASE_URL}/geo:{city.latitude};{city.longitude}/"
        payload, trace = self.http.get_json(
            "aqicn",
            url,
            params={"token": self.settings.aqicn_api_token},
        )
        if trace.error:
            return {}, pd.DataFrame(), [trace]
        if payload.get("status") != "ok":
            raise ProviderError(f"AQICN returned status={payload.get('status')}: {payload.get('data')}")
        data = payload.get("data") or {}
        current = {
            "provider": "aqicn",
            "aqi": _number(data.get("aqi")),
            "station": (data.get("city") or {}).get("name"),
            "timestamp": (data.get("time") or {}).get("iso"),
            "dominant_pollutant": data.get("dominentpol"),
        }
        rows: list[dict[str, Any]] = []
        forecast = (data.get("forecast") or {}).get("daily") or {}
        for pollutant, entries in forecast.items():
            for item in entries or []:
                rows.append({
                    "date": item.get("day"),
                    "pollutant": pollutant,
                    "avg": _number(item.get("avg")),
                    "min": _number(item.get("min")),
                    "max": _number(item.get("max")),
                })
        return current, pd.DataFrame(rows), [trace]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
