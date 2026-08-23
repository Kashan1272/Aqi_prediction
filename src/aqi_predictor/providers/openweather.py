from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..aqi import concentration_aqi
from ..config import City, Settings
from .base import APITrace, HTTPClient, ProviderError

WEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"
AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"


class OpenWeatherClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HTTPClient(
            timeout_seconds=settings.api_timeout_seconds,
            attempts=settings.api_retry_attempts,
            pause_seconds=settings.request_pause_seconds,
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.openweather_api_key)

    def fetch_live(self, city: City) -> tuple[pd.DataFrame, pd.DataFrame, list[APITrace]]:
        if not self.configured:
            raise ProviderError("OPENWEATHER_API_KEY is not configured")
        common = {
            "lat": city.latitude,
            "lon": city.longitude,
            "appid": self.settings.openweather_api_key,
        }
        weather_payload, weather_trace = self.http.get_json(
            "openweather_weather", WEATHER_URL, params={**common, "units": "metric"}
        )
        air_payload, air_trace = self.http.get_json(
            "openweather_air", AIR_URL, params=common
        )

        weather_rows: list[dict[str, Any]] = []
        for item in weather_payload.get("list") or []:
            main = item.get("main") or {}
            wind = item.get("wind") or {}
            clouds = item.get("clouds") or {}
            rain = item.get("rain") or {}
            weather_rows.append({
                "timestamp": pd.to_datetime(item.get("dt"), unit="s", utc=True),
                "temperature_2m": main.get("temp"),
                "relative_humidity_2m": main.get("humidity"),
                "surface_pressure": main.get("grnd_level") or main.get("pressure"),
                "cloud_cover": clouds.get("all"),
                "wind_speed_10m": wind.get("speed"),
                "wind_direction_10m": wind.get("deg"),
                "precipitation": rain.get("3h", 0),
            })
        air_rows: list[dict[str, Any]] = []
        for item in air_payload.get("list") or []:
            components = item.get("components") or {}
            pm25 = components.get("pm2_5")
            pm10 = components.get("pm10")
            air_rows.append({
                "timestamp": pd.to_datetime(item.get("dt"), unit="s", utc=True),
                "pm2_5": pm25,
                "pm10": pm10,
                "nitrogen_dioxide": components.get("no2"),
                "ozone": components.get("o3"),
                "sulphur_dioxide": components.get("so2"),
                "carbon_monoxide": components.get("co"),
                "us_aqi": concentration_aqi(pm25, pm10),
            })
        return (
            pd.DataFrame(weather_rows).sort_values("timestamp").reset_index(drop=True)
            if weather_rows else pd.DataFrame(),
            pd.DataFrame(air_rows).sort_values("timestamp").reset_index(drop=True)
            if air_rows else pd.DataFrame(),
            [weather_trace, air_trace],
        )
