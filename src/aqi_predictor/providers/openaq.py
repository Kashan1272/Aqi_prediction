from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..aqi import concentration_aqi
from ..config import City, Settings
from .base import APITrace, HTTPClient, ProviderError

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.openaq.org/v3"


class OpenAQClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HTTPClient(
            timeout_seconds=settings.api_timeout_seconds,
            attempts=settings.api_retry_attempts,
            pause_seconds=settings.request_pause_seconds,
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.openaq_api_key)

    def _headers(self) -> dict[str, str]:
        if not self.configured:
            raise ProviderError("OPENAQ_API_KEY is not configured")
        return {"X-API-Key": self.settings.openaq_api_key}

    def nearest_location(self, city: City) -> tuple[dict[str, Any], APITrace]:
        params = {
            "coordinates": f"{city.latitude},{city.longitude}",
            "radius": self.settings.openaq_radius_meters,
            "limit": 20,
            "page": 1,
        }
        payload, trace = self.http.get_json(
            "openaq_locations",
            f"{BASE_URL}/locations",
            params=params,
            headers=self._headers(),
        )
        if trace.error:
            return {}, trace
        results = payload.get("results") or []
        if not results:
            return {}, trace
        candidates = [
            item for item in results
            if item.get("isMobile") is not True and (item.get("sensors") or [])
        ]
        return (candidates or results)[0], trace


    def historical_hourly(
        self,
        city: City,
        *,
        datetime_from: str,
        datetime_to: str,
    ) -> tuple[pd.DataFrame, list[APITrace]]:
        """Download hourly PM2.5/PM10 observations from the nearest OpenAQ location."""
        location, location_trace = self.nearest_location(city)
        traces = [location_trace]
        if not location:
            return pd.DataFrame(), traces

        sensors = location.get("sensors") or []
        if not sensors and location.get("id") is not None:
            payload, sensor_trace = self.http.get_json(
                "openaq_sensors",
                f"{BASE_URL}/locations/{location['id']}/sensors",
                headers=self._headers(),
            )
            traces.append(sensor_trace)
            sensors = payload.get("results") or []

        selected: list[tuple[int, str]] = []
        for sensor in sensors:
            parameter = sensor.get("parameter") or {}
            name = str(parameter.get("name") or sensor.get("name") or "").lower()
            sensor_id = sensor.get("id")
            if sensor_id is None:
                continue
            normalized = {
                "pm25": "observed_pm2_5",
                "pm2.5": "observed_pm2_5",
                "pm10": "observed_pm10",
            }.get(name)
            if normalized:
                selected.append((int(sensor_id), normalized))
        if not selected:
            return pd.DataFrame(), traces

        series_frames: list[pd.DataFrame] = []
        for sensor_id, output_column in selected:
            page = 1
            rows: list[dict[str, Any]] = []
            while True:
                payload, trace = self.http.get_json(
                    "openaq_sensor_hours",
                    f"{BASE_URL}/sensors/{sensor_id}/hours",
                    params={
                        "datetime_from": datetime_from,
                        "datetime_to": datetime_to,
                        "limit": 1000,
                        "page": page,
                    },
                    headers=self._headers(),
                )
                traces.append(trace)
                if trace.error:
                    break
                results = payload.get("results") or []
                if not results:
                    break
                for item in results:
                    period = item.get("period") or {}
                    timestamp = (
                        (period.get("datetimeFrom") or {}).get("utc")
                        or (period.get("datetimeTo") or {}).get("utc")
                    )
                    value = item.get("value")
                    if timestamp is None or value is None:
                        continue
                    rows.append({"timestamp": timestamp, output_column: value})
                meta = payload.get("meta") or {}
                found = meta.get("found")
                try:
                    found_count = int(str(found).lstrip(">"))
                except (TypeError, ValueError):
                    found_count = None
                if len(results) < 1000 or (found_count is not None and page * 1000 >= found_count):
                    break
                page += 1
            if rows:
                sensor_frame = pd.DataFrame(rows)
                sensor_frame["timestamp"] = pd.to_datetime(
                    sensor_frame["timestamp"], utc=True, errors="coerce"
                )
                sensor_frame[output_column] = pd.to_numeric(
                    sensor_frame[output_column], errors="coerce"
                )
                sensor_frame = (
                    sensor_frame.dropna(subset=["timestamp"])
                    .groupby("timestamp", observed=True)[output_column]
                    .mean()
                    .reset_index()
                )
                series_frames.append(sensor_frame)

        if not series_frames:
            return pd.DataFrame(), traces
        frame = series_frames[0]
        for item in series_frames[1:]:
            frame = frame.merge(item, on="timestamp", how="outer")
        pm25 = frame.get("observed_pm2_5", pd.Series(np.nan, index=frame.index))
        pm10 = frame.get("observed_pm10", pd.Series(np.nan, index=frame.index))
        frame["observed_us_aqi"] = [
            concentration_aqi(a, b) for a, b in zip(pm25, pm10, strict=False)
        ]
        frame["observed_source"] = "openaq"
        frame["observed_location_id"] = location.get("id")
        return frame.sort_values("timestamp").reset_index(drop=True), traces

    def latest(self, city: City) -> tuple[dict[str, Any], list[APITrace]]:
        location, location_trace = self.nearest_location(city)
        traces = [location_trace]
        if not location:
            return {}, traces
        location_id = location.get("id")
        payload, latest_trace = self.http.get_json(
            "openaq_latest",
            f"{BASE_URL}/locations/{location_id}/latest",
            headers=self._headers(),
        )
        traces.append(latest_trace)
        if latest_trace.error:
            return {}, traces
        measurements: dict[str, float] = {}
        for item in payload.get("results") or []:
            parameter = item.get("parameter") or {}
            name = str(parameter.get("name") or "").lower()
            value = item.get("value")
            if value is None:
                continue
            try:
                measurements[name] = float(value)
            except (TypeError, ValueError):
                continue
        pm25 = measurements.get("pm25") or measurements.get("pm2.5")
        pm10 = measurements.get("pm10")
        return {
            "provider": "openaq",
            "location_id": location_id,
            "location_name": location.get("name"),
            "distance_m": location.get("distance"),
            "measurements": measurements,
            "computed_us_aqi": concentration_aqi(pm25, pm10),
        }, traces
