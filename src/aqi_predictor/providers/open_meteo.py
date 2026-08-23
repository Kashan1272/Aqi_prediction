from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..aqi import add_computed_aqi
from ..config import City, Settings
from .base import APITrace, HTTPClient, ProviderError

LOGGER = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Keep historical requests below ten variables where possible. Open-Meteo
# counts requests with >10 variables or >2 weeks as multiple API calls.
HISTORICAL_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]
LIVE_WEATHER_VARIABLES = [
    *HISTORICAL_WEATHER_VARIABLES,
    "apparent_temperature",
    "rain",
]
AIR_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]
# Seven variables × three lead days gives 21 series. This retains the strongest
# dispersion/washout/weather signals while keeping the free-API call cost below
# the previous 27-series request contract.
LEAD_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
]


def _hourly_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    frame = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True, errors="coerce")})
    for key, values in hourly.items():
        if key == "time":
            continue
        if isinstance(values, list) and len(values) == len(frame):
            frame[key] = pd.to_numeric(pd.Series(values), errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _estimated_call_units(start: date, end: date, variable_count: int) -> int:
    """Estimate Open-Meteo weighted API-call units conservatively.

    Open-Meteo notes that long time ranges and requests with more than ten
    variables can count as multiple calls. Backfill chunks are capped at 14
    days, but the variable multiplier still matters for previous-runs data.
    """
    days = max(1, (end - start).days + 1)
    time_units = max(1, math.ceil(days / 14))
    variable_units = max(1, math.ceil(max(1, variable_count) / 10))
    return time_units * variable_units


class OpenMeteoClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HTTPClient(
            timeout_seconds=settings.api_timeout_seconds,
            attempts=settings.api_retry_attempts,
            pause_seconds=settings.request_pause_seconds,
            quota_state_path=getattr(
                settings,
                "api_quota_state_path",
                settings.reports_dir / "open_meteo_quota_state.json",
            ),
            quota_minute_limit=getattr(settings, "api_quota_max_per_minute", 180),
            quota_hour_limit=getattr(settings, "api_quota_max_per_hour", 2800),
            quota_day_limit=getattr(settings, "api_quota_max_per_day", 7500),
        )

    @staticmethod
    def _base_params(city: City) -> dict[str, Any]:
        return {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "timezone": "UTC",
            "timeformat": "iso8601",
        }

    def _cache_path(
        self,
        kind: str,
        city: City,
        start: date,
        end: date,
        params: dict[str, Any],
    ) -> Path:
        contract = "|".join(
            f"{key}={params[key]}" for key in sorted(params) if key not in {"start_date", "end_date"}
        )
        signature = hashlib.sha1(contract.encode("utf-8")).hexdigest()[:12]
        return (
            self.settings.api_cache_dir
            / kind
            / city.key
            / f"{start.isoformat()}_{end.isoformat()}_{signature}.csv.gz"
        )

    @staticmethod
    def _atomic_cache(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            frame.to_csv(tmp_path, index=False, compression="gzip")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _fetch_chunk(
        self,
        *,
        kind: str,
        provider: str,
        url: str,
        city: City,
        chunk_start: date,
        chunk_end: date,
        params: dict[str, Any],
        variable_count: int,
        rename: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, APITrace]:
        cache_path = self._cache_path(kind, city, chunk_start, chunk_end, params)
        if self.settings.backfill_cache_enabled and cache_path.exists():
            frame = pd.read_csv(cache_path, compression="gzip", low_memory=False)
            if "timestamp" in frame:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            return frame, APITrace(
                provider=provider,
                endpoint=f"cache://{cache_path}",
                status_code=200,
                elapsed_ms=0.0,
                attempt=0,
                response_bytes=cache_path.stat().st_size,
                from_cache=True,
            )

        payload, trace = self.http.get_json(
            provider,
            url,
            params=params,
            cost_units=_estimated_call_units(chunk_start, chunk_end, variable_count),
        )
        if trace.error:
            raise ProviderError(trace.error)
        frame = _hourly_frame(payload)
        if rename:
            frame = frame.rename(columns=rename)
        if self.settings.backfill_cache_enabled and not frame.empty:
            self._atomic_cache(cache_path, frame)
        return frame, trace

    def fetch_historical_weather(
        self,
        city: City,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, list[APITrace]]:
        frames: list[pd.DataFrame] = []
        traces: list[APITrace] = []
        for index, (chunk_start, chunk_end) in enumerate(
            _chunks(start, end, self.settings.chunk_days), start=1
        ):
            LOGGER.info("Open-Meteo weather chunk %s for %s: %s to %s", index, city.key, chunk_start, chunk_end)
            params = {
                **self._base_params(city),
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(HISTORICAL_WEATHER_VARIABLES),
            }
            frame, trace = self._fetch_chunk(
                kind="weather_history",
                provider="open_meteo_weather_history",
                url=ARCHIVE_URL,
                city=city,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                params=params,
                variable_count=len(HISTORICAL_WEATHER_VARIABLES),
            )
            traces.append(trace)
            if not frame.empty:
                frames.append(frame)
        return self._combine(frames), traces

    def fetch_historical_air_quality(
        self,
        city: City,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, list[APITrace]]:
        frames: list[pd.DataFrame] = []
        traces: list[APITrace] = []
        for index, (chunk_start, chunk_end) in enumerate(
            _chunks(start, end, self.settings.chunk_days), start=1
        ):
            LOGGER.info("Open-Meteo AQ chunk %s for %s: %s to %s", index, city.key, chunk_start, chunk_end)
            params = {
                **self._base_params(city),
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(AIR_VARIABLES),
                "domains": "cams_global",
            }
            frame, trace = self._fetch_chunk(
                kind="air_history",
                provider="open_meteo_air_history",
                url=AIR_QUALITY_URL,
                city=city,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                params=params,
                variable_count=len(AIR_VARIABLES),
            )
            traces.append(trace)
            if not frame.empty:
                frames.append(frame)
        return add_computed_aqi(self._combine(frames)), traces

    def fetch_lead_weather(
        self,
        city: City,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, list[APITrace]]:
        requested = [
            f"{variable}_previous_day{lead}"
            for lead in (1, 2, 3)
            for variable in LEAD_WEATHER_VARIABLES
        ]
        rename = {
            f"{variable}_previous_day{lead}": f"lead_{lead}d_{variable}"
            for lead in (1, 2, 3)
            for variable in LEAD_WEATHER_VARIABLES
        }
        frames: list[pd.DataFrame] = []
        traces: list[APITrace] = []
        for index, (chunk_start, chunk_end) in enumerate(
            _chunks(start, end, self.settings.chunk_days), start=1
        ):
            LOGGER.info("Previous-runs weather chunk %s for %s: %s to %s", index, city.key, chunk_start, chunk_end)
            params = {
                **self._base_params(city),
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(requested),
            }
            frame, trace = self._fetch_chunk(
                kind="previous_runs",
                provider="open_meteo_previous_runs",
                url=PREVIOUS_RUNS_URL,
                city=city,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                params=params,
                variable_count=len(requested),
                rename=rename,
            )
            traces.append(trace)
            if not frame.empty:
                frames.append(frame)
        return self._combine(frames), traces

    def fetch_live(
        self,
        city: City,
        *,
        forecast_days: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[APITrace]]:
        weather_params = {
            **self._base_params(city),
            "hourly": ",".join(LIVE_WEATHER_VARIABLES),
            "forecast_days": max(3, forecast_days),
            "past_days": 2,
        }
        air_params = {
            **self._base_params(city),
            "hourly": ",".join(AIR_VARIABLES),
            "forecast_days": max(3, forecast_days),
            "past_days": 2,
            "domains": "cams_global",
        }
        live_days = max(3, forecast_days) + 2
        live_start = date.today() - timedelta(days=2)
        live_end = live_start + timedelta(days=live_days - 1)
        weather_payload, weather_trace = self.http.get_json(
            "open_meteo_live_weather",
            FORECAST_URL,
            params=weather_params,
            cost_units=_estimated_call_units(live_start, live_end, len(LIVE_WEATHER_VARIABLES)),
        )
        air_payload, air_trace = self.http.get_json(
            "open_meteo_live_air",
            AIR_QUALITY_URL,
            params=air_params,
            cost_units=_estimated_call_units(live_start, live_end, len(AIR_VARIABLES)),
        )
        weather = _hourly_frame(weather_payload)
        air = add_computed_aqi(_hourly_frame(air_payload))
        return weather, air, [weather_trace, air_trace]

    def clear_city_cache(self, city: City) -> None:
        if self.settings.backfill_keep_cache:
            return
        for kind in ("weather_history", "air_history", "previous_runs"):
            directory = self.settings.api_cache_dir / kind / city.key
            if not directory.exists():
                continue
            for path in directory.glob("*.csv.gz"):
                path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass

    @staticmethod
    def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [frame for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return pd.DataFrame()
        all_columns = sorted(set().union(*(frame.columns for frame in usable)))
        aligned = [frame.reindex(columns=all_columns) for frame in usable]
        result = pd.concat(aligned, ignore_index=True, copy=False)
        return result.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
