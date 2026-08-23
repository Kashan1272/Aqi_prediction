from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .aqi import category, health_message
from .config import City, Settings
from .features import add_provider_snapshot_features, latest_live_feature_rows
from .providers import AQICNClient, OpenAQClient, OpenMeteoClient, OpenWeatherClient
from .providers.base import ProviderError
from .registry import LocalModelRegistry
from .storage import LocalStore

LOGGER = logging.getLogger(__name__)


def _daily_air(frame: pd.DataFrame, city: City, provider: str) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame or "us_aqi" not in frame:
        return pd.DataFrame()
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["date"] = data["timestamp"].dt.tz_convert(city.timezone).dt.date
    result = data.groupby("date", observed=True)["us_aqi"].agg(["mean", "max"]).reset_index()
    result["date"] = pd.to_datetime(result["date"])
    result["provider"] = provider
    return result.rename(columns={"mean": "aqi_mean", "max": "aqi_max"})


def _weighted_average(values: list[tuple[float, float]]) -> float:
    valid = [(float(value), float(weight)) for value, weight in values if np.isfinite(value) and weight > 0]
    if not valid:
        return np.nan
    total = sum(weight for _, weight in valid)
    return float(sum(value * weight for value, weight in valid) / total)


def _aqicn_daily(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame()
    pm = forecast[forecast["pollutant"].isin(["pm25", "pm10"])].copy()
    if pm.empty:
        return pd.DataFrame()
    daily = pm.groupby("date", observed=True).agg(
        aqi_mean=("avg", "max"),
        aqi_max=("max", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["provider"] = "aqicn"
    return daily.dropna(subset=["date"])


def _provider_rows(daily: pd.DataFrame, provider: str) -> dict[pd.Timestamp, dict[str, float]]:
    subset = daily[daily["provider"] == provider] if not daily.empty else pd.DataFrame()
    return {
        pd.Timestamp(row["date"]).normalize(): {
            "mean": float(row["aqi_mean"]),
            "max": float(row["aqi_max"]),
        }
        for _, row in subset.iterrows()
    }


def _calibrated_hourly_curve(
    open_meteo_air: pd.DataFrame,
    daily_forecast: list[dict[str, Any]],
    city: City,
) -> list[dict[str, Any]]:
    if open_meteo_air.empty:
        rows: list[dict[str, Any]] = []
        for day in daily_forecast:
            date = pd.Timestamp(day["date"])
            for hour in range(24):
                wave = 0.12 * np.sin(2 * np.pi * (hour - 7) / 24)
                value = np.clip(day["aqi_mean"] * (1 + wave), 0, day["aqi_max"])
                rows.append({
                    "timestamp": (date + pd.Timedelta(hours=hour)).isoformat(),
                    "aqi": round(float(value), 1),
                })
        return rows

    air = open_meteo_air.copy()
    air["timestamp"] = pd.to_datetime(air["timestamp"], utc=True, errors="coerce")
    air = air.dropna(subset=["timestamp", "us_aqi"])
    air["local_date"] = air["timestamp"].dt.tz_convert(city.timezone).dt.normalize().dt.tz_localize(None)
    forecast_map = {pd.Timestamp(item["date"]).normalize(): item for item in daily_forecast}
    rows: list[dict[str, Any]] = []
    for date, group in air.groupby("local_date", observed=True):
        target = forecast_map.get(pd.Timestamp(date).normalize())
        if not target:
            continue
        values = pd.to_numeric(group["us_aqi"], errors="coerce").to_numpy(dtype=float)
        provider_mean = np.nanmean(values)
        provider_max = np.nanmax(values)
        if not np.isfinite(provider_mean) or provider_mean <= 0:
            continue
        mean_scale = target["aqi_mean"] / provider_mean
        scaled = values * mean_scale
        if np.nanmax(scaled) > 0:
            peak_scale = target["aqi_max"] / np.nanmax(scaled)
            scaled = 0.72 * scaled + 0.28 * scaled * peak_scale
        for timestamp, value in zip(group["timestamp"], scaled, strict=False):
            rows.append({
                "timestamp": timestamp.isoformat(),
                "aqi": round(float(np.clip(value, 0, 500)), 1),
            })
    return rows


def forecast_city(settings: Settings, city_key: str) -> dict[str, Any]:
    city = settings.city(city_key)
    store = LocalStore(settings)
    historical = store.read_city(city.key)
    if historical.empty:
        raise FileNotFoundError(
            f"No historical data for {city.key}. Run scripts/backfill.py first."
        )

    traces: list[dict[str, Any]] = []
    provider_errors: dict[str, str] = {}
    open_meteo = OpenMeteoClient(settings)
    weather, open_meteo_air, om_traces = open_meteo.fetch_live(
        city, forecast_days=settings.forecast_days
    )
    traces.extend(trace.to_dict() for trace in om_traces)

    # Refresh history with observations that are already in the past. Never
    # write future provider forecasts into the historical training partition.
    recent = weather.merge(open_meteo_air, on="timestamp", how="outer", suffixes=("", "_air"))
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True, errors="coerce")
    completed_hour_cutoff = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    recent = recent[recent["timestamp"] <= completed_hour_cutoff].copy()
    if not recent.empty:
        recent["city"] = city.key
        recent["province"] = city.province
        recent["latitude"] = city.latitude
        recent["longitude"] = city.longitude
        recent["elevation_m"] = city.elevation_m
        recent["data_source"] = "open_meteo_live_completed_hour"
        recent["ingested_at"] = datetime.now(UTC)
        store.write_city(city.key, recent, merge=True)
        historical = store.read_city(city.key)

    feature_rows = latest_live_feature_rows(
        historical,
        weather,
        city,
        forecast_days=settings.forecast_days,
    )
    issue_date = pd.Timestamp(feature_rows["issue_date"].iloc[0])

    provider_daily_frames = [_daily_air(open_meteo_air, city, "open_meteo")]
    openweather_air = pd.DataFrame()
    if settings.openweather_api_key:
        try:
            _, openweather_air, ow_traces = OpenWeatherClient(settings).fetch_live(city)
            traces.extend(trace.to_dict() for trace in ow_traces)
            provider_daily_frames.append(_daily_air(openweather_air, city, "openweather"))
        except Exception as exc:
            provider_errors["openweather"] = str(exc)
    aqicn_current: dict[str, Any] = {}
    if settings.aqicn_api_token:
        try:
            aqicn_current, aqicn_forecast, aqicn_traces = AQICNClient(settings).fetch_live(city)
            traces.extend(trace.to_dict() for trace in aqicn_traces)
            provider_daily_frames.append(_aqicn_daily(aqicn_forecast))
        except Exception as exc:
            provider_errors["aqicn"] = str(exc)
    openaq_current: dict[str, Any] = {}
    if settings.openaq_api_key:
        try:
            openaq_current, openaq_traces = OpenAQClient(settings).latest(city)
            traces.extend(trace.to_dict() for trace in openaq_traces)
        except Exception as exc:
            provider_errors["openaq"] = str(exc)

    provider_daily = pd.concat(
        [frame for frame in provider_daily_frames if frame is not None and not frame.empty],
        ignore_index=True,
        copy=False,
    ) if any(not frame.empty for frame in provider_daily_frames) else pd.DataFrame()

    maps = {
        "open_meteo": _provider_rows(provider_daily, "open_meteo"),
        "openweather": _provider_rows(provider_daily, "openweather"),
        "aqicn": _provider_rows(provider_daily, "aqicn"),
    }
    open_meteo_current = np.nan
    if not open_meteo_air.empty and "timestamp" in open_meteo_air:
        current_frame = open_meteo_air.copy()
        current_frame["timestamp"] = pd.to_datetime(
            current_frame["timestamp"], utc=True, errors="coerce"
        )
        current_frame["us_aqi"] = pd.to_numeric(
            current_frame.get("us_aqi"), errors="coerce"
        )
        current_frame = current_frame.dropna(subset=["timestamp", "us_aqi"])
        if not current_frame.empty:
            now = pd.Timestamp.now(tz="UTC")
            nearest_index = (current_frame["timestamp"] - now).abs().idxmin()
            open_meteo_current = float(current_frame.loc[nearest_index, "us_aqi"])
    observed = float(openaq_current.get("computed_us_aqi", np.nan))
    observation_bias = observed - open_meteo_current if np.isfinite(observed) and np.isfinite(open_meteo_current) else 0.0
    bias_decay = {1: 0.75, 2: 0.45, 3: 0.25}

    snapshot_rows: list[dict[str, Any]] = []
    collected_at = datetime.now(UTC)
    for day in range(1, settings.forecast_days + 1):
        target_date = (issue_date + pd.Timedelta(days=day)).normalize()
        snapshot_row: dict[str, Any] = {
            "city": city.key,
            "issue_date": issue_date.normalize(),
            "target_date": target_date,
            "horizon_day": day,
            "collected_at": collected_at,
            "provider_openaq_current_aqi": (
                observed if np.isfinite(observed) else np.nan
            ),
            "provider_open_meteo_current_aqi": (
                open_meteo_current if np.isfinite(open_meteo_current) else np.nan
            ),
        }
        provider_count = 0
        for provider in ("open_meteo", "openweather", "aqicn"):
            values = maps[provider].get(target_date)
            if values:
                snapshot_row[f"provider_{provider}_aqi_mean"] = values["mean"]
                snapshot_row[f"provider_{provider}_aqi_max"] = values["max"]
                provider_count += 1
        snapshot_row["provider_forecast_count"] = provider_count
        snapshot_rows.append(snapshot_row)
    if snapshot_rows:
        snapshot_frame = pd.DataFrame(snapshot_rows)
        store.write_provider_snapshots(city.key, snapshot_frame)
        live_snapshot = snapshot_frame.drop(
            columns=["target_date", "collected_at"], errors="ignore"
        )
        feature_rows = feature_rows.merge(
            live_snapshot,
            on=["city", "issue_date", "horizon_day"],
            how="left",
            validate="one_to_one",
        )
    feature_rows = add_provider_snapshot_features(feature_rows)

    model, model_report, model_dir = LocalModelRegistry(settings).load_production()
    # Preserve backward and forward compatibility with stored champions. Any
    # optional issue-time feature absent from the live provider response is
    # supplied as NaN and handled by the model's fitted imputer.
    for column in getattr(model, "numeric_features", []):
        if column not in feature_rows:
            feature_rows[column] = np.nan
    for column in getattr(model, "categorical_features", []):
        if column not in feature_rows:
            feature_rows[column] = "unknown"
    model_daily = model.predict(feature_rows)
    model_daily["date"] = [
        issue_date + pd.Timedelta(days=int(day))
        for day in model_daily["horizon_day"]
    ]

    daily_output: list[dict[str, Any]] = []
    for _, row in model_daily.iterrows():
        day = int(row["horizon_day"])
        date = pd.Timestamp(row["date"]).normalize()
        provider_values_mean = [
            (float(row["aqi_mean"]), settings.provider_weight_model),
        ]
        provider_values_max = [
            (float(row["aqi_max"]), settings.provider_weight_model),
        ]
        contributions: dict[str, Any] = {
            "model": {
                "mean": round(float(row["aqi_mean"]), 2),
                "max": round(float(row["aqi_max"]), 2),
            }
        }
        for provider, weight in (
            ("open_meteo", settings.provider_weight_open_meteo),
            ("openweather", settings.provider_weight_openweather),
            ("aqicn", settings.provider_weight_aqicn),
        ):
            values = maps[provider].get(date)
            if values:
                provider_values_mean.append((values["mean"], weight))
                provider_values_max.append((values["max"], weight))
                contributions[provider] = values
        mean_value = _weighted_average(provider_values_mean) + observation_bias * bias_decay.get(day, 0)
        max_value = _weighted_average(provider_values_max) + observation_bias * bias_decay.get(day, 0)
        max_value = max(mean_value, max_value)
        daily_output.append({
            "horizon_day": day,
            "date": date.date().isoformat(),
            "aqi_mean": round(float(np.clip(mean_value, 0, 500)), 1),
            "aqi_max": round(float(np.clip(max_value, 0, 500)), 1),
            "aqi_mean_lower": round(float(row["aqi_mean_lower"]), 1),
            "aqi_mean_upper": round(float(row["aqi_mean_upper"]), 1),
            "category": category(mean_value),
            "health_message": health_message(mean_value),
            "contributions": contributions,
        })

    hourly = _calibrated_hourly_curve(open_meteo_air, daily_output, city)
    payload = {
        "project_version": "6.9.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "city": {
            "key": city.key,
            "name": city.name,
            "province": city.province,
            "latitude": city.latitude,
            "longitude": city.longitude,
        },
        "model": {
            "name": settings.model_name,
            "version_path": str(model_dir),
            "promoted": bool(model_report.get("promotion", {}).get("promoted")),
            "test_metrics": model_report.get("test_metrics", {}),
            "selected_algorithms": model_report.get("selected_algorithms", {}),
        },
        "current_observations": {
            "open_meteo_aqi": open_meteo_current if np.isfinite(open_meteo_current) else None,
            "openaq": openaq_current,
            "aqicn": aqicn_current,
            "sensor_bias_applied": round(float(observation_bias), 2),
        },
        "daily_forecast": daily_output,
        "hourly_forecast": hourly,
        "provider_health": {
            "errors": provider_errors,
            "traces": traces,
        },
    }
    store.save_prediction(city.key, payload)
    return payload
