from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from .config import City, Settings
from .providers import OpenMeteoClient
from .providers.base import RateLimitError
from .storage import LocalStore

LOGGER = logging.getLogger(__name__)


def _coverage_ok(
    store: LocalStore,
    city: City,
    start: date,
    end: date,
    *,
    with_lead_weather: bool,
) -> bool:
    frame = store.read_city(
        city.key,
        columns=[
            "timestamp",
            "temperature_2m",
            "us_aqi",
            "lead_1d_temperature_2m",
            "lead_2d_temperature_2m",
            "lead_3d_temperature_2m",
        ],
    )
    if frame.empty or "timestamp" not in frame:
        return False
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frame = frame[(frame["timestamp"] >= lower) & (frame["timestamp"] < upper)]
    expected_hours = ((end - start).days + 1) * 24
    if frame["timestamp"].nunique() < int(expected_hours * 0.95):
        return False
    for column in ("temperature_2m", "us_aqi"):
        if column not in frame or frame[column].notna().mean() < 0.80:
            return False
    if with_lead_weather:
        for column in (
            "lead_1d_temperature_2m",
            "lead_2d_temperature_2m",
            "lead_3d_temperature_2m",
        ):
            if column not in frame or frame[column].notna().mean() < 0.60:
                return False
    return True


def backfill_city(
    settings: Settings,
    city: City,
    *,
    days: int | None = None,
    with_lead_weather: bool = True,
    client: OpenMeteoClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    end = datetime.now(UTC).date() - timedelta(days=2)
    start = end - timedelta(days=(days or settings.backfill_days) - 1)
    store = LocalStore(settings)

    if settings.backfill_skip_complete and not force and _coverage_ok(
        store,
        city,
        start,
        end,
        with_lead_weather=with_lead_weather,
    ):
        existing = store.read_city(city.key, columns=["timestamp"])
        LOGGER.info("Skipping %s; requested coverage is already complete", city.key)
        return {
            "city": city.key,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "rows_received": 0,
            "rows_stored": len(existing),
            "path": str(store.city_path(city.key)),
            "with_lead_weather": with_lead_weather,
            "skipped_complete": True,
            "api_traces": [],
        }

    client = client or OpenMeteoClient(settings)
    weather, weather_traces = client.fetch_historical_weather(city, start, end)
    air, air_traces = client.fetch_historical_air_quality(city, start, end)
    if weather.empty or air.empty:
        raise RuntimeError(
            f"Open-Meteo returned empty history for {city.key}: weather={len(weather)}, air={len(air)}"
        )
    merged = weather.merge(air, on="timestamp", how="outer", suffixes=("", "_air"))
    lead_traces = []
    if with_lead_weather:
        lead, lead_traces = client.fetch_lead_weather(city, start, end)
        if lead.empty:
            raise RuntimeError(f"Previous-runs weather is empty for {city.key}")
        merged = merged.merge(lead, on="timestamp", how="left")
    merged["city"] = city.key
    merged["province"] = city.province
    merged["latitude"] = city.latitude
    merged["longitude"] = city.longitude
    merged["elevation_m"] = city.elevation_m
    merged["data_source"] = "open_meteo"
    merged["ingested_at"] = datetime.now(UTC)
    path = store.write_city(city.key, merged, merge=True)
    rows_stored = len(store.read_city(city.key, columns=["timestamp"]))
    client.clear_city_cache(city)
    return {
        "city": city.key,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "rows_received": len(merged),
        "rows_stored": rows_stored,
        "path": str(path),
        "with_lead_weather": with_lead_weather,
        "skipped_complete": False,
        "api_traces": [
            trace.to_dict() for trace in [*weather_traces, *air_traces, *lead_traces]
        ],
    }


def _payload(
    city_keys: list[str],
    reports: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    rate_limited: bool = False,
    retry_after_seconds: int | None = None,
    next_city: str | None = None,
) -> dict[str, Any]:
    completed = {report["city"] for report in reports}
    failed_keys = {failure["city"] for failure in failures}
    remaining = [key for key in city_keys if key not in completed and key not in failed_keys]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "requested": len(city_keys),
        "succeeded": len(reports),
        "failed": len(failures),
        "rate_limited": rate_limited,
        "retry_after_seconds": retry_after_seconds,
        "next_city": next_city,
        "remaining_cities": remaining,
        "cities": reports,
        "failures": failures,
    }


def backfill_many(
    settings: Settings,
    city_keys: list[str],
    *,
    days: int | None = None,
    with_lead_weather: bool = True,
    wait_on_rate_limit: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    store = LocalStore(settings)
    client = OpenMeteoClient(settings)
    index = 0

    while index < len(city_keys):
        key = city_keys[index]
        try:
            LOGGER.info("Backfilling %s", key)
            reports.append(
                backfill_city(
                    settings,
                    settings.city(key),
                    days=days,
                    with_lead_weather=with_lead_weather,
                    client=client,
                    force=force,
                )
            )
            index += 1
            store.save_report(
                "backfill_report_v6.json",
                _payload(city_keys, reports, failures, next_city=city_keys[index] if index < len(city_keys) else None),
            )
        except RateLimitError as exc:
            payload = _payload(
                city_keys,
                reports,
                failures,
                rate_limited=True,
                retry_after_seconds=exc.retry_after_seconds,
                next_city=key,
            )
            store.save_report("backfill_report_v6.json", payload)
            if not wait_on_rate_limit:
                LOGGER.error(
                    "Provider rate limit reached. Stop now and rerun after %s seconds; cached chunks will resume.",
                    exc.retry_after_seconds,
                )
                return payload
            LOGGER.warning(
                "Provider rate limit reached. Sleeping %s seconds, then resuming %s from cached chunks.",
                exc.retry_after_seconds,
                key,
            )
            time.sleep(exc.retry_after_seconds)
            # Retry the same city. Successfully downloaded chunks are read from
            # the cache, so the run does not start that city from zero.
        except Exception as exc:
            LOGGER.exception("Backfill failed for %s", key)
            failures.append({"city": key, "error": str(exc)})
            index += 1
            store.save_report(
                "backfill_report_v6.json",
                _payload(city_keys, reports, failures, next_city=city_keys[index] if index < len(city_keys) else None),
            )

    payload = _payload(city_keys, reports, failures)
    store.save_report("backfill_report_v6.json", payload)
    return payload
