from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.storage import LocalStore


def city_status(settings, store, key: str, days: int, require_lead: bool) -> dict:
    end = datetime.now(UTC).date() - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    columns = [
        "timestamp",
        "temperature_2m",
        "us_aqi",
        "lead_1d_temperature_2m",
        "lead_2d_temperature_2m",
        "lead_3d_temperature_2m",
    ]
    frame = store.read_city(key, columns=columns)
    expected = days * 24
    if frame.empty or "timestamp" not in frame:
        return {
            "city": key,
            "complete": False,
            "rows": 0,
            "expected_hours": expected,
            "coverage_ratio": 0.0,
            "reason": "no stored history",
        }
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frame = frame[(frame["timestamp"] >= lower) & (frame["timestamp"] < upper)]
    rows = int(frame["timestamp"].nunique())

    def ratio(column: str) -> float:
        return round(float(frame[column].notna().mean()), 4) if column in frame and len(frame) else 0.0

    lead = {
        f"day{day}": ratio(f"lead_{day}d_temperature_2m")
        for day in (1, 2, 3)
    }
    complete = (
        rows >= int(expected * 0.95)
        and ratio("temperature_2m") >= 0.80
        and ratio("us_aqi") >= 0.80
        and (not require_lead or all(value >= 0.60 for value in lead.values()))
    )
    return {
        "city": key,
        "complete": complete,
        "rows": rows,
        "expected_hours": expected,
        "coverage_ratio": round(rows / expected, 4) if expected else 0.0,
        "weather_non_null_ratio": ratio("temperature_2m"),
        "aqi_non_null_ratio": ratio("us_aqi"),
        "lead_temperature_non_null_ratio": lead,
        "start": frame["timestamp"].min().isoformat() if rows else None,
        "end": frame["timestamp"].max().isoformat() if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check resume-safe backfill completeness for every city.")
    parser.add_argument("--city", default="all")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--without-lead-weather", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    days = args.days or settings.backfill_days
    keys = list(settings.cities) if args.city.lower() == "all" else [x.strip() for x in args.city.split(",") if x.strip()]
    store = LocalStore(settings)
    cities = [city_status(settings, store, key, days, not args.without_lead_weather) for key in keys]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requested": len(keys),
        "complete": sum(bool(item["complete"]) for item in cities),
        "incomplete": sum(not bool(item["complete"]) for item in cities),
        "cities": cities,
    }
    store.save_report("backfill_status_v6.json", payload)
    print(json.dumps(payload, indent=2))
    if args.strict and payload["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
