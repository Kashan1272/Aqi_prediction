from __future__ import annotations

import argparse
import json
import math

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.providers.open_meteo import (
    AIR_VARIABLES,
    HISTORICAL_WEATHER_VARIABLES,
    LEAD_WEATHER_VARIABLES,
)


def weighted_units(days: int, variables: int, chunk_days: int) -> tuple[int, int]:
    chunks = math.ceil(days / chunk_days)
    days_per_request_units = max(1, math.ceil(chunk_days / 14))
    variable_units = max(1, math.ceil(variables / 10))
    return chunks, chunks * days_per_request_units * variable_units


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Open-Meteo backfill workload.")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    settings = get_settings()
    days = args.days or settings.backfill_days
    cities = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    chunk_days = settings.chunk_days

    weather_chunks, weather_units = weighted_units(
        days,
        len(HISTORICAL_WEATHER_VARIABLES),
        chunk_days,
    )
    air_chunks, air_units = weighted_units(days, len(AIR_VARIABLES), chunk_days)
    lead_variable_count = len(LEAD_WEATHER_VARIABLES) * 3
    lead_chunks, lead_units = weighted_units(days, lead_variable_count, chunk_days)

    physical_calls_per_city = weather_chunks + air_chunks + lead_chunks
    weighted_calls_per_city = weather_units + air_units + lead_units
    physical_calls = physical_calls_per_city * len(cities)
    weighted_calls = weighted_calls_per_city * len(cities)
    estimated_seconds = physical_calls * settings.request_pause_seconds

    payload = {
        "cities": len(cities),
        "days": days,
        "chunk_days": chunk_days,
        "physical_calls_estimate": physical_calls,
        "weighted_calls_estimate": weighted_calls,
        "soft_daily_budget": getattr(settings, 'api_quota_max_per_day', 7500),
        "within_project_daily_soft_budget": weighted_calls <= getattr(settings, 'api_quota_max_per_day', 7500),
        "minimum_paced_runtime_hours": round(estimated_seconds / 3600, 2),
        "per_city": {
            "physical_calls": physical_calls_per_city,
            "weighted_calls": weighted_calls_per_city,
            "weather_variables": len(HISTORICAL_WEATHER_VARIABLES),
            "air_variables": len(AIR_VARIABLES),
            "lead_series": lead_variable_count,
        },
        "note": (
            "This is a conservative project-side estimate. Provider-side HTTP 429 may still "
            "occur when the public IP has other Open-Meteo traffic; the backfill waits and resumes."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
