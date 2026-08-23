from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.pipeline import backfill_many


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe, quota-aware historical weather, AQI and lead-weather backfill. "
            "The configured v6.3 profile contains 25 Pakistani cities."
        )
    )
    parser.add_argument("--city", default="all", help="City key, comma-separated keys, or all")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--without-lead-weather", action="store_true")
    parser.add_argument(
        "--wait-on-rate-limit",
        dest="wait_on_rate_limit",
        action="store_true",
        default=True,
        help="Wait for provider reset and resume cached chunks (default).",
    )
    parser.add_argument(
        "--no-wait-on-rate-limit",
        dest="wait_on_rate_limit",
        action="store_false",
        help="Exit cleanly on HTTP 429; rerun later to resume cached chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore complete-city coverage checks. Cached API chunks are still reused.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=0,
        help="Process only the first N selected cities in this run (0 means all).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    city_keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    if args.max_cities > 0:
        city_keys = city_keys[: args.max_cities]

    print(
        json.dumps(
            {
                "configured_cities": len(settings.cities),
                "selected_cities": len(city_keys),
                "days": args.days or settings.backfill_days,
                "chunk_days": settings.chunk_days,
                "request_pause_seconds": settings.request_pause_seconds,
                "quota_soft_limits": {
                    "minute": settings.api_quota_max_per_minute,
                    "hour": settings.api_quota_max_per_hour,
                    "day": settings.api_quota_max_per_day,
                },
                "wait_on_rate_limit": args.wait_on_rate_limit,
                "cache_enabled": settings.backfill_cache_enabled,
            },
            indent=2,
        )
    )

    report = backfill_many(
        settings,
        city_keys,
        days=args.days,
        with_lead_weather=not args.without_lead_weather,
        wait_on_rate_limit=args.wait_on_rate_limit,
        force=args.force,
    )
    print(json.dumps(report, indent=2))
    if report.get("rate_limited"):
        raise SystemExit(75)
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
