from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.city_selection import load_selected_city_keys
from aqi_predictor.inference import forecast_city
from aqi_predictor.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate live three-day forecasts.")
    parser.add_argument("--city", default="all")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    city_keys = (
        load_selected_city_keys(settings)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    failures = []
    for city in city_keys:
        try:
            payload = forecast_city(settings, city)
            print(json.dumps({
                "city": city,
                "generated_at": payload["generated_at"],
                "daily_forecast": payload["daily_forecast"],
            }, indent=2))
        except Exception as exc:
            failures.append({"city": city, "error": str(exc)})
            print(json.dumps(failures[-1], indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
