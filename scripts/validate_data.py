from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.validation import validate_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stored historical training data.")
    parser.add_argument("--city", default="all")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    city_keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    report = validate_training_data(settings, city_keys)
    print(json.dumps(report, indent=2))
    if args.strict and not report["ready_for_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
