from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.probe import probe_city


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every configured data provider.")
    parser.add_argument("--city", default="lahore")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--strict", action="store_true", help="Fail only when a required Open-Meteo source fails")
    parser.add_argument("--strict-optional", action="store_true", help="Also fail when a configured optional provider is unavailable")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    report = probe_city(settings, args.city, days=args.days)
    print(json.dumps(report, indent=2))
    if args.strict and not report["required_sources_ok"]:
        raise SystemExit(1)
    if args.strict_optional and not report["configured_optional_sources_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
