from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.providers import OpenAQClient
from aqi_predictor.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill optional OpenAQ sensor observations for calibration training."
    )
    parser.add_argument("--city", default="all")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.openaq_api_key:
        raise SystemExit("OPENAQ_API_KEY is not configured")
    city_keys = list(settings.cities) if args.city == "all" else [
        item.strip() for item in args.city.split(",") if item.strip()
    ]
    end = datetime.now(UTC) - timedelta(days=1)
    start = end - timedelta(days=args.days)
    client = OpenAQClient(settings)
    store = LocalStore(settings)
    reports = []
    failures = []
    for key in city_keys:
        try:
            frame, traces = client.historical_hourly(
                settings.city(key),
                datetime_from=start.isoformat(),
                datetime_to=end.isoformat(),
            )
            if frame.empty:
                reports.append({
                    "city": key,
                    "rows": 0,
                    "status": "no_station_or_no_hourly_data",
                    "traces": [trace.to_dict() for trace in traces],
                })
                continue
            path = store.write_city(key, frame, merge=True)
            reports.append({
                "city": key,
                "rows": len(frame),
                "status": "stored",
                "path": str(path),
                "traces": [trace.to_dict() for trace in traces],
            })
        except Exception as exc:
            failures.append({"city": key, "error": str(exc)})
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requested": len(city_keys),
        "reports": reports,
        "failures": failures,
    }
    store.save_report("openaq_observation_backfill_v6.json", payload)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
