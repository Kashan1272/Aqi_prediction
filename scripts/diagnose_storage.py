from __future__ import annotations

import argparse

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check readability of locally stored AQI history partitions."
    )
    parser.add_argument(
        "--city",
        default="all",
        help="City key, comma-separated city keys, or all.",
    )
    parser.add_argument(
        "--group",
        default="aqi_history",
        help=(
            "Compatibility option. v6 stores historical data as city files; "
            "the supported logical group is aqi_history."
        ),
    )
    args = parser.parse_args()

    if args.group.strip().lower() != "aqi_history":
        parser.error("Only --group aqi_history is supported by the v6 local store.")

    settings = get_settings()
    store = LocalStore(settings)
    city_keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )

    total = 0
    failures = 0
    for key in city_keys:
        try:
            frame = store.read_city(key)
            if frame.empty:
                print(f"FAIL city={key}: no data")
                failures += 1
                continue
            total += len(frame)
            print(
                f"OK   city={key:24s} rows={len(frame):7d} "
                f"columns={len(frame.columns):3d}"
            )
        except Exception as exc:
            print(f"FAIL city={key}: {exc}")
            failures += 1

    print(
        f"group=aqi_history partitions={len(city_keys)} "
        f"rows={total} failures={failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
