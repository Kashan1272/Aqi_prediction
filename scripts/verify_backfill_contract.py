from __future__ import annotations

import inspect

import aqi_predictor
import aqi_predictor.pipeline as pipeline
from aqi_predictor.config import get_settings


def main() -> None:
    settings = get_settings()
    signature = inspect.signature(pipeline.backfill_many)
    required = {"wait_on_rate_limit", "force", "with_lead_weather", "days"}
    present = set(signature.parameters)
    missing = sorted(required - present)
    payload = {
        "version": getattr(aqi_predictor, "__version__", "unknown"),
        "pipeline_loaded_from": pipeline.__file__,
        "cities": len(settings.cities),
        "backfill_many_signature": str(signature),
        "missing_parameters": missing,
        "quota_day": getattr(settings, "api_quota_max_per_day", None),
        "chunk_days": settings.chunk_days,
        "request_pause_seconds": settings.request_pause_seconds,
    }
    print(payload)
    if missing:
        raise SystemExit(
            "backfill_many is still stale; missing parameters: " + ", ".join(missing)
        )
    if len(settings.cities) != 25:
        raise SystemExit(f"Expected 25 configured cities, found {len(settings.cities)}")
    if payload["quota_day"] != 7500:
        raise SystemExit(f"Expected quota_day=7500, found {payload['quota_day']}")
    print("v6.3.2 backfill contract verified successfully.")


if __name__ == "__main__":
    main()
