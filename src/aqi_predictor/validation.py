from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .storage import LocalStore


def _city_report(city: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"city": city, "ready": False, "issues": ["no stored data"]}
    timestamp = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    required = ["timestamp", "us_aqi", "pm2_5", "pm10", "temperature_2m"]
    issues: list[str] = []
    missing = [column for column in required if column not in frame]
    if missing:
        issues.append(f"missing required columns: {missing}")
    duplicates = int(timestamp.duplicated().sum())
    if duplicates:
        issues.append(f"{duplicates} duplicate timestamps")
    rows = len(frame)
    valid_aqi = float(pd.to_numeric(frame.get("us_aqi"), errors="coerce").notna().mean()) if "us_aqi" in frame else 0
    if valid_aqi < 0.90:
        issues.append(f"AQI coverage is only {valid_aqi:.1%}")
    lead_columns = [
        f"lead_{lead}d_{variable}"
        for lead in (1, 2, 3)
        for variable in (
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
        )
        if f"lead_{lead}d_{variable}" in frame
    ]
    lead_coverage = (
        float(frame[lead_columns].notna().mean().mean()) if lead_columns else 0.0
    )
    if lead_coverage < 0.65:
        issues.append(f"lead-weather coverage is only {lead_coverage:.1%}")
    target_std = float(pd.to_numeric(frame.get("us_aqi"), errors="coerce").std()) if "us_aqi" in frame else 0
    if not np.isfinite(target_std) or target_std < 5:
        issues.append(f"AQI variation is too low: std={target_std}")
    return {
        "city": city,
        "ready": not issues,
        "issues": issues,
        "rows": rows,
        "first_timestamp": timestamp.min().isoformat() if timestamp.notna().any() else None,
        "last_timestamp": timestamp.max().isoformat() if timestamp.notna().any() else None,
        "duplicate_timestamps": duplicates,
        "aqi_non_null_ratio": valid_aqi,
        "lead_weather_non_null_ratio": lead_coverage,
        "aqi_std": target_std,
        "column_count": len(frame.columns),
    }


def validate_training_data(
    settings: Settings,
    city_keys: list[str],
) -> dict[str, Any]:
    store = LocalStore(settings)
    city_reports = [_city_report(key, store.read_city(key)) for key in city_keys]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requested_cities": len(city_keys),
        "ready_cities": sum(report["ready"] for report in city_reports),
        "ready_for_training": all(report["ready"] for report in city_reports),
        "cities": city_reports,
    }
    store.save_report("training_data_validation_v6.json", payload)
    return payload
