from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .features import SENSOR_TARGET_COLUMNS, TARGET_COLUMNS


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    values = pd.to_datetime(frame.get("issue_date"), errors="coerce").dropna()
    return {
        "start": values.min().isoformat() if len(values) else None,
        "end": values.max().isoformat() if len(values) else None,
    }


def _split_city_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "city" not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame.groupby("city", observed=True).size().items()
    }


def audit_training_matrix(
    training: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected_cities: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Validate the exact matrix used by the model before any fitting starts."""
    errors: list[str] = []
    warnings: list[str] = []

    city_counts = _split_city_counts(training)
    horizon_counts = {
        str(int(key)): int(value)
        for key, value in training.groupby("horizon_day", observed=True).size().items()
    }
    duplicates = int(training.duplicated(["city", "issue_date", "horizon_day"]).sum())
    duplicate_ratio = float(duplicates / max(len(training), 1))
    issue_dates = int(pd.to_datetime(training["issue_date"], errors="coerce").nunique())

    expected = list(dict.fromkeys(selected_cities))
    missing_cities = [city for city in expected if city not in city_counts]
    unexpected_cities = [city for city in city_counts if city not in expected]
    if missing_cities:
        errors.append("Missing configured cities in the matrix: " + ", ".join(missing_cities))
    if unexpected_cities:
        errors.append("Unexpected cities in the matrix: " + ", ".join(unexpected_cities))
    if len(city_counts) != len(expected):
        errors.append(
            f"Expected {len(expected)} cities but generated {len(city_counts)} city matrices"
        )

    low_row_cities = {
        city: rows
        for city, rows in city_counts.items()
        if rows < settings.matrix_min_rows_per_city
    }
    if low_row_cities:
        errors.append(
            "Cities below minimum matrix rows: "
            + ", ".join(f"{city}={rows}" for city, rows in low_row_cities.items())
        )
    if issue_dates < settings.matrix_min_issue_dates:
        errors.append(
            f"Only {issue_dates} unique issue dates; minimum is {settings.matrix_min_issue_dates}"
        )
    if duplicate_ratio > settings.matrix_max_duplicate_ratio:
        errors.append(
            f"Duplicate city/date/horizon ratio {duplicate_ratio:.4%} exceeds "
            f"{settings.matrix_max_duplicate_ratio:.4%}"
        )

    expected_horizons = {str(day) for day in range(1, settings.forecast_days + 1)}
    missing_horizons = sorted(expected_horizons - set(horizon_counts))
    if missing_horizons:
        errors.append("Missing forecast horizons: " + ", ".join(missing_horizons))

    feature_names = list(dict.fromkeys(numeric_features + categorical_features))
    target_leakage = [name for name in feature_names if name in TARGET_COLUMNS + SENSOR_TARGET_COLUMNS]
    lead_leakage = [name for name in numeric_features if name.startswith("lead_")]
    future_target_leakage = [
        name for name in feature_names
        if name.startswith("target_aqi")
        or name.startswith("target_observed")
        or name in {"actual", "prediction", "baseline"}
    ]
    leakage = sorted(set(target_leakage + lead_leakage + future_target_leakage))
    if leakage:
        errors.append("Leakage-prone feature columns detected: " + ", ".join(leakage))

    if len(numeric_features) < settings.matrix_min_numeric_features:
        errors.append(
            f"Only {len(numeric_features)} numeric features; minimum is "
            f"{settings.matrix_min_numeric_features}"
        )

    missing_ratio = training[numeric_features].isna().mean() if numeric_features else pd.Series(dtype=float)
    high_missing = {
        str(column): float(value)
        for column, value in missing_ratio.items()
        if float(value) > settings.matrix_max_feature_missing_ratio
    }
    if high_missing:
        warnings.append(
            f"{len(high_missing)} numeric features exceed the configured missingness threshold"
        )

    finite_variation: dict[str, int] = {}
    near_constant: list[str] = []
    for column in numeric_features:
        values = pd.to_numeric(training[column], errors="coerce")
        unique = int(values.dropna().nunique())
        finite_variation[column] = unique
        if unique <= 1:
            near_constant.append(column)
    if near_constant:
        warnings.append(f"{len(near_constant)} numeric features are constant or all-missing")

    required_future = [
        column for column in numeric_features
        if column.startswith("future_temperature_2m")
        or column.startswith("future_relative_humidity_2m")
        or column.startswith("future_surface_pressure")
        or column.startswith("future_wind_speed_10m")
    ]
    future_coverage = (
        float(training[required_future].notna().mean().mean())
        if required_future else 0.0
    )
    if future_coverage < settings.matrix_min_future_weather_coverage:
        errors.append(
            f"Future-weather feature coverage is {future_coverage:.2%}; minimum is "
            f"{settings.matrix_min_future_weather_coverage:.2%}"
        )

    target_summary: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(training[target], errors="coerce")
        finite = values[np.isfinite(values)]
        target_summary[target] = {
            "finite_ratio": float(np.isfinite(values).mean()),
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "unique": int(finite.nunique()),
        }
        if len(finite) != len(training):
            errors.append(f"Target {target} contains non-finite rows after matrix construction")
        if len(finite) and (finite.min() < 0 or finite.max() > 500):
            errors.append(f"Target {target} is outside the expected AQI range 0-500")
        if finite.nunique() < 20:
            errors.append(f"Target {target} has insufficient variation")

    split_ranges = {
        "train": _date_range(train),
        "validation": _date_range(validation),
        "test": _date_range(test),
    }
    train_dates = set(pd.to_datetime(train["issue_date"], errors="coerce").dropna())
    validation_dates = set(pd.to_datetime(validation["issue_date"], errors="coerce").dropna())
    test_dates = set(pd.to_datetime(test["issue_date"], errors="coerce").dropna())
    overlap = {
        "train_validation": len(train_dates & validation_dates),
        "train_test": len(train_dates & test_dates),
        "validation_test": len(validation_dates & test_dates),
    }
    if any(overlap.values()):
        errors.append(f"Chronological split overlap detected: {overlap}")

    split_city_counts = {
        "train": _split_city_counts(train),
        "validation": _split_city_counts(validation),
        "test": _split_city_counts(test),
    }
    for split_name, counts in split_city_counts.items():
        absent = [city for city in expected if counts.get(city, 0) == 0]
        if absent:
            errors.append(
                f"{split_name} split is missing cities: " + ", ".join(absent)
            )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": "v6.9_day3_precision_city_balanced_oof_stack",
        "ready_for_training": not errors,
        "errors": errors,
        "warnings": warnings,
        "matrix": {
            "rows": int(len(training)),
            "columns": int(len(training.columns)),
            "numeric_features": int(len(numeric_features)),
            "categorical_features": categorical_features,
            "selected_cities": expected,
            "city_rows": city_counts,
            "horizon_rows": horizon_counts,
            "unique_issue_dates": issue_dates,
            "duplicate_rows": duplicates,
            "duplicate_ratio": duplicate_ratio,
            "future_weather_coverage": future_coverage,
            "high_missing_features": high_missing,
            "constant_features": near_constant,
        },
        "targets": target_summary,
        "splits": {
            "ranges": split_ranges,
            "rows": {
                "train": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
            },
            "city_rows": split_city_counts,
            "date_overlap": overlap,
        },
    }
    return payload
