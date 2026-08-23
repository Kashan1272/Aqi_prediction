from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_predictor.config import get_settings
from aqi_predictor.matrix_validation import audit_training_matrix


def _frame() -> pd.DataFrame:
    settings = get_settings()
    dates = pd.date_range("2024-01-01", periods=540, freq="D")
    rows = []
    for city in settings.cities:
        for date in dates:
            for horizon in (1, 2, 3):
                rows.append({
                    "city": city,
                    "province": settings.city(city).province,
                    "issue_date": date,
                    "target_date": date + pd.Timedelta(days=horizon),
                    "horizon_day": horizon,
                    "target_aqi_mean": 80 + horizon + np.sin(date.dayofyear / 20),
                    "target_aqi_max": 110 + horizon + np.cos(date.dayofyear / 20),
                    "future_temperature_2m__mean": 25.0,
                    "future_relative_humidity_2m__mean": 55.0,
                    "future_surface_pressure__mean": 1008.0,
                    "future_wind_speed_10m__mean": 8.0,
                    **{f"f{i}": float(i + horizon) for i in range(70)},
                })
    return pd.DataFrame(rows)


def test_training_matrix_audit_passes_balanced_core10():
    settings = get_settings()
    frame = _frame()
    dates = sorted(frame["issue_date"].unique())
    train = frame[frame["issue_date"] <= dates[350]]
    validation = frame[(frame["issue_date"] > dates[350]) & (frame["issue_date"] <= dates[440])]
    test = frame[frame["issue_date"] > dates[440]]
    numeric = [
        "future_temperature_2m__mean",
        "future_relative_humidity_2m__mean",
        "future_surface_pressure__mean",
        "future_wind_speed_10m__mean",
        *[f"f{i}" for i in range(70)],
    ]
    report = audit_training_matrix(
        frame,
        numeric_features=numeric,
        categorical_features=["city", "province"],
        train=train,
        validation=validation,
        test=test,
        selected_cities=list(settings.cities),
        settings=settings,
    )
    assert report["ready_for_training"] is True
    assert report["matrix"]["rows"] == len(frame)
    assert report["splits"]["date_overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
