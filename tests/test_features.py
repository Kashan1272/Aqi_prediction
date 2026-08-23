from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_predictor.config import get_settings
from aqi_predictor.features import build_city_training_frame, feature_columns


def test_daily_training_contract():
    settings = get_settings()
    city = settings.city("lahore")
    periods = 80 * 24
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "us_aqi": 80 + 15 * np.sin(np.arange(periods) / 72),
        "pm2_5": 35 + 5 * np.sin(np.arange(periods) / 48),
        "pm10": 55 + 5 * np.sin(np.arange(periods) / 48),
        "temperature_2m": 25 + 4 * np.sin(np.arange(periods) / 24),
        "relative_humidity_2m": 60,
        "surface_pressure": 1005,
        "wind_speed_10m": 2,
    })
    for lead in (1, 2, 3):
        for column in (
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
        ):
            frame[f"lead_{lead}d_{column}"] = frame[column].shift(-lead * 24)
    training = build_city_training_frame(frame, city)
    assert not training.empty
    assert set(training["horizon_day"].unique()) == {1, 2, 3}
    assert {"target_aqi_mean", "target_aqi_max"}.issubset(training)
    numeric, categorical = feature_columns(training)
    assert "future_temperature_2m__mean" in numeric
    assert "us_aqi__mean__lag28" in numeric
    assert "us_aqi__mean__same_weekday_4w_mean" in numeric
    assert categorical == ["city", "province"]
