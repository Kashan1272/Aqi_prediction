from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from aqi_predictor.config import get_settings
from aqi_predictor.features import (
    _add_future_interactions,
    _add_history_features,
    add_provider_snapshot_features,
    feature_columns,
)
from aqi_predictor.matrix_manifest import build_matrix_manifest
from aqi_predictor.models import build_candidate_models
from aqi_predictor.registry import LocalModelRegistry


def test_pressure_anomaly_and_provider_consensus_are_available():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    daily = pd.DataFrame({
        "date": dates,
        "us_aqi__mean": np.linspace(90, 140, len(dates)),
        "us_aqi__max": np.linspace(110, 170, len(dates)),
        "surface_pressure__mean": 1005 + np.sin(np.arange(len(dates)) / 3),
        "temperature_2m__mean": 25 + np.sin(np.arange(len(dates)) / 5),
        "relative_humidity_2m__mean": 55 + np.cos(np.arange(len(dates)) / 6),
        "wind_speed_10m__mean": 3 + np.sin(np.arange(len(dates)) / 7),
    })
    history = _add_history_features(daily)
    row = history.iloc[-1:].copy()
    row["horizon_day"] = 3
    row["future_surface_pressure__mean"] = 1007.0
    row["future_temperature_2m__mean"] = 30.0
    row["future_relative_humidity_2m__mean"] = 60.0
    row["future_dew_point_2m__mean"] = 20.0
    row["future_wind_speed_10m__mean"] = 2.5
    row["future_wind_direction_10m__mean"] = 180.0
    row["future_rain__sum"] = 0.0
    row["future_precipitation__sum"] = 0.0
    enriched = _add_future_interactions(row)
    assert np.isfinite(enriched["future_pressure_anomaly"].iloc[0])
    assert np.isfinite(enriched["aqi_horizon_trend_projection"].iloc[0])

    enriched["provider_open_meteo_aqi_mean"] = 120.0
    enriched["provider_openweather_aqi_mean"] = 126.0
    enriched["provider_aqicn_aqi_mean"] = 123.0
    enriched = add_provider_snapshot_features(enriched)
    assert enriched["provider_mean_consensus"].iloc[0] == 123.0
    assert enriched["provider_mean_range"].iloc[0] == 6.0


def test_all_missing_numeric_feature_is_not_selected():
    frame = pd.DataFrame({
        "issue_date": pd.date_range("2024-01-01", periods=4),
        "target_date": pd.date_range("2024-01-02", periods=4),
        "horizon_day": [1, 1, 1, 1],
        "city": ["karachi"] * 4,
        "province": ["Sindh"] * 4,
        "useful": [1.0, 2.0, 3.0, 4.0],
        "all_missing": [np.nan] * 4,
        "target_aqi_mean": [10.0, 11.0, 12.0, 13.0],
        "target_aqi_max": [12.0, 13.0, 14.0, 15.0],
    })
    numeric, categorical = feature_columns(frame)
    assert "useful" in numeric
    assert "all_missing" not in numeric
    assert categorical == ["city", "province"]


def test_day3_has_long_horizon_candidates():
    candidates = build_candidate_models(
        ["x"], ["city", "province"], random_seed=42, n_jobs=1,
        horizon_day=3, target="mean",
    )
    assert "hist_gradient_long" in candidates
    assert "extra_trees_long" in candidates


def _full_report(r2: float, rmse: float, bias: float, day3_r2: float) -> dict:
    return {
        "project_version": "test",
        "quality_gate": {"passed": True},
        "test_metrics": {
            "daily_mean": {"r2": r2, "rmse": rmse, "mae": rmse / 2, "bias": bias},
            "by_day": {"day3": {"model": {"r2": day3_r2, "rmse": rmse + 3, "bias": bias}}},
            "macro_city": {"r2": 0.5},
        },
    }


def test_champion_guard_requires_day3_gain_and_bias_safety(tmp_path):
    settings = replace(
        get_settings(),
        project_root=tmp_path,
        model_name="day3_guard_test",
        champion_min_r2_gain=0.0005,
        champion_min_day3_r2_gain=0.0005,
        champion_max_abs_bias_regression=0.15,
    )
    registry = LocalModelRegistry(settings)
    registry.register(
        {"champion": True}, _full_report(0.7473, 17.31, -0.02, 0.6311),
        promote=True,
    )
    worse_day3 = registry.compare_with_production(
        _full_report(0.7500, 17.20, 0.01, 0.6290)
    )
    assert worse_day3["candidate_is_better"] is False
    better = registry.compare_with_production(
        _full_report(0.7500, 17.20, 0.08, 0.6400)
    )
    assert better["candidate_is_better"] is True


def test_matrix_manifest_is_deterministic():
    frame = pd.DataFrame({
        "issue_date": pd.date_range("2024-01-01", periods=6),
        "target_date": pd.date_range("2024-01-02", periods=6),
        "horizon_day": [1, 2, 3, 1, 2, 3],
        "city": ["karachi"] * 6,
        "province": ["Sindh"] * 6,
        "x": np.arange(6, dtype=float),
        "target_aqi_mean": np.arange(10, 16, dtype=float),
        "target_aqi_max": np.arange(12, 18, dtype=float),
    })
    kwargs = dict(
        numeric_features=["x"], categorical_features=["city", "province"],
        train=frame.iloc[:3], validation=frame.iloc[3:4], test=frame.iloc[4:],
        selected_cities=["karachi"], removed_features={},
    )
    first = build_matrix_manifest(frame, **kwargs)
    second = build_matrix_manifest(frame, **kwargs)
    assert first["schema_sha256"] == second["schema_sha256"]
    assert first["data_contract_sha256"] == second["data_contract_sha256"]
