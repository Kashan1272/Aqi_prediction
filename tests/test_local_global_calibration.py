from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from aqi_predictor.config import get_settings
from aqi_predictor.local_experts import fit_hierarchical_calibration, fit_local_experts
from aqi_predictor.models import regression_metrics


def _ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer(
            [("numeric", "passthrough", ["x", "us_aqi__mean__lag1", "us_aqi__mean__lag7"])],
            remainder="drop",
        )),
        ("model", Ridge(alpha=1.0)),
    ])


def test_local_experts_improve_city_specific_oof_bias():
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    rows = []
    for city, province, shift in [("karachi", "Sindh", 18.0), ("multan", "Punjab", -14.0)]:
        for index, date in enumerate(dates):
            x = 80 + 15 * np.sin(index / 20) + shift
            rows.append({
                "issue_date": date,
                "city": city,
                "province": province,
                "horizon_day": 2,
                "x": x,
                "us_aqi__mean__lag1": x - 1,
                "us_aqi__mean__lag7": x - 2,
                "target_aqi_mean": x + 0.2 * shift,
            })
    frame = pd.DataFrame(rows).sort_values(["issue_date", "city"]).reset_index(drop=True)
    oof = frame[frame["issue_date"] >= dates[180]].copy().reset_index(drop=True)
    boundaries = np.array_split(np.arange(len(dates[180:])), 3)
    date_to_fold = {
        date: fold
        for fold, indices in enumerate(boundaries, start=1)
        for date in dates[180:][indices]
    }
    oof["oof_fold"] = oof["issue_date"].map(date_to_fold)
    oof["actual"] = oof["target_aqi_mean"]
    global_prediction = oof["x"].to_numpy(dtype=np.float32)

    settings = replace(
        get_settings(),
        local_experts_enabled=True,
        local_expert_candidates=("ridge",),
        local_expert_min_rows=240,
        local_expert_min_oof_rows=120,
        local_expert_max_weight=0.5,
        local_expert_weight_shrinkage=10.0,
        local_expert_min_gain=0.001,
        local_expert_max_cities_per_target=2,
    )
    result = fit_local_experts(
        {"ridge": _ridge_pipeline()},
        frame,
        "target_aqi_mean",
        oof,
        global_prediction,
        settings=settings,
    )
    before = regression_metrics(oof["actual"], global_prediction)
    after = regression_metrics(oof["actual"], result.oof_prediction)
    assert after.rmse < before.rmse
    assert set(result.models) == {"karachi", "multan"}
    assert all(0 < weight <= 0.5 for weight in result.weights.values())


def test_hierarchical_calibration_reduces_month_and_city_bias():
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    oof = pd.DataFrame({
        "issue_date": np.tile(dates, 2),
        "city": np.repeat(["karachi", "multan"], len(dates)),
        "province": np.repeat(["Sindh", "Punjab"], len(dates)),
        "oof_fold": np.tile(np.repeat([1, 2, 3], 80), 2),
    })
    baseline = np.full(len(oof), 100.0, dtype=np.float32)
    month = pd.to_datetime(oof["issue_date"]).dt.month.to_numpy()
    city_bias = np.where(oof["city"].to_numpy() == "multan", 8.0, -4.0)
    month_bias = np.where(np.isin(month, [11, 12, 1, 2]), 6.0, 0.0)
    actual = baseline + city_bias + month_bias
    settings = replace(
        get_settings(),
        stacking_province_shrinkage=20.0,
        stacking_city_shrinkage=20.0,
        calibration_month_shrinkage=20.0,
        calibration_city_month_shrinkage=30.0,
        calibration_recent_city_enabled=True,
        calibration_recent_city_shrinkage=40.0,
    )
    calibrated, offsets, report = fit_hierarchical_calibration(
        oof, actual, baseline, settings=settings
    )
    assert regression_metrics(actual, calibrated).rmse < regression_metrics(actual, baseline).rmse
    assert offsets["city"]
    assert offsets["month"]
    assert report["metrics_after"]["rmse"] < report["metrics_before"]["rmse"]
