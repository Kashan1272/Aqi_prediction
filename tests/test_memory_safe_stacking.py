from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from aqi_predictor.config import get_settings
from aqi_predictor.training import _fit_oof_stacking_model


class MemoryFailRegressor(BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        raise MemoryError("Unable to allocate synthetic test array")

    def predict(self, x):
        return np.zeros(len(x), dtype=float)


def _pipeline(model):
    return Pipeline([
        ("features", ColumnTransformer(
            [("numeric", "passthrough", ["x", "us_aqi__mean__lag1", "us_aqi__mean__lag7"])],
            remainder="drop",
        )),
        ("model", model),
    ])


def test_oof_stacking_recovers_from_candidate_memory_error():
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    rows = []
    for city, province, shift in [
        ("lahore", "Punjab", 0.0),
        ("karachi", "Sindh", 12.0),
    ]:
        for index, date in enumerate(dates):
            signal = 90 + shift + 20 * np.sin(index / 18)
            rows.append({
                "issue_date": date,
                "city": city,
                "province": province,
                "horizon_day": 1,
                "x": signal,
                "us_aqi__mean": signal,
                "us_aqi__mean__lag1": signal - 1,
                "us_aqi__mean__lag7": signal - 2,
                "target_aqi_mean": signal + 2,
            })
    frame = pd.DataFrame(rows).sort_values(["issue_date", "city"]).reset_index(drop=True)
    settings = replace(
        get_settings(),
        stacking_memory_recovery=True,
        stacking_max_base_models=1,
        stacking_cv_folds=3,
        stacking_min_component_weight=0.0,
    )
    model, report, residuals = _fit_oof_stacking_model(
        {
            "good": _pipeline(Ridge(alpha=1.0)),
            "bad": _pipeline(MemoryFailRegressor()),
        },
        frame,
        "target_aqi_mean",
        settings=settings,
        folds=3,
    )
    prediction = model.predict(frame.tail(10))
    assert np.isfinite(prediction).all()
    assert len(residuals) > 0
    assert report["candidate_failures"]["bad"]["memory_related"] is True
    assert len(model.base_models) <= 1
