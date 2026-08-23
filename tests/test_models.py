from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_predictor.models import (
    OOFStackingRegressor,
    fit_convex_blend,
    regression_metrics,
    seasonal_baseline,
)


def test_metrics_and_baseline():
    frame = pd.DataFrame({
        "horizon_day": [1, 2, 3],
        "us_aqi__mean": [100, 100, 100],
        "us_aqi__mean__lag1": [90, 90, 90],
        "us_aqi__mean__lag7": [80, 80, 80],
    })
    baseline = seasonal_baseline(frame, "target_aqi_mean")
    assert len(baseline) == 3
    metrics = regression_metrics([90, 95, 100], np.array([89, 94, 101]))
    assert metrics.rmse < 2
    assert metrics.r2 > 0.9


def test_convex_oof_blend_is_non_negative_and_improves():
    rng = np.random.default_rng(42)
    truth = np.linspace(40, 200, 300) + rng.normal(0, 4, 300)
    first = truth + rng.normal(0, 12, 300)
    second = truth + 8 + rng.normal(0, 9, 300)
    baseline = truth - 10 + rng.normal(0, 14, 300)
    matrix = np.column_stack([first, second, baseline])
    weights, intercept, info = fit_convex_blend(matrix, truth)
    prediction = matrix @ weights + intercept
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert info["rows"] == 300
    assert regression_metrics(truth, prediction).rmse <= min(
        regression_metrics(truth, first).rmse,
        regression_metrics(truth, second).rmse,
        regression_metrics(truth, baseline).rmse,
    )


def test_city_balanced_sample_weights_change_stacking_objective():
    rng = np.random.default_rng(7)
    truth_large = np.linspace(40, 160, 900)
    truth_small = np.linspace(180, 260, 100)
    truth = np.concatenate([truth_large, truth_small])
    first = np.concatenate([
        truth_large + rng.normal(0, 4, len(truth_large)),
        truth_small + 28 + rng.normal(0, 4, len(truth_small)),
    ])
    second = np.concatenate([
        truth_large - 12 + rng.normal(0, 5, len(truth_large)),
        truth_small + rng.normal(0, 4, len(truth_small)),
    ])
    matrix = np.column_stack([first, second])
    unweighted, _, _ = fit_convex_blend(matrix, truth, l2_regularization=0.0)
    sample_weight = np.concatenate([
        np.full(len(truth_large), 1 / len(truth_large)),
        np.full(len(truth_small), 1 / len(truth_small)),
    ])
    balanced, _, info = fit_convex_blend(
        matrix, truth, sample_weight=sample_weight, l2_regularization=0.0
    )
    assert info["sample_weighted"] is True
    assert balanced[1] > unweighted[1]


def test_v67_stacking_artifact_remains_backward_compatible():
    class ConstantModel:
        def predict(self, frame):
            return np.full(len(frame), 100.0)

    model = OOFStackingRegressor(
        base_models={"constant": ConstantModel()},
        weights={"constant": 1.0},
        intercept=0.0,
        target_column="target_aqi_mean",
    )
    # Simulate an object pickled before the v6.8 fields existed.
    for name in (
        "local_models",
        "local_weights",
        "local_algorithms",
        "month_offsets",
        "city_month_offsets",
        "recent_city_offsets",
    ):
        model.__dict__.pop(name, None)
    frame = pd.DataFrame({
        "city": ["karachi"],
        "province": ["Sindh"],
        "issue_date": [pd.Timestamp("2026-07-29")],
    })
    assert np.isclose(model.predict(frame)[0], 100.0)
