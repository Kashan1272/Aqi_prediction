from __future__ import annotations

import pandas as pd

from aqi_predictor.models import chronological_partitions, rolling_folds


def _frame(days: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "issue_date": dates,
        "target_date": dates + pd.Timedelta(days=1),
        "city": "lahore",
        "horizon_day": 1,
        "target_aqi_mean": range(days),
    })


def test_chronological_partitions_apply_embargo():
    train, validation, test = chronological_partitions(_frame(), embargo_days=3)
    assert train["issue_date"].max() < validation["issue_date"].min()
    assert validation["issue_date"].max() < test["issue_date"].min()
    assert (validation["issue_date"].min() - train["issue_date"].max()).days >= 4
    assert (test["issue_date"].min() - validation["issue_date"].max()).days >= 4


def test_rolling_folds_are_forward_only():
    folds = rolling_folds(_frame(), folds=3, embargo_days=3)
    assert folds
    for train, validation in folds:
        assert train["issue_date"].max() < validation["issue_date"].min()
