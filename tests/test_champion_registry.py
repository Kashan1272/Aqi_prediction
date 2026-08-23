from __future__ import annotations

from dataclasses import replace

from aqi_predictor.config import get_settings
from aqi_predictor.registry import LocalModelRegistry


def _report(r2: float, rmse: float) -> dict:
    return {
        "project_version": "test",
        "quality_gate": {"passed": True},
        "test_metrics": {"daily_mean": {"r2": r2, "rmse": rmse, "mae": rmse / 2}},
    }


def test_champion_is_preserved_and_can_be_rolled_back(tmp_path):
    settings = replace(
        get_settings(),
        project_root=tmp_path,
        model_name="test_model",
        champion_min_r2_gain=0.001,
        champion_min_rmse_gain=0.05,
        champion_max_rmse_regression=0.05,
    )
    registry = LocalModelRegistry(settings)
    first = registry.register({"model": 1}, _report(0.74, 17.5), promote=True)
    comparison = registry.compare_with_production(_report(0.73, 17.4))
    assert comparison["candidate_is_better"] is False
    second = registry.register({"model": 2}, _report(0.75, 17.2), promote=True)
    assert registry.production_dir() == second
    restored = registry.rollback()
    assert restored == first
    assert registry.production_dir() == first
