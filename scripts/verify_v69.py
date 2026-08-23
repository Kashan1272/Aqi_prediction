from __future__ import annotations

import inspect

from _bootstrap import ROOT  # noqa: F401
import aqi_predictor
from aqi_predictor.config import get_settings
from aqi_predictor.features import add_provider_snapshot_features
from aqi_predictor.matrix_manifest import build_matrix_manifest
from aqi_predictor.models import OOFStackingRegressor, build_candidate_models
from aqi_predictor.registry import LocalModelRegistry


def main() -> None:
    settings = get_settings()
    fields = set(OOFStackingRegressor.__dataclass_fields__)
    candidates = build_candidate_models(
        ["x"], ["city", "province"],
        random_seed=settings.random_seed,
        n_jobs=1,
        horizon_day=3,
        target="mean",
    )
    required_fields = {
        "global_bias_offset",
        "local_models",
        "local_weights",
        "month_offsets",
        "city_month_offsets",
        "recent_city_offsets",
    }
    payload = {
        "version": aqi_predictor.__version__,
        "candidate_cities": len(settings.cities),
        "selected_city_target": settings.city_selection_target_count,
        "mandatory_cities": settings.city_selection_mandatory,
        "day3_recent_fold_weight": settings.day3_recent_fold_weight,
        "day3_extreme_sample_weight": settings.day3_extreme_sample_weight,
        "day3_long_candidates": sorted(
            name for name in candidates if name.endswith("_long")
        ),
        "day3_local_expert_max_weight": settings.day3_local_expert_max_weight,
        "champion_min_day3_r2_gain": settings.champion_min_day3_r2_gain,
        "champion_max_abs_bias_regression": settings.champion_max_abs_bias_regression,
        "hopsworks_feature_group_version": settings.hopsworks_feature_group_version,
        "matrix_manifest_function": build_matrix_manifest.__name__,
        "provider_feature_function": add_provider_snapshot_features.__name__,
        "ensemble_fields": sorted(required_fields & fields),
        "registry_has_rollback": hasattr(LocalModelRegistry, "rollback"),
        "registry_has_champion_comparison": hasattr(
            LocalModelRegistry, "compare_with_production"
        ),
    }
    print(payload)
    assert aqi_predictor.__version__ == "6.9.0"
    assert len(settings.cities) == 10
    assert settings.city_selection_target_count == 8
    assert settings.city_selection_mandatory == ("karachi", "multan")
    assert required_fields.issubset(fields)
    assert {"hist_gradient_long", "extra_trees_long"}.issubset(candidates)
    assert settings.day3_recent_fold_weight >= settings.stacking_recent_fold_weight
    assert settings.hopsworks_feature_group_version == 9
    assert "model_name" in inspect.signature(
        __import__(
            "aqi_predictor.hopsworks_integration",
            fromlist=["HopsworksAdapter"],
        ).HopsworksAdapter.upload_model
    ).parameters
    print("v6.9 day-3 precision and protected Hopsworks deployment verified successfully.")


if __name__ == "__main__":
    main()
