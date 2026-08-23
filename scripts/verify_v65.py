from __future__ import annotations

import aqi_predictor
from aqi_predictor.config import get_settings
from aqi_predictor.features import CATEGORICAL_FEATURES
from aqi_predictor.models import RecencyWeightedRegressor, build_candidate_models


def main() -> None:
    settings = get_settings()
    candidates = build_candidate_models(
        ["x", "us_aqi__mean", "us_aqi__mean__lag1", "us_aqi__mean__lag7"],
        CATEGORICAL_FEATURES,
        random_seed=settings.random_seed,
        n_jobs=settings.model_n_jobs,
        horizon_day=2,
        target="mean",
        recency_half_life_days=settings.recency_half_life_days,
    )
    payload = {
        "version": aqi_predictor.__version__,
        "cities": len(settings.cities),
        "city_keys": list(settings.cities),
        "climate_clusters": sorted({city.climate_cluster for city in settings.cities.values()}),
        "categorical_features": CATEGORICAL_FEATURES,
        "candidate_models": list(candidates),
        "recent_candidates_valid": all(
            isinstance(candidates[name], RecencyWeightedRegressor)
            for name in ("recent_hist_gradient", "recent_extra_trees")
        ),
        "stacking_folds": settings.stacking_cv_folds,
        "max_base_models": settings.stacking_max_base_models,
        "hopsworks_feature_group_version": settings.hopsworks_feature_group_version,
    }
    print(payload)
    assert payload["version"] == "6.5.0"
    assert payload["cities"] == 15
    assert payload["recent_candidates_valid"] is True
    assert "climate_cluster" in CATEGORICAL_FEATURES
    assert "aqi_regime" in CATEGORICAL_FEATURES
    assert settings.hopsworks_feature_group_version == 8
    print("v6.5 regime-aware core-15 training verified successfully.")


if __name__ == "__main__":
    main()
