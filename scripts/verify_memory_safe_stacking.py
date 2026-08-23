from __future__ import annotations

import inspect

import aqi_predictor
from aqi_predictor.config import get_settings
from aqi_predictor.models import build_candidate_models
from aqi_predictor.training import _fit_oof_stacking_model


def main() -> None:
    settings = get_settings()
    candidates = build_candidate_models(
        ["us_aqi__mean", "us_aqi__mean__lag1", "us_aqi__mean__lag7"],
        ["city", "province"],
        random_seed=settings.random_seed,
        n_jobs=settings.model_n_jobs,
        horizon_day=1,
        target="mean",
    )
    hist = candidates["hist_gradient"].named_steps["model"]
    payload = {
        "version": aqi_predictor.__version__,
        "cities": len(settings.cities),
        "stacking_folds": settings.stacking_cv_folds,
        "max_base_models": settings.stacking_max_base_models,
        "min_component_weight": settings.stacking_min_component_weight,
        "memory_recovery": settings.stacking_memory_recovery,
        "model_n_jobs": settings.model_n_jobs,
        "hist_gradient_max_bins": hist.max_bins,
        "hist_gradient_max_leaf_nodes": hist.max_leaf_nodes,
        "training_function": str(inspect.signature(_fit_oof_stacking_model)),
    }
    print(payload)

    assert aqi_predictor.__version__ == "6.4.2"
    assert len(settings.cities) == 25
    assert settings.stacking_cv_folds == 3
    assert settings.stacking_max_base_models == 2
    assert settings.stacking_memory_recovery is True
    assert settings.model_n_jobs == 1
    assert hist.max_bins == 64
    print("v6.4.2 memory-safe stacking verified successfully.")


if __name__ == "__main__":
    main()
