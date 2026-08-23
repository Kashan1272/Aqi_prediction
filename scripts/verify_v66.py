from __future__ import annotations

import inspect

from _bootstrap import ROOT  # noqa: F401
import aqi_predictor
from aqi_predictor.config import get_settings
from aqi_predictor.models import fit_convex_blend


def main() -> None:
    settings = get_settings()
    payload = {
        "version": aqi_predictor.__version__,
        "cities": list(settings.cities),
        "city_count": len(settings.cities),
        "stacking_city_balanced": settings.stacking_city_balanced,
        "stacking_folds": settings.stacking_cv_folds,
        "max_base_models": settings.stacking_max_base_models,
        "matrix_strict": settings.matrix_strict,
        "matrix_min_rows_per_city": settings.matrix_min_rows_per_city,
        "matrix_min_issue_dates": settings.matrix_min_issue_dates,
        "fit_convex_blend_signature": str(inspect.signature(fit_convex_blend)),
    }
    print(payload)
    assert aqi_predictor.__version__ == "6.6.0"
    assert len(settings.cities) == 10
    assert settings.stacking_city_balanced is True
    assert "sample_weight" in inspect.signature(fit_convex_blend).parameters
    print("v6.6 core-10 precision training verified successfully.")


if __name__ == "__main__":
    main()
