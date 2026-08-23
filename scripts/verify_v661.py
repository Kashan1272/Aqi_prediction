from __future__ import annotations

import aqi_predictor
from aqi_predictor.config import get_settings
from aqi_predictor.features import CATEGORICAL_FEATURES
from aqi_predictor.models import iter_rolling_folds


def main() -> None:
    settings = get_settings()
    payload = {
        "version": aqi_predictor.__version__,
        "cities": len(settings.cities),
        "categorical_features": CATEGORICAL_FEATURES,
        "rolling_fold_function": iter_rolling_folds.__name__,
    }
    print(payload)

    assert aqi_predictor.__version__ == "6.6.1"
    assert len(settings.cities) == 10
    assert CATEGORICAL_FEATURES == ["city", "province"]
    print("v6.6.1 feature contract verified successfully.")


if __name__ == "__main__":
    main()
