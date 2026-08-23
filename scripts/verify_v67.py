from __future__ import annotations

import aqi_predictor
from aqi_predictor.city_selection import load_selected_city_keys
from aqi_predictor.config import get_settings


def main() -> None:
    settings = get_settings()
    payload = {
        "version": aqi_predictor.__version__,
        "candidate_city_count": len(settings.cities),
        "selection_enabled": settings.city_selection_enabled,
        "target_city_count": settings.city_selection_target_count,
        "mandatory_cities": settings.city_selection_mandatory,
        "recent_fold_weight": settings.stacking_recent_fold_weight,
        "active_profile": load_selected_city_keys(settings),
    }
    print(payload)
    assert aqi_predictor.__version__ == "6.7.0"
    assert len(settings.cities) == 10
    assert settings.city_selection_enabled is True
    assert settings.city_selection_target_count == 8
    assert settings.city_selection_mandatory == ("karachi", "multan")
    assert settings.stacking_recent_fold_weight >= 1.0
    print("v6.7 development-selected precision verified successfully.")


if __name__ == "__main__":
    main()
