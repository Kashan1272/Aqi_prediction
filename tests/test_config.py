from aqi_predictor.config import get_settings


def test_pakistan_city_catalogue():
    settings = get_settings()
    assert len(settings.cities) == 10
    assert list(settings.cities) == [
        "lahore", "karachi", "islamabad", "faisalabad", "multan",
        "peshawar", "quetta", "hyderabad", "sialkot", "abbottabad",
    ]
    assert settings.city("lahore").province == "Punjab"
    assert settings.city("karachi").province == "Sindh"
    assert settings.stacking_city_balanced is True


def test_precision_city_selection_defaults():
    settings = get_settings()
    assert settings.city_selection_enabled is True
    assert settings.city_selection_target_count == 8
    assert settings.city_selection_mandatory == ("karachi", "multan")
    assert settings.stacking_recent_fold_weight >= 1.0
    assert settings.day3_recent_fold_weight >= settings.stacking_recent_fold_weight
    assert settings.day3_local_expert_max_weight >= settings.local_expert_max_weight
    assert settings.champion_min_day3_r2_gain >= 0.0
    assert settings.hopsworks_feature_group_version == 9
