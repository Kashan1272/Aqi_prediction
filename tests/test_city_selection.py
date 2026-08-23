from __future__ import annotations

import json
from dataclasses import replace

from aqi_predictor.city_selection import load_selected_city_keys
from aqi_predictor.config import get_settings


def test_selected_city_profile_keeps_mandatory_cities(tmp_path):
    settings = replace(get_settings(), project_root=tmp_path)
    profile = settings.selected_city_profile_path
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        json.dumps({
            "selected_cities": ["karachi", "multan", "lahore"],
        }),
        encoding="utf-8",
    )
    selected = load_selected_city_keys(settings)
    assert selected == ["karachi", "multan", "lahore"]


def test_invalid_selected_city_profile_falls_back_to_catalogue(tmp_path):
    settings = replace(get_settings(), project_root=tmp_path)
    profile = settings.selected_city_profile_path
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("not json", encoding="utf-8")
    assert load_selected_city_keys(settings) == list(settings.cities)
