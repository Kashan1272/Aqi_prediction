from __future__ import annotations

import pandas as pd

from aqi_predictor.config import get_settings
from aqi_predictor.storage import LocalStore


def test_storage_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AQI_PROJECT_ROOT", str(tmp_path))
    # Config is already imported in this test session, so use the project
    # settings and redirect only the path-bearing frozen dataclass.
    settings = get_settings()
    from dataclasses import replace

    settings = replace(settings, project_root=tmp_path)
    store = LocalStore(settings)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
        "us_aqi": [10, 20, 30, 40],
    })
    path = store.write_city("lahore", frame, merge=False)
    assert path.exists()
    read = store.read_city("lahore")
    assert len(read) == 4
    assert read["us_aqi"].tolist() == [10, 20, 30, 40]
