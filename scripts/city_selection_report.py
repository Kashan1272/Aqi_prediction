from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the development-only city selection used by v6.7."
    )
    parser.add_argument(
        "--report",
        default="reports/city_selection_v67.json",
    )
    args = parser.parse_args()
    settings = get_settings()
    path = Path(args.report)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        raise SystemExit(
            f"City-selection report not found: {path}. Run training first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "methodology": payload.get("methodology"),
        "test_data_used_for_selection": payload.get("test_data_used_for_selection"),
        "mandatory_cities": payload.get("mandatory_cities"),
        "selected_cities": payload.get("selected_cities"),
        "rejected_cities": payload.get("rejected_cities"),
        "scores": {
            key: {
                "selected": value.get("selected"),
                "mandatory": value.get("mandatory"),
                "selection_score": value.get("selection_score"),
                "oof_r2": (value.get("model") or {}).get("r2"),
                "rmse_gain_vs_baseline": value.get("rmse_gain_vs_baseline"),
                "selection_reason": value.get("selection_reason"),
            }
            for key, value in (payload.get("city_scores") or {}).items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
