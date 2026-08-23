from __future__ import annotations

import argparse
import json
import os

# Limit native learner thread pools before NumPy/scikit-learn are imported.
# This reduces peak RAM on Windows while MODEL_N_JOBS controls tree models.
_native_threads = os.getenv("MODEL_N_JOBS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, _native_threads)

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.training import train_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three-day daily AQI ensemble.")
    parser.add_argument("--city", default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    city_keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    result = train_project(
        settings,
        city_keys=city_keys,
        allow_promotion=not args.quick,
        quick=args.quick,
    )
    print(json.dumps({
        "version_path": str(result.version_path),
        "promoted": result.promoted,
        "quality_gate_passed": result.quality_gate_passed,
        "report": str(result.report_path),
        "test_metrics": result.test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
