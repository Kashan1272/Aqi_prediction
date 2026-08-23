from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _bootstrap import ROOT
from aqi_predictor.config import get_settings


def metrics(group: pd.DataFrame) -> dict[str, Any]:
    actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(group["prediction"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(group["baseline"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & np.isfinite(baseline)
    actual, prediction, baseline = actual[valid], prediction[valid], baseline[valid]
    if not len(actual):
        return {"rows": 0, "mae": None, "rmse": None, "r2": None, "baseline_rmse": None}
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)) if len(actual) > 1 else None,
        "bias": float(np.mean(prediction - actual)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(actual, baseline))),
        "beats_baseline": bool(mean_squared_error(actual, prediction) < mean_squared_error(actual, baseline)),
    }


def grouped(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(key): metrics(group)
        for key, group in frame.groupby(column, observed=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose untouched AQI test performance by regime.")
    parser.add_argument("--predictions", default="reports/test_predictions_v6.parquet")
    parser.add_argument("--output", default="reports/regime_test_report_v65.json")
    args = parser.parse_args()

    path = Path(args.predictions)
    if not path.is_absolute():
        path = ROOT / path
    frame = pd.read_parquet(path)
    if frame.empty:
        raise SystemExit("The test prediction file is empty.")

    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    frame["month"] = frame["target_date"].dt.month
    frame["season"] = pd.cut(
        frame["month"],
        bins=[0, 2, 3, 6, 9, 10, 12],
        labels=["winter_smog", "spring", "pre_monsoon", "monsoon", "post_monsoon", "winter_smog_2"],
        include_lowest=True,
    ).astype(str).replace("winter_smog_2", "winter_smog")

    settings = get_settings()
    cluster_map = {key: city.climate_cluster for key, city in settings.cities.items()}
    province_map = {key: city.province for key, city in settings.cities.items()}
    if "climate_cluster" not in frame:
        frame["climate_cluster"] = frame["city"].astype(str).map(cluster_map).fillna("unknown")
    if "province" not in frame:
        frame["province"] = frame["city"].astype(str).map(province_map).fillna("unknown")
    if "aqi_regime" not in frame:
        frame["aqi_regime"] = frame["season"]

    mean_frame = frame[frame["target_kind"] == "mean"].copy()
    output = {
        "overall_daily_mean": metrics(mean_frame),
        "by_city": grouped(mean_frame, "city"),
        "by_climate_cluster": grouped(mean_frame, "climate_cluster"),
        "by_aqi_regime": grouped(mean_frame, "aqi_regime"),
        "by_month": grouped(mean_frame, "month"),
        "by_horizon_day": grouped(mean_frame, "horizon_day"),
        "worst_cities_by_rmse": [],
    }
    city_rank = sorted(
        ((city, values) for city, values in output["by_city"].items() if values["rmse"] is not None),
        key=lambda item: item[1]["rmse"],
        reverse=True,
    )
    output["worst_cities_by_rmse"] = [
        {"city": city, **values} for city, values in city_rank[:8]
    ]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
