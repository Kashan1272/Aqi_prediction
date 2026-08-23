from __future__ import annotations

import argparse
import json

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.features import build_national_training_frame, feature_columns
from aqi_predictor.matrix_validation import audit_training_matrix
from aqi_predictor.models import chronological_partitions
from aqi_predictor.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the exact daily feature/target matrix before training."
    )
    parser.add_argument("--city", default="all")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    store = LocalStore(settings)
    histories = []
    for key in keys:
        frame = store.read_city(key)
        if not frame.empty:
            histories.append((settings.city(key), frame))
    training = build_national_training_frame(histories, forecast_days=settings.forecast_days)

    snapshots = []
    for key in keys:
        frame = store.read_provider_snapshots(key)
        if frame.empty:
            continue
        frame["city"] = key
        snapshots.append(frame)
    if snapshots:
        provider = pd.concat(snapshots, ignore_index=True, copy=False, sort=False)
        provider["issue_date"] = pd.to_datetime(provider["issue_date"], errors="coerce").dt.normalize()
        provider["horizon_day"] = pd.to_numeric(provider["horizon_day"], errors="coerce").astype("Int64")
        provider = provider.drop(columns=["target_date", "collected_at"], errors="ignore")
        training = training.merge(
            provider,
            on=["city", "issue_date", "horizon_day"],
            how="left",
            validate="many_to_one",
        )

    numeric, categorical = feature_columns(training)
    train, validation, test = chronological_partitions(training)
    report = audit_training_matrix(
        training,
        numeric_features=numeric,
        categorical_features=categorical,
        train=train,
        validation=validation,
        test=test,
        selected_cities=keys,
        settings=settings,
    )
    store.save_report("training_matrix_audit_v66.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.strict and not report["ready_for_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
