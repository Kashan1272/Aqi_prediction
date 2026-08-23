from __future__ import annotations

import argparse
import json

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.city_selection import (
    save_selected_city_profile,
    select_training_cities,
)
from aqi_predictor.config import get_settings
from aqi_predictor.features import (
    SENSOR_TARGET_COLUMNS,
    TARGET_COLUMNS,
    build_national_training_frame,
    feature_columns,
)
from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.matrix_validation import audit_training_matrix
from aqi_predictor.models import chronological_partitions
from aqi_predictor.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select precision training cities from development OOF metrics."
    )
    parser.add_argument("--city", default="all")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    candidate_keys = (
        list(settings.cities)
        if args.city.strip().lower() == "all"
        else [item.strip() for item in args.city.split(",") if item.strip()]
    )
    store = LocalStore(settings)
    histories = []
    for key in candidate_keys:
        frame = store.read_city(key)
        if frame.empty:
            raise SystemExit(f"No hourly history is stored for {key}")
        histories.append((settings.city(key), frame))
    training = build_national_training_frame(
        histories, forecast_days=settings.forecast_days
    )

    snapshot_frames = []
    for key in candidate_keys:
        snapshot = store.read_provider_snapshots(key)
        if snapshot.empty:
            continue
        snapshot["city"] = key
        snapshot_frames.append(snapshot)
    if snapshot_frames:
        snapshots = pd.concat(snapshot_frames, ignore_index=True, copy=False, sort=False)
        snapshots["issue_date"] = pd.to_datetime(
            snapshots["issue_date"], errors="coerce"
        ).dt.normalize()
        snapshots["horizon_day"] = pd.to_numeric(
            snapshots["horizon_day"], errors="coerce"
        ).astype("Int64")
        snapshots = snapshots.drop(columns=["target_date", "collected_at"], errors="ignore")
        training = training.merge(
            snapshots,
            on=["city", "issue_date", "horizon_day"],
            how="left",
            validate="many_to_one",
        )

    numeric_features, categorical_features = feature_columns(training)
    for column in list(dict.fromkeys(numeric_features + TARGET_COLUMNS + SENSOR_TARGET_COLUMNS)):
        if column in training:
            training[column] = pd.to_numeric(
                training[column], errors="coerce", downcast="float"
            )
    train, validation, test = chronological_partitions(training)
    development = pd.concat([train, validation], ignore_index=True, copy=False)
    result = select_training_cities(
        development,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        candidate_keys=candidate_keys,
        settings=settings,
    )
    save_selected_city_profile(settings, result)
    store.save_report("city_selection_v67.json", result.to_dict())

    selected = result.selected_cities
    selected_training = training[training["city"].astype(str).isin(selected)].copy()
    selected_train = train[train["city"].astype(str).isin(selected)].copy()
    selected_validation = validation[
        validation["city"].astype(str).isin(selected)
    ].copy()
    selected_test = test[test["city"].astype(str).isin(selected)].copy()
    audit = audit_training_matrix(
        selected_training,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train=selected_train,
        validation=selected_validation,
        test=selected_test,
        selected_cities=selected,
        settings=settings,
    )
    store.save_report("training_matrix_audit_selected_v67.json", audit)
    print(json.dumps({
        "candidate_cities": candidate_keys,
        "mandatory_cities": result.mandatory_cities,
        "selected_cities": selected,
        "rejected_cities": result.rejected_cities,
        "selected_matrix_ready": audit.get("ready_for_training"),
        "selected_matrix_errors": audit.get("errors"),
        "selection_report": str(settings.reports_dir / "city_selection_v67.json"),
        "profile": str(settings.selected_city_profile_path),
    }, indent=2))
    if settings.matrix_strict and not audit.get("ready_for_training"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
