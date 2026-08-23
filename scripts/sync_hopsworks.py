from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.features import add_provider_snapshot_features, build_national_training_frame
from aqi_predictor.hopsworks_integration import HopsworksAdapter
from aqi_predictor.registry import LocalModelRegistry
from aqi_predictor.storage import LocalStore


def _upload_registered_model(
    adapter: HopsworksAdapter,
    registry: LocalModelRegistry,
    *,
    role: str,
    version: str | None,
) -> dict[str, object]:
    if role == "champion":
        _, report, version_dir = registry.load_production()
        model_name = registry.settings.model_name
    else:
        if version:
            _, report, version_dir = registry.load_version(version)
        else:
            version_dir = registry.latest_candidate_dir()
            _, report, version_dir = registry.load_version(version_dir.name)
        model_name = registry.settings.hopsworks_candidate_model_name

    metrics = report["test_metrics"]["daily_mean"]
    local_manifest_path = version_dir / "registry_manifest.json"
    local_manifest = (
        json.loads(local_manifest_path.read_text(encoding="utf-8"))
        if local_manifest_path.exists()
        else {}
    )
    hopsworks_version = adapter.upload_model(
        version_dir,
        {
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias": metrics.get("bias", float("nan")),
            "day3_r2": report["test_metrics"]["by_day"]["day3"]["model"]["r2"],
            "day3_rmse": report["test_metrics"]["by_day"]["day3"]["model"]["rmse"],
            "day3_bias": report["test_metrics"]["by_day"]["day3"]["model"].get("bias", float("nan")),
        },
        model_name=model_name,
        tags={
            "role": role,
            "local_version": version_dir.name,
            "project_version": report.get("project_version", "unknown"),
            "quality_gate_passed": bool(report.get("quality_gate", {}).get("passed", False)),
            "locally_promoted": bool(local_manifest.get("promoted", role == "champion")),
            "selected_cities": report.get("training_data", {}).get("selected_cities", []),
            "day3_r2": report.get("test_metrics", {}).get("by_day", {}).get("day3", {}).get("model", {}).get("r2"),
            "daily_mean_bias": report.get("test_metrics", {}).get("daily_mean", {}).get("bias"),
            "matrix_schema_sha256": report.get("matrix_manifest", {}).get("schema_sha256", "unknown"),
            "matrix_data_contract_sha256": report.get("matrix_manifest", {}).get("data_contract_sha256", "unknown"),
        },
        description=(
            "Production champion. Uploading a challenger never changes this model name."
            if role == "champion"
            else "Evaluation challenger. It is isolated from the production champion model name."
        ),
    )
    return {
        "role": role,
        "model_name": model_name,
        "hopsworks_version": hopsworks_version,
        "local_version": version_dir.name,
        "local_version_path": str(version_dir),
        "metrics": {key: metrics[key] for key in ("r2", "rmse", "mae", "bias") if key in metrics},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize features and champion/challenger models with Hopsworks."
    )
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--model", "--production", dest="production", action="store_true")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--version", help="Specific local challenger version to upload")
    parser.add_argument("--city", default="all")
    args = parser.parse_args()
    if not args.features and not args.production and not args.candidate:
        args.features = args.production = True

    settings = get_settings()
    adapter = HopsworksAdapter(settings)
    store = LocalStore(settings)
    registry = LocalModelRegistry(settings)
    city_keys = list(settings.cities) if args.city == "all" else args.city.split(",")
    output: dict[str, object] = {}

    if args.features:
        hourly_frames = []
        histories = []
        for key in city_keys:
            key = key.strip()
            frame = store.read_city(key)
            if frame.empty:
                continue
            hourly_frames.append(frame)
            histories.append((settings.city(key), frame))
        if not hourly_frames:
            raise ValueError("No hourly city data is available for Hopsworks synchronization")
        hourly = pd.concat(hourly_frames, ignore_index=True, copy=False, sort=False)
        output["hourly"] = adapter.upsert(
            "aqi_hourly_v69",
            hourly,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
        )
        snapshot_frames = []
        for key in city_keys:
            snapshot = store.read_provider_snapshots(key.strip())
            if snapshot.empty:
                continue
            snapshot["city"] = key.strip()
            snapshot["snapshot_id"] = (
                snapshot["city"].astype(str)
                + "_"
                + snapshot["issue_date"].astype(str)
                + "_d"
                + snapshot["horizon_day"].astype(str)
            )
            snapshot_frames.append(snapshot)
        all_snapshots = (
            pd.concat(snapshot_frames, ignore_index=True, copy=False, sort=False)
            if snapshot_frames else pd.DataFrame()
        )
        if not all_snapshots.empty:
            output["provider_snapshots"] = adapter.upsert(
                "aqi_provider_snapshots_v69",
                all_snapshots,
                primary_key=["snapshot_id"],
                event_time="issue_date",
            )
        daily = build_national_training_frame(
            histories, forecast_days=settings.forecast_days
        )
        if not all_snapshots.empty:
            matrix_snapshots = all_snapshots.copy()
            matrix_snapshots["issue_date"] = pd.to_datetime(
                matrix_snapshots["issue_date"], errors="coerce"
            ).dt.normalize()
            matrix_snapshots["horizon_day"] = pd.to_numeric(
                matrix_snapshots["horizon_day"], errors="coerce"
            ).astype("Int64")
            matrix_snapshots = matrix_snapshots.drop(
                columns=["snapshot_id", "target_date", "collected_at"],
                errors="ignore",
            )
            matrix_snapshots = matrix_snapshots.drop_duplicates(
                ["city", "issue_date", "horizon_day"], keep="last"
            )
            daily = daily.merge(
                matrix_snapshots,
                on=["city", "issue_date", "horizon_day"],
                how="left",
                validate="many_to_one",
            )
        daily = add_provider_snapshot_features(daily)
        daily["feature_id"] = (
            daily["city"].astype(str)
            + "_"
            + daily["issue_date"].astype(str)
            + "_d"
            + daily["horizon_day"].astype(str)
        )
        output["daily_training"] = adapter.upsert(
            "aqi_daily_training_v69",
            daily,
            primary_key=["feature_id"],
            event_time="issue_date",
        )
        try:
            _, production_report, _ = registry.load_production()
            manifest = production_report.get("matrix_manifest", {})
            if manifest.get("schema_sha256"):
                manifest_row = pd.DataFrame([{
                    "schema_sha256": manifest.get("schema_sha256"),
                    "data_contract_sha256": manifest.get("data_contract_sha256"),
                    "generated_at": manifest.get("generated_at"),
                    "project_version": production_report.get("project_version"),
                    "selected_cities": json.dumps(
                        production_report.get("training_data", {}).get(
                            "selected_cities", []
                        )
                    ),
                    "numeric_feature_count": len(
                        manifest.get("schema", {}).get("numeric_features", [])
                    ),
                }])
                output["matrix_manifest"] = adapter.upsert(
                    "aqi_matrix_manifest_v69",
                    manifest_row,
                    primary_key=["schema_sha256"],
                    event_time="generated_at",
                )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        prediction_rows = []
        for key in city_keys:
            payload = store.load_prediction(key.strip())
            for row in payload.get("daily_forecast", []):
                prediction_rows.append({
                    "prediction_id": f"{key}_{row['date']}",
                    "city": key.strip(),
                    "date": row["date"],
                    "aqi_mean": row["aqi_mean"],
                    "aqi_max": row["aqi_max"],
                    "category": row["category"],
                    "generated_at": payload.get("generated_at"),
                })
        if prediction_rows:
            output["predictions"] = adapter.upsert(
                "aqi_daily_predictions_v69",
                pd.DataFrame(prediction_rows),
                primary_key=["prediction_id"],
                event_time="date",
            )

    if args.candidate:
        output["candidate_model"] = _upload_registered_model(
            adapter, registry, role="challenger", version=args.version
        )
    if args.production:
        output["production_model"] = _upload_registered_model(
            adapter, registry, role="champion", version=None
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
