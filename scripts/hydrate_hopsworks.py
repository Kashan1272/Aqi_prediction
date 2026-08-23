from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.hopsworks_integration import HopsworksAdapter
from aqi_predictor.storage import LocalStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate local data/model from Hopsworks.")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    if not args.features and not args.model:
        args.features = args.model = True

    settings = get_settings()
    adapter = HopsworksAdapter(settings)
    store = LocalStore(settings)
    output = {}

    if args.features:
        frame = adapter.read_group("aqi_hourly_v69")
        if frame.empty:
            raise RuntimeError("Hopsworks feature group aqi_hourly_v6 is empty")
        counts = {}
        for city, subset in frame.groupby("city", observed=True):
            path = store.write_city(str(city), subset, merge=False)
            counts[str(city)] = {"rows": len(subset), "path": str(path)}
        output["features"] = counts
        try:
            snapshots = adapter.read_group("aqi_provider_snapshots_v69")
        except Exception as exc:
            output["provider_snapshots"] = {"available": False, "detail": str(exc)}
        else:
            snapshot_counts = {}
            if not snapshots.empty:
                snapshots = snapshots.drop(columns=["snapshot_id"], errors="ignore")
                for city, subset in snapshots.groupby("city", observed=True):
                    path = store.write_provider_snapshots(str(city), subset)
                    snapshot_counts[str(city)] = {"rows": len(subset), "path": str(path)}
            output["provider_snapshots"] = snapshot_counts

    if args.model:
        destination = settings.project_root / "artifacts" / "models" / "_hopsworks_latest"
        downloaded = adapter.download_latest_model(destination)
        model_file = downloaded / "model.joblib"
        report_file = downloaded / "report.json"
        if not model_file.exists() or not report_file.exists():
            raise RuntimeError(
                "Downloaded Hopsworks model package does not contain model.joblib and report.json"
            )
        pointer = settings.project_root / "artifacts" / "models" / "production.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({
            "model_name": settings.model_name,
            "version": "hopsworks_latest",
            "version_path": str(downloaded),
            "source": "hopsworks",
        }, indent=2), encoding="utf-8")
        output["model"] = str(downloaded)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
