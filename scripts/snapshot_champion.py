from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.registry import LocalModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Preserve or initialize the current production champion.")
    parser.add_argument("--initialize-if-missing", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    registry = LocalModelRegistry(settings)
    if registry.pointer.exists():
        payload = json.loads(registry.pointer.read_text(encoding="utf-8"))
        registry._append_history(payload, action="pre_v69_champion_snapshot")
        print(json.dumps({"status": "snapshotted", **payload}, indent=2))
        return

    if not args.initialize_if_missing:
        raise FileNotFoundError("production.json is missing")
    report_path = settings.reports_dir / "training_report_v6.json"
    if not report_path.exists():
        raise FileNotFoundError("No training report exists to initialize the champion")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    quality_passed = bool(report.get("quality_gate", {}).get("passed", False))
    version_path = Path(str(report.get("version_path", "")))
    if not quality_passed or not (version_path / "model.joblib").exists():
        raise ValueError(
            "The latest report is not a passing, complete model; production was not initialized."
        )
    registry.promote_version(version_path.name, reason="initialize_pre_v69_champion")
    payload = json.loads(registry.pointer.read_text(encoding="utf-8"))
    registry._append_history(payload, action="pre_v69_champion_snapshot")
    print(json.dumps({"status": "initialized_and_snapshotted", **payload}, indent=2))


if __name__ == "__main__":
    main()
