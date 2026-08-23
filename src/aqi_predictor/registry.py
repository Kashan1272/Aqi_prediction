from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import Settings


class LocalModelRegistry:
    """Versioned champion/challenger registry with safe rollback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.model_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_root = settings.project_root / "artifacts" / "models"
        self.pointer = self.registry_root / "production.json"
        self.history_path = self.registry_root / "production_history.jsonl"
        self.candidate_index_path = self.registry_root / "candidate_index.json"
        self.snapshots_dir = self.registry_root / "production_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metrics(report: dict[str, Any]) -> dict[str, float]:
        test_metrics = report.get("test_metrics", {})
        metrics = test_metrics.get("daily_mean", {})
        day3 = test_metrics.get("by_day", {}).get("day3", {}).get("model", {})
        macro_city = test_metrics.get("macro_city", {})
        return {
            "r2": float(metrics.get("r2", float("nan"))),
            "rmse": float(metrics.get("rmse", float("nan"))),
            "mae": float(metrics.get("mae", float("nan"))),
            "bias": float(metrics.get("bias", float("nan"))),
            "day3_r2": float(day3.get("r2", float("nan"))),
            "day3_rmse": float(day3.get("rmse", float("nan"))),
            "day3_bias": float(day3.get("bias", float("nan"))),
            "macro_city_r2": float(macro_city.get("r2", float("nan"))),
        }

    @staticmethod
    def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
        return {
            key: (float(value) if np.isfinite(value) else None)
            for key, value in metrics.items()
        }

    def compare_with_production(self, candidate_report: dict[str, Any]) -> dict[str, Any]:
        candidate = self._metrics(candidate_report)
        if not self.pointer.exists():
            return {
                "production_exists": False,
                "candidate_is_better": True,
                "reason": "No production champion exists yet.",
                "candidate_metrics": self._json_safe_metrics(candidate),
                "production_metrics": None,
            }
        try:
            _, production_report, production_dir = self.load_production()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {
                "production_exists": False,
                "candidate_is_better": True,
                "reason": "Production pointer was unavailable or invalid.",
                "candidate_metrics": self._json_safe_metrics(candidate),
                "production_metrics": None,
            }
        production = self._metrics(production_report)
        r2_gain = candidate["r2"] - production["r2"]
        rmse_gain = production["rmse"] - candidate["rmse"]
        day3_r2_gain = candidate["day3_r2"] - production["day3_r2"]
        absolute_bias_regression = abs(candidate["bias"]) - abs(production["bias"])
        r2_path = (
            r2_gain >= self.settings.champion_min_r2_gain
            and candidate["rmse"]
            <= production["rmse"] + self.settings.champion_max_rmse_regression
        )
        rmse_path = (
            rmse_gain >= self.settings.champion_min_rmse_gain
            and candidate["r2"] >= production["r2"]
        )
        candidate_day3_finite = np.isfinite(candidate["day3_r2"])
        production_day3_finite = np.isfinite(production["day3_r2"])
        day3_guard = bool(
            (not candidate_day3_finite and not production_day3_finite)
            or (
                candidate_day3_finite
                and (
                    not production_day3_finite
                    or day3_r2_gain >= self.settings.champion_min_day3_r2_gain
                )
            )
        )
        candidate_bias_finite = np.isfinite(candidate["bias"])
        production_bias_finite = np.isfinite(production["bias"])
        bias_guard = bool(
            (not candidate_bias_finite and not production_bias_finite)
            or (
                candidate_bias_finite
                and (
                    not production_bias_finite
                    or absolute_bias_regression
                    <= self.settings.champion_max_abs_bias_regression
                )
            )
        )
        overall_guard = bool(r2_path or rmse_path)
        better = bool(overall_guard and day3_guard and bias_guard)
        failed_guards: list[str] = []
        if not overall_guard:
            failed_guards.append("overall R²/RMSE")
        if not day3_guard:
            failed_guards.append("day-3 R² improvement")
        if not bias_guard:
            failed_guards.append("absolute bias")
        return {
            "production_exists": True,
            "candidate_is_better": better,
            "reason": (
                "Candidate improves overall accuracy and day-3 R² without violating the bias guard."
                if better
                else "Candidate remains a challenger because it failed: "
                + ", ".join(failed_guards)
                + "."
            ),
            "candidate_metrics": self._json_safe_metrics(candidate),
            "production_metrics": self._json_safe_metrics(production),
            "r2_gain": float(r2_gain),
            "rmse_gain": float(rmse_gain),
            "day3_r2_gain": float(day3_r2_gain),
            "absolute_bias_regression": float(absolute_bias_regression),
            "production_version": production_dir.name,
            "production_version_path": str(production_dir),
            "guards": {
                "minimum_r2_gain": self.settings.champion_min_r2_gain,
                "minimum_rmse_gain": self.settings.champion_min_rmse_gain,
                "maximum_rmse_regression": self.settings.champion_max_rmse_regression,
                "minimum_day3_r2_gain": self.settings.champion_min_day3_r2_gain,
                "maximum_absolute_bias_regression": self.settings.champion_max_abs_bias_regression,
                "overall_guard_passed": overall_guard,
                "day3_guard_passed": day3_guard,
                "bias_guard_passed": bias_guard,
            },
        }

    def register(
        self,
        model: Any,
        report: dict[str, Any],
        *,
        promote: bool,
    ) -> Path:
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        version_dir = self.root / version
        version_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, version_dir / "model.joblib", compress=3)
        report_path = version_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )
        manifest = {
            "model_name": self.settings.model_name,
            "local_version": version,
            "registered_at": datetime.now(UTC).isoformat(),
            "role": "champion" if promote else "challenger",
            "promoted": bool(promote),
            "project_version": report.get("project_version"),
            "metrics": self._json_safe_metrics(self._metrics(report)),
            "quality_gate_passed": bool(
                report.get("quality_gate", {}).get("passed", False)
            ),
            "version_path": str(version_dir),
        }
        self._atomic_json(version_dir / "registry_manifest.json", manifest)
        self._update_candidate_index(manifest)
        if promote:
            self.promote_version(version, reason="training_champion_gate")
        return version_dir

    def _update_candidate_index(self, manifest: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"model_name": self.settings.model_name, "versions": []}
        if self.candidate_index_path.exists():
            try:
                payload = json.loads(self.candidate_index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        versions = [
            item for item in payload.get("versions", [])
            if item.get("local_version") != manifest.get("local_version")
        ]
        versions.append(manifest)
        versions.sort(key=lambda item: str(item.get("local_version", "")), reverse=True)
        payload["versions"] = versions
        self._atomic_json(self.candidate_index_path, payload)

    def _append_history(self, payload: dict[str, Any], *, action: str) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **payload,
            "history_action": action,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self._atomic_json(self.snapshots_dir / f"{stamp}_{action}.json", record)

    def promote_version(self, version: str, *, reason: str = "manual") -> Path:
        version_dir = self.root / str(version)
        if not (version_dir / "model.joblib").exists() or not (version_dir / "report.json").exists():
            raise FileNotFoundError(f"Model version is incomplete: {version_dir}")
        if self.pointer.exists():
            current = json.loads(self.pointer.read_text(encoding="utf-8"))
            self._append_history(current, action="replaced_champion")
        payload = {
            "model_name": self.settings.model_name,
            "version": version_dir.name,
            "version_path": str(version_dir),
            "promoted_at": datetime.now(UTC).isoformat(),
            "reason": reason,
        }
        self._atomic_json(self.pointer, payload)
        manifest_path = version_dir / "registry_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"role": "champion", "promoted": True, "promoted_at": payload["promoted_at"]})
            self._atomic_json(manifest_path, manifest)
            self._update_candidate_index(manifest)
        return version_dir

    def rollback(self, *, steps: int = 1) -> Path:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if not self.history_path.exists():
            raise FileNotFoundError("No production history exists for rollback")
        records: list[dict[str, Any]] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = Path(str(item.get("version_path", "")))
            if path.exists() and (path / "model.joblib").exists():
                records.append(item)
        if len(records) < steps:
            raise FileNotFoundError(f"Only {len(records)} rollback target(s) are available")
        target = records[-steps]
        if self.pointer.exists():
            current = json.loads(self.pointer.read_text(encoding="utf-8"))
            self._append_history(current, action="rolled_back_from")
        payload = {
            "model_name": self.settings.model_name,
            "version": str(target["version"]),
            "version_path": str(target["version_path"]),
            "promoted_at": datetime.now(UTC).isoformat(),
            "reason": f"rollback_{steps}_step(s)",
        }
        self._atomic_json(self.pointer, payload)
        return Path(payload["version_path"])

    def list_versions(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        production_version = None
        if self.pointer.exists():
            try:
                production_version = json.loads(self.pointer.read_text(encoding="utf-8")).get("version")
            except json.JSONDecodeError:
                pass
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not directory.is_dir() or not (directory / "report.json").exists():
                continue
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            output.append({
                "version": directory.name,
                "version_path": str(directory),
                "production": directory.name == production_version,
                "project_version": report.get("project_version"),
                "quality_gate_passed": bool(report.get("quality_gate", {}).get("passed", False)),
                "metrics": self._json_safe_metrics(self._metrics(report)),
            })
        return output

    def latest_candidate_dir(self) -> Path:
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError("No locally registered model versions exist")
        return Path(versions[0]["version_path"])

    def load_version(self, version: str) -> tuple[Any, dict[str, Any], Path]:
        version_dir = self.root / str(version)
        if not version_dir.exists():
            raise FileNotFoundError(f"Unknown model version: {version}")
        model = joblib.load(version_dir / "model.joblib")
        report = json.loads((version_dir / "report.json").read_text(encoding="utf-8"))
        return model, report, version_dir

    def production_dir(self) -> Path:
        if not self.pointer.exists():
            raise FileNotFoundError(
                "No production model is registered. Train a model that passes the quality and champion gates."
            )
        payload = json.loads(self.pointer.read_text(encoding="utf-8"))
        path = Path(payload["version_path"])
        if not path.exists():
            raise FileNotFoundError(f"Production model directory is missing: {path}")
        return path

    def load_production(self) -> tuple[Any, dict[str, Any], Path]:
        version_dir = self.production_dir()
        model = joblib.load(version_dir / "model.joblib")
        report = json.loads((version_dir / "report.json").read_text(encoding="utf-8"))
        return model, report, version_dir

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
        ) as tmp:
            json.dump(payload, tmp, indent=2, allow_nan=False)
            tmp_path = Path(tmp.name)
        try:
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
