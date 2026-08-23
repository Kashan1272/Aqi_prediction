from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings


class LocalStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.hourly_dir.mkdir(parents=True, exist_ok=True)
        self.settings.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.settings.provider_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)

    def city_path(self, city: str) -> Path:
        parquet = self.settings.hourly_dir / f"city={city}.parquet"
        csv = self.settings.hourly_dir / f"city={city}.csv.gz"
        if parquet.exists():
            return parquet
        if csv.exists():
            return csv
        return parquet

    def read_city(
        self,
        city: str,
        *,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        path = self.city_path(city)
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, compression="gzip", low_memory=False)
        if columns is not None:
            requested = list(dict.fromkeys(columns))
            frame = frame[[column for column in requested if column in frame.columns]]
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame.sort_values("timestamp").reset_index(drop=True) if "timestamp" in frame else frame

    def write_city(self, city: str, frame: pd.DataFrame, *, merge: bool = True) -> Path:
        if frame.empty:
            raise ValueError(f"Cannot save an empty hourly frame for {city}")
        path = self.city_path(city)
        result = frame.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result = result.dropna(subset=["timestamp"])
        if merge and path.exists():
            old = self.read_city(city)
            result = pd.concat([old, result], ignore_index=True, copy=False, sort=False)
        result = (
            result.drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        try:
            self._atomic_parquet(path, result)
            csv_path = self.settings.hourly_dir / f"city={city}.csv.gz"
            csv_path.unlink(missing_ok=True)
            return path
        except (ImportError, ModuleNotFoundError, ValueError):
            csv_path = self.settings.hourly_dir / f"city={city}.csv.gz"
            self._atomic_csv(csv_path, result)
            path.unlink(missing_ok=True)
            return csv_path


    def provider_snapshot_path(self, city: str) -> Path:
        parquet = self.settings.provider_snapshots_dir / f"city={city}.parquet"
        csv = self.settings.provider_snapshots_dir / f"city={city}.csv.gz"
        if parquet.exists():
            return parquet
        if csv.exists():
            return csv
        return parquet

    def read_provider_snapshots(self, city: str) -> pd.DataFrame:
        path = self.provider_snapshot_path(city)
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, compression="gzip", low_memory=False)
        for column in ("issue_date", "target_date", "collected_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
                if column in {"issue_date", "target_date"}:
                    frame[column] = frame[column].dt.tz_localize(None).dt.normalize()
        return frame.sort_values(["issue_date", "horizon_day"]).reset_index(drop=True)

    def write_provider_snapshots(self, city: str, frame: pd.DataFrame) -> Path:
        if frame.empty:
            raise ValueError(f"Cannot save an empty provider snapshot for {city}")
        path = self.provider_snapshot_path(city)
        result = frame.copy()
        for column in ("issue_date", "target_date"):
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
        result["collected_at"] = pd.to_datetime(
            result.get("collected_at"), utc=True, errors="coerce"
        )
        result = result.dropna(subset=["issue_date", "horizon_day"])
        if path.exists():
            old = self.read_provider_snapshots(city)
            result = pd.concat([old, result], ignore_index=True, copy=False, sort=False)
        result = (
            result.sort_values("collected_at")
            .drop_duplicates(["issue_date", "horizon_day"], keep="last")
            .sort_values(["issue_date", "horizon_day"])
            .reset_index(drop=True)
        )
        try:
            self._atomic_parquet(path, result)
            csv_path = self.settings.provider_snapshots_dir / f"city={city}.csv.gz"
            csv_path.unlink(missing_ok=True)
            return path
        except (ImportError, ModuleNotFoundError, ValueError):
            csv_path = self.settings.provider_snapshots_dir / f"city={city}.csv.gz"
            self._atomic_csv(csv_path, result)
            path.unlink(missing_ok=True)
            return csv_path

    def save_prediction(self, city: str, payload: dict[str, Any]) -> Path:
        path = self.settings.predictions_dir / f"{city}.json"
        self._atomic_json(path, payload)
        return path

    def load_prediction(self, city: str) -> dict[str, Any]:
        path = self.settings.predictions_dir / f"{city}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_report(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.settings.reports_dir / name
        self._atomic_json(path, payload)
        return path

    @staticmethod
    def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", dir=path.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            frame.to_parquet(tmp_path, index=False, compression="zstd")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


    @staticmethod
    def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".csv.gz", dir=path.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            frame.to_csv(tmp_path, index=False, compression="gzip")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert pandas/NumPy values into strict, portable JSON values.

        API diagnostics can legitimately contain missing values. Python's JSON
        encoder rejects NaN and infinity when ``allow_nan=False``. Keep strict
        JSON output by converting all non-finite and missing values to ``null``
        while preserving useful scalar, timestamp, mapping, and sequence data.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(key): LocalStore._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [LocalStore._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return [LocalStore._json_safe(item) for item in value.tolist()]
        if isinstance(value, (pd.Timestamp,)):
            return None if pd.isna(value) else value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
        return value

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = LocalStore._json_safe(payload)
        with tempfile.NamedTemporaryFile(
            suffix=".json", dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as tmp:
            json.dump(safe_payload, tmp, indent=2, default=str, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        try:
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
