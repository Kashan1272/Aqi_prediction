from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings

LOGGER = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not name:
        raise ValueError("Hopsworks entity name cannot be empty")
    return name[:63]


def _login(settings: Settings):
    try:
        import hopsworks
    except ImportError as exc:
        raise RuntimeError(
            "Install Hopsworks support with: python -m pip install -r requirements-hopsworks.txt"
        ) from exc
    if settings.running_in_hopsworks and not settings.hopsworks_api_key:
        return hopsworks.login()
    if not settings.hopsworks_project or not settings.hopsworks_api_key:
        raise ValueError("HOPSWORKS_PROJECT and HOPSWORKS_API_KEY are required")
    kwargs: dict[str, Any] = {
        "project": settings.hopsworks_project,
        "api_key_value": settings.hopsworks_api_key,
        "engine": "python",
    }
    if settings.hopsworks_host:
        kwargs["host"] = (
            settings.hopsworks_host
            .removeprefix("https://")
            .removeprefix("http://")
            .rstrip("/")
        )
        kwargs["port"] = settings.hopsworks_port
    return hopsworks.login(**kwargs)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result:
        if "timestamp" in column or column.endswith("_date") or column == "ingested_at":
            parsed = pd.to_datetime(result[column], utc=True, errors="coerce")
            if parsed.notna().any():
                result[column] = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        if result[column].dtype == "object":
            result[column] = result[column].map(
                lambda value: str(value) if isinstance(value, (dict, list, tuple, set)) else value
            )
    return result


class HopsworksAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.project = _login(settings)
        self.feature_store = self.project.get_feature_store()
        self.model_registry = self.project.get_model_registry()
        self.version = settings.hopsworks_feature_group_version

    def upsert(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        primary_key: list[str],
        event_time: str | None = None,
    ) -> str:
        if frame.empty:
            raise ValueError(f"Cannot upload empty feature group {name}")

        prepared = _prepare(frame)

        kwargs: dict[str, Any] = {
            "name": _safe_name(name),
            "version": self.version,
            "description": f"Pearls AQI Predictor v6: {name}",
            "primary_key": [
                column for column in primary_key
                if column in prepared
            ],
            "online_enabled": self.settings.hopsworks_online_enabled,
            "time_travel_format": "HUDI",
            "statistics_config": False,
        }

        if event_time and event_time in prepared:
            kwargs["event_time"] = event_time

        group = self.feature_store.get_or_create_feature_group(**kwargs)

        # Existing groups may still have statistics enabled from their
        # original creation, so persistently disable them.
        if getattr(group, "id", None) is not None:
            group.statistics_config = {
                "enabled": False,
                "histograms": False,
                "correlations": False,
                "exact_uniqueness": False,
            }
            group.update_statistics_config()

        try:
            group.insert(prepared, wait=True)
        except TypeError:
            group.insert(
                prepared,
                write_options={"wait_for_job": True},
            )

        return (
            f"hopsworks://{self.project.name}/"
            f"{_safe_name(name)}/v{self.version}"
        )


    def read_group(self, name: str) -> pd.DataFrame:
        group = self.feature_store.get_feature_group(
            _safe_name(name),
            version=self.version,
        )
        try:
            frame = group.select_all().read()
        except Exception:
            frame = group.read()
        for column in frame.columns:
            if "timestamp" in column or column.endswith("_date") or column == "ingested_at":
                parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
                if parsed.notna().any():
                    frame[column] = parsed
        return frame

    def upload_model(
        self,
        model_directory: Path,
        metrics: dict[str, float],
        *,
        model_name: str | None = None,
        tags: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> str:
        registry_name = _safe_name(model_name or self.settings.model_name)
        model = self.model_registry.sklearn.create_model(
            name=registry_name,
            metrics={key: float(value) for key, value in metrics.items()},
            description=description or (
                "Pearls AQI Predictor champion/challenger three-day ensemble. "
                "The artifact includes its local registry manifest and evaluation report."
            ),
        )
        saved = model.save(
            str(model_directory),
            upload_configuration={
                "chunk_size": 10 * 1024 * 1024,
                "simultaneous_uploads": 1,
                "max_chunk_retries": 5,
            },
        )
        handle = saved if saved is not None else model
        for name, value in (tags or {}).items():
            try:
                handle.set_tag(str(name), value)
            except Exception as exc:
                LOGGER.warning("Could not set Hopsworks model tag %s: %s", name, exc)
        return str(getattr(handle, "version", getattr(model, "version", "unknown")))

    def download_latest_model(self, destination: Path, *, model_name: str | None = None) -> Path:
        models = self.model_registry.get_models(model_name or self.settings.model_name)
        if not models:
            raise FileNotFoundError(
                f"No Hopsworks model named {model_name or self.settings.model_name}"
            )
        model = max(models, key=lambda item: int(getattr(item, "version", 0)))
        downloaded = Path(model.download())
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(downloaded, destination)
        return destination
