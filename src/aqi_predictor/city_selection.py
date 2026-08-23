from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import Settings
from .features import TARGET_COLUMNS
from .models import (
    build_candidate_models,
    iter_rolling_folds,
    regression_metrics,
    seasonal_baseline,
)

LOGGER = logging.getLogger(__name__)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class CitySelectionResult:
    candidate_cities: list[str]
    selected_cities: list[str]
    mandatory_cities: list[str]
    rejected_cities: list[str]
    city_scores: dict[str, dict[str, Any]]
    methodology: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "candidate_cities": self.candidate_cities,
            "selected_cities": self.selected_cities,
            "mandatory_cities": self.mandatory_cities,
            "rejected_cities": self.rejected_cities,
            "city_scores": self.city_scores,
            "methodology": self.methodology,
            "test_data_used_for_selection": False,
        }


def _selector_candidates(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    settings: Settings,
    horizon_day: int,
) -> dict[str, Any]:
    candidates = build_candidate_models(
        numeric_features,
        categorical_features,
        random_seed=settings.random_seed + 7000,
        n_jobs=settings.model_n_jobs,
        horizon_day=horizon_day,
        target="mean",
    )
    selected = {
        name: model
        for name, model in candidates.items()
        if name in {"hist_gradient", "extra_trees"}
    }
    # The selector is a development-only ranking stage, not the final model.
    # Smaller candidate sizes keep it fast and memory-safe.
    if "hist_gradient" in selected:
        selected["hist_gradient"].set_params(model__max_iter=180)
    if "extra_trees" in selected:
        selected["extra_trees"].set_params(model__n_estimators=90)
    return selected


def _safe_metric_payload(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    metrics = regression_metrics(actual, prediction)
    return metrics.to_dict()


def select_training_cities(
    development: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    candidate_keys: list[str],
    settings: Settings,
) -> CitySelectionResult:
    """Select a compact city profile using development OOF predictions only.

    The final chronological test partition must never be supplied to this
    function. Karachi and Multan can be mandatory while the remaining places
    are selected from stable OOF performance, baseline gain, and bias.
    """
    candidate_keys = list(dict.fromkeys(str(key) for key in candidate_keys))
    mandatory = [
        key for key in settings.city_selection_mandatory
        if key in candidate_keys
    ]
    missing_mandatory = sorted(set(settings.city_selection_mandatory) - set(candidate_keys))
    if missing_mandatory:
        raise ValueError(
            "Mandatory training cities are not in the candidate pool: "
            + ", ".join(missing_mandatory)
        )

    target_count = min(
        len(candidate_keys),
        max(len(mandatory), settings.city_selection_target_count),
    )
    if not settings.city_selection_enabled or target_count >= len(candidate_keys):
        scores = {
            key: {
                "selected": True,
                "mandatory": key in mandatory,
                "selection_score": None,
                "reason": "selection disabled or target count includes all candidates",
            }
            for key in candidate_keys
        }
        return CitySelectionResult(
            candidate_cities=candidate_keys,
            selected_cities=candidate_keys,
            mandatory_cities=mandatory,
            rejected_cities=[],
            city_scores=scores,
            methodology="all configured cities retained",
        )

    oof_parts: list[pd.DataFrame] = []
    for horizon_day in range(1, settings.forecast_days + 1):
        horizon = development[
            (development["horizon_day"] == horizon_day)
            & development["city"].astype(str).isin(candidate_keys)
        ].copy()
        if horizon.empty:
            continue
        selector_models = _selector_candidates(
            numeric_features,
            categorical_features,
            settings=settings,
            horizon_day=horizon_day,
        )
        for fold_index, (fold_train, fold_validation) in enumerate(
            iter_rolling_folds(horizon, folds=settings.city_selection_cv_folds),
            start=1,
        ):
            predictions: list[np.ndarray] = []
            completed_models: list[str] = []
            for name, candidate in selector_models.items():
                fitted = None
                try:
                    fitted = clone(candidate)
                    fitted.fit(fold_train, fold_train[TARGET_COLUMNS[0]])
                    predictions.append(
                        np.clip(
                            np.asarray(fitted.predict(fold_validation), dtype=np.float32),
                            0,
                            500,
                        )
                    )
                    completed_models.append(name)
                except (MemoryError, ValueError) as exc:
                    LOGGER.warning(
                        "City selector candidate %s failed on horizon %s fold %s: %s",
                        name,
                        horizon_day,
                        fold_index,
                        exc,
                    )
                finally:
                    if fitted is not None:
                        del fitted
                    gc.collect()
            if not predictions:
                raise RuntimeError(
                    f"No city-selector model completed horizon {horizon_day}, fold {fold_index}"
                )
            model_prediction = np.mean(np.vstack(predictions), axis=0)
            actual = pd.to_numeric(
                fold_validation[TARGET_COLUMNS[0]], errors="coerce"
            ).to_numpy(dtype=np.float32)
            baseline = seasonal_baseline(
                fold_validation, TARGET_COLUMNS[0]
            ).astype(np.float32, copy=False)
            part = fold_validation[["city", "issue_date"]].copy()
            part["horizon_day"] = horizon_day
            part["selector_fold"] = fold_index
            part["actual"] = actual
            part["prediction"] = model_prediction
            part["baseline"] = baseline
            part["selector_models"] = "+".join(completed_models)
            oof_parts.append(part)
            del fold_train, fold_validation, predictions
            gc.collect()

    if not oof_parts:
        raise ValueError("City selection produced no development out-of-fold predictions")
    scored = pd.concat(oof_parts, ignore_index=True, copy=False)

    city_scores: dict[str, dict[str, Any]] = {}
    for city in candidate_keys:
        subset = scored[scored["city"].astype(str) == city]
        actual = subset["actual"].to_numpy(dtype=float)
        prediction = subset["prediction"].to_numpy(dtype=float)
        baseline = subset["baseline"].to_numpy(dtype=float)
        model_metrics = regression_metrics(actual, prediction)
        baseline_metrics = regression_metrics(actual, baseline)
        rmse_gain = (
            (baseline_metrics.rmse - model_metrics.rmse)
            / max(baseline_metrics.rmse, 1e-9)
        )
        horizon_details: dict[str, Any] = {}
        horizon_gains: list[float] = []
        horizon_r2s: list[float] = []
        for horizon_day, horizon_subset in subset.groupby("horizon_day", observed=True):
            h_model = regression_metrics(
                horizon_subset["actual"], horizon_subset["prediction"]
            )
            h_baseline = regression_metrics(
                horizon_subset["actual"], horizon_subset["baseline"]
            )
            gain = (
                (h_baseline.rmse - h_model.rmse)
                / max(h_baseline.rmse, 1e-9)
            )
            horizon_gains.append(float(gain))
            if np.isfinite(h_model.r2):
                horizon_r2s.append(float(h_model.r2))
            horizon_details[f"day{int(horizon_day)}"] = {
                "model": h_model.to_dict(),
                "baseline": h_baseline.to_dict(),
                "rmse_gain": float(gain),
            }
        actual_std = float(np.nanstd(actual))
        normalized_bias = abs(float(model_metrics.bias)) / max(actual_std, 1.0)
        minimum_horizon_gain = min(horizon_gains) if horizon_gains else -1.0
        median_horizon_r2 = float(np.median(horizon_r2s)) if horizon_r2s else -1.0
        selection_score = (
            0.50 * float(np.clip(rmse_gain, -1.0, 1.0))
            + 0.30 * float(np.clip(model_metrics.r2, -0.5, 1.0))
            + 0.15 * float(np.clip(minimum_horizon_gain, -1.0, 1.0))
            + 0.05 * float(np.clip(median_horizon_r2, -0.5, 1.0))
            - 0.05 * float(np.clip(normalized_bias, 0.0, 2.0))
        )
        eligible = bool(
            model_metrics.sample_count >= settings.city_selection_min_oof_rows
            and np.isfinite(selection_score)
            and rmse_gain > settings.city_selection_min_baseline_gain
        )
        city_scores[city] = {
            "mandatory": city in mandatory,
            "eligible": eligible,
            "selection_score": float(selection_score),
            "rmse_gain_vs_baseline": float(rmse_gain),
            "minimum_horizon_gain": float(minimum_horizon_gain),
            "median_horizon_r2": float(median_horizon_r2),
            "normalized_abs_bias": float(normalized_bias),
            "model": model_metrics.to_dict(),
            "baseline": baseline_metrics.to_dict(),
            "by_day": horizon_details,
            "oof_rows": int(len(subset)),
        }

    ranked = sorted(
        [key for key in candidate_keys if key not in mandatory],
        key=lambda key: (
            bool(city_scores[key]["eligible"]),
            float(city_scores[key]["selection_score"]),
        ),
        reverse=True,
    )
    selected = list(mandatory)
    selected.extend(ranked[: max(0, target_count - len(selected))])
    # Preserve the candidate-catalogue ordering for stable dashboards/reports.
    selected_set = set(selected)
    selected = [key for key in candidate_keys if key in selected_set]
    rejected = [key for key in candidate_keys if key not in selected_set]
    for key, payload in city_scores.items():
        payload["selected"] = key in selected_set
        payload["selection_reason"] = (
            "mandatory" if key in mandatory
            else "top development OOF score" if key in selected_set
            else "lower development OOF score"
        )

    return CitySelectionResult(
        candidate_cities=candidate_keys,
        selected_cities=selected,
        mandatory_cities=mandatory,
        rejected_cities=rejected,
        city_scores=city_scores,
        methodology=(
            "Development-only chronological OOF ranking using an Extra Trees and "
            "Histogram Gradient ensemble. Final test rows were excluded."
        ),
    )


def save_selected_city_profile(
    settings: Settings,
    result: CitySelectionResult,
) -> Path:
    path = settings.selected_city_profile_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_json(result.to_dict()), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def load_selected_city_keys(settings: Settings) -> list[str]:
    path = settings.selected_city_profile_path
    if not path.exists():
        return list(settings.cities)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = [
            str(key) for key in payload.get("selected_cities", [])
            if str(key) in settings.cities
        ]
        return selected or list(settings.cities)
    except (OSError, ValueError, TypeError):
        return list(settings.cities)
