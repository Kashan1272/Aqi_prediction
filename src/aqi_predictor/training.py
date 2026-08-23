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
from .city_selection import select_training_cities, save_selected_city_profile
from .explainability import explain_pipeline
from .features import (
    TARGET_COLUMNS,
    SENSOR_TARGET_COLUMNS,
    build_national_training_frame,
    feature_columns,
    add_provider_snapshot_features,
)
from .models import (
    DailyAQIModelBundle,
    OOFStackingRegressor,
    build_candidate_models,
    chronological_partitions,
    fit_convex_blend,
    regression_metrics,
    weighted_rmse,
    iter_rolling_folds,
    rolling_folds,
    seasonal_baseline,
    shrunken_group_offsets,
)
from .matrix_validation import audit_training_matrix
from .matrix_manifest import build_matrix_manifest
from .local_experts import fit_hierarchical_calibration, fit_local_experts
from .registry import LocalModelRegistry
from .storage import LocalStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    version_path: Path
    promoted: bool
    quality_gate_passed: bool
    report_path: Path
    test_metrics: dict[str, Any]


def _candidate_cv(
    model: Any,
    frame: pd.DataFrame,
    target_column: str,
    *,
    folds: int = 3,
) -> tuple[dict[str, Any], list[float]]:
    fold_reports: list[dict[str, Any]] = []
    residuals: list[float] = []
    for fold_index, (train, validation) in enumerate(rolling_folds(frame, folds=folds), start=1):
        fitted = clone(model)
        fitted.fit(train, train[target_column])
        prediction = np.clip(fitted.predict(validation), 0, 500)
        metrics = regression_metrics(validation[target_column], prediction)
        baseline_prediction = seasonal_baseline(validation, target_column)
        baseline = regression_metrics(validation[target_column], baseline_prediction)
        residuals.extend(
            (pd.to_numeric(validation[target_column], errors="coerce").to_numpy(dtype=float) - prediction).tolist()
        )
        fold_reports.append({
            "fold": fold_index,
            "metrics": metrics.to_dict(),
            "baseline": baseline.to_dict(),
            "beats_baseline": bool(metrics.rmse < baseline.rmse),
            "train_rows": len(train),
            "validation_rows": len(validation),
        })
    valid = [item for item in fold_reports if np.isfinite(item["metrics"]["rmse"])]
    if not valid:
        return {
            "folds": fold_reports,
            "mean_rmse": np.nan,
            "mean_r2": np.nan,
            "rmse_std": np.nan,
            "beats_baseline_ratio": 0.0,
            "selection_score": np.inf,
        }, residuals
    rmses = np.asarray([item["metrics"]["rmse"] for item in valid], dtype=float)
    r2s = np.asarray([item["metrics"]["r2"] for item in valid], dtype=float)
    beat_ratio = float(np.mean([item["beats_baseline"] for item in valid]))
    mean_rmse = float(np.mean(rmses))
    mean_r2 = float(np.mean(r2s))
    stability = float(np.std(rmses))
    penalty = max(0.0, 0.55 - mean_r2) * 25.0 + (1.0 - beat_ratio) * 8.0
    return {
        "folds": fold_reports,
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "rmse_std": stability,
        "beats_baseline_ratio": beat_ratio,
        "selection_score": mean_rmse + 0.35 * stability + penalty,
    }, residuals



def _aggregate_fold_reports(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in fold_reports if np.isfinite(item["metrics"]["rmse"])]
    if not valid:
        return {
            "folds": fold_reports,
            "mean_rmse": np.nan,
            "mean_r2": np.nan,
            "rmse_std": np.nan,
            "beats_baseline_ratio": 0.0,
            "selection_score": np.inf,
        }
    rmses = np.asarray([item["metrics"]["rmse"] for item in valid], dtype=float)
    r2s = np.asarray([item["metrics"]["r2"] for item in valid], dtype=float)
    beat_ratio = float(np.mean([item["beats_baseline"] for item in valid]))
    mean_rmse = float(np.mean(rmses))
    mean_r2 = float(np.mean(r2s))
    stability = float(np.std(rmses))
    penalty = max(0.0, 0.55 - mean_r2) * 25.0 + (1.0 - beat_ratio) * 8.0
    return {
        "folds": fold_reports,
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "rmse_std": stability,
        "beats_baseline_ratio": beat_ratio,
        "selection_score": mean_rmse + 0.35 * stability + penalty,
    }



def _is_memory_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or "unable to allocate" in message or "out of memory" in message


def _normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        str(name): max(0.0, float(value))
        for name, value in weights.items()
        if np.isfinite(value) and float(value) > 1e-12
    }
    total = float(sum(cleaned.values()))
    if total <= 0:
        return {"seasonal_baseline": 1.0}
    return {name: value / total for name, value in cleaned.items()}


def _fit_oof_stacking_model(
    candidates: dict[str, Any],
    frame: pd.DataFrame,
    target_column: str,
    *,
    settings: Settings,
    folds: int,
) -> tuple[OOFStackingRegressor, dict[str, Any], np.ndarray]:
    """Fit a leakage-safe convex ensemble with bounded peak memory.

    Folds are generated one at a time, candidate estimators are explicitly
    released after prediction, and only materially weighted base learners are
    retained in the final serializable ensemble.
    """
    oof_blocks: list[pd.DataFrame] = []
    baseline_reports: list[dict[str, Any]] = []
    for fold_index, (fold_train, fold_validation) in enumerate(
        iter_rolling_folds(frame, folds=folds),
        start=1,
    ):
        block = fold_validation[["city", "province", "issue_date"]].copy()
        block["oof_fold"] = fold_index
        block["actual"] = pd.to_numeric(
            fold_validation[target_column], errors="coerce"
        ).to_numpy(dtype=np.float32)
        baseline_prediction = seasonal_baseline(
            fold_validation, target_column
        ).astype(np.float32, copy=False)
        block["seasonal_baseline"] = baseline_prediction
        baseline_metrics = regression_metrics(block["actual"], baseline_prediction)
        baseline_reports.append({
            "fold": fold_index,
            "metrics": baseline_metrics.to_dict(),
            "train_rows": len(fold_train),
            "validation_rows": len(fold_validation),
        })
        oof_blocks.append(block)
        del fold_train, fold_validation, baseline_prediction
        gc.collect()

    if len(oof_blocks) < 2:
        raise ValueError("At least two rolling folds are required for OOF stacking")

    fold_reports: dict[str, list[dict[str, Any]]] = {}
    candidate_failures: dict[str, dict[str, Any]] = {}
    successful_candidates: dict[str, Any] = {}

    # Candidate-major fitting prevents several fold estimators and transformed
    # matrices from coexisting in memory.
    for name, candidate in candidates.items():
        reports: list[dict[str, Any]] = []
        predictions_by_fold: list[np.ndarray] = []
        failure: BaseException | None = None

        for fold_index, (fold_train, fold_validation) in enumerate(
            iter_rolling_folds(frame, folds=folds),
            start=1,
        ):
            fitted = None
            try:
                fitted = clone(candidate)
                fitted.fit(fold_train, fold_train[target_column])
                prediction = np.clip(
                    np.asarray(fitted.predict(fold_validation), dtype=np.float32),
                    0,
                    500,
                )
                actual = pd.to_numeric(
                    fold_validation[target_column], errors="coerce"
                ).to_numpy(dtype=np.float32)
                metrics = regression_metrics(actual, prediction)
                baseline_metrics = baseline_reports[fold_index - 1]["metrics"]
                reports.append({
                    "fold": fold_index,
                    "metrics": metrics.to_dict(),
                    "baseline": baseline_metrics,
                    "beats_baseline": bool(
                        metrics.rmse < float(baseline_metrics["rmse"])
                    ),
                    "train_rows": len(fold_train),
                    "validation_rows": len(fold_validation),
                })
                predictions_by_fold.append(prediction)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failure = exc
                if not settings.stacking_memory_recovery and _is_memory_failure(exc):
                    raise
                LOGGER.warning(
                    "Skipping stacking candidate %s for %s after fold %s failed: %s",
                    name,
                    target_column,
                    fold_index,
                    exc,
                )
                break
            finally:
                if fitted is not None:
                    del fitted
                del fold_train, fold_validation
                gc.collect()

        fold_reports[name] = reports
        if failure is not None or len(predictions_by_fold) != len(oof_blocks):
            candidate_failures[name] = {
                "type": type(failure).__name__ if failure is not None else "IncompleteOOF",
                "memory_related": bool(failure is not None and _is_memory_failure(failure)),
                "error": str(failure) if failure is not None else "Candidate did not complete every OOF fold",
                "completed_folds": len(predictions_by_fold),
                "required_folds": len(oof_blocks),
            }
            del predictions_by_fold
            gc.collect()
            continue

        successful_candidates[name] = candidate
        for block, prediction in zip(oof_blocks, predictions_by_fold, strict=True):
            block[name] = prediction
        del predictions_by_fold
        gc.collect()

    if not successful_candidates:
        LOGGER.warning(
            "All ML candidates failed for %s; using the seasonal baseline only",
            target_column,
        )

    oof = pd.concat(oof_blocks, ignore_index=True, copy=False)
    del oof_blocks
    gc.collect()

    candidate_reports = {
        name: _aggregate_fold_reports(reports)
        for name, reports in fold_reports.items()
    }
    component_names = list(successful_candidates) + ["seasonal_baseline"]
    matrix = oof[component_names].to_numpy(dtype=np.float32)
    actual = oof["actual"].to_numpy(dtype=np.float32)
    horizon_values = pd.to_numeric(frame.get("horizon_day"), errors="coerce").dropna()
    horizon_day = int(horizon_values.iloc[0]) if len(horizon_values) else 1

    city_sample_weight = np.ones(len(oof), dtype=np.float32)
    if settings.stacking_city_balanced:
        city_counts = oof["city"].astype(str).value_counts()
        city_sample_weight *= oof["city"].astype(str).map(
            {city: len(oof) / (len(city_counts) * count) for city, count in city_counts.items()}
        ).to_numpy(dtype=np.float32)
    # Give the most recent development fold modestly greater influence when
    # learning stack weights. This targets temporal drift without touching the
    # final chronological test set or changing the quality gate.
    recent_fold_weight = (
        settings.day3_recent_fold_weight
        if horizon_day == 3
        else settings.stacking_recent_fold_weight
    )
    if "oof_fold" in oof and recent_fold_weight > 1.0:
        fold_values = pd.to_numeric(
            oof["oof_fold"], errors="coerce"
        ).fillna(1).to_numpy(dtype=np.float32)
        max_fold = float(np.nanmax(fold_values)) if len(fold_values) else 1.0
        if max_fold > 1.0:
            recency = 1.0 + (fold_values - 1.0) / (max_fold - 1.0) * (
                recent_fold_weight - 1.0
            )
            city_sample_weight *= recency.astype(np.float32)
    if horizon_day == 3 and settings.day3_extreme_sample_weight > 1.0:
        extreme = actual >= float(settings.day3_extreme_threshold)
        city_sample_weight[extreme] *= float(settings.day3_extreme_sample_weight)
    city_sample_weight *= len(city_sample_weight) / max(
        float(city_sample_weight.sum()), 1e-9
    )

    component_rmse: list[float] = []
    base_metrics: dict[str, dict[str, Any]] = {}
    for name in component_names:
        component_prediction = oof[name].to_numpy(dtype=np.float32)
        metric = regression_metrics(actual, component_prediction)
        balanced_rmse = weighted_rmse(actual, component_prediction, city_sample_weight)
        base_metrics[name] = {
            **metric.to_dict(),
            "city_balanced_rmse": float(balanced_rmse),
        }
        component_rmse.append(max(balanced_rmse, 1e-6))

    inverse_error = 1.0 / np.square(np.asarray(component_rmse, dtype=float))
    prior = inverse_error / inverse_error.sum()
    stacking_l2 = (
        settings.day3_stacking_l2_regularization
        if horizon_day == 3
        else settings.stacking_l2_regularization
    )
    full_weights, full_intercept, full_optimizer = fit_convex_blend(
        matrix,
        actual,
        prior=prior,
        sample_weight=city_sample_weight,
        l2_regularization=stacking_l2,
        max_iterations=settings.stacking_max_iterations,
    )
    full_weight_map = {
        name: float(weight)
        for name, weight in zip(component_names, full_weights, strict=True)
    }

    best_name = min(component_names, key=lambda name: base_metrics[name]["city_balanced_rmse"])
    ranked_base = sorted(
        successful_candidates,
        key=lambda name: full_weight_map.get(name, 0.0),
        reverse=True,
    )
    selected_base: list[str] = []
    if best_name in successful_candidates:
        selected_base.append(best_name)
    for name in ranked_base:
        if name in selected_base:
            continue
        if (
            full_weight_map.get(name, 0.0) >= settings.stacking_min_component_weight
            or not selected_base
        ):
            selected_base.append(name)
        if len(selected_base) >= settings.stacking_max_base_models:
            break

    # The baseline costs essentially no model memory, so retain it as a
    # component while capping only fitted base learners.
    sparse_components = selected_base + ["seasonal_baseline"]
    sparse_matrix = oof[sparse_components].to_numpy(dtype=np.float32)
    sparse_prior = np.asarray(
        [max(full_weight_map.get(name, 0.0), 1e-6) for name in sparse_components],
        dtype=float,
    )
    sparse_prior /= sparse_prior.sum()
    weights, intercept, optimizer = fit_convex_blend(
        sparse_matrix,
        actual,
        prior=sparse_prior,
        sample_weight=city_sample_weight,
        l2_regularization=stacking_l2,
        max_iterations=settings.stacking_max_iterations,
    )
    component_names = sparse_components
    matrix = sparse_matrix

    raw_prediction = matrix @ weights + intercept
    # Stack selection is based on raw OOF predictions. Residual calibration is
    # selected separately on a temporal OOF holdout below, so it cannot make a
    # weak stack appear better through in-sample group offsets.
    stacked_prediction = np.clip(raw_prediction, 0, 500)
    stacked_metrics = regression_metrics(actual, stacked_prediction)
    stacked_city_balanced_rmse = weighted_rmse(
        actual, stacked_prediction, city_sample_weight
    )

    best_metrics = base_metrics[best_name]
    relative_gain = (
        float(best_metrics["city_balanced_rmse"]) - stacked_city_balanced_rmse
    ) / max(float(best_metrics["city_balanced_rmse"]), 1e-9)

    strategy = "memory_safe_oof_stack"
    if relative_gain < settings.stacking_min_improvement:
        strategy = "guarded_best_component"
        component_names = [best_name]
        weights = np.ones(1, dtype=float)
        intercept = 0.0
        matrix = oof[[best_name]].to_numpy(dtype=np.float32)
        raw_prediction = matrix[:, 0]
        stacked_prediction = np.clip(raw_prediction, 0, 500)
        stacked_metrics = regression_metrics(actual, stacked_prediction)

    preliminary_weight_map = _normalize_weight_map({
        name: float(weight)
        for name, weight in zip(component_names, weights, strict=True)
    })

    # Fit only components with non-zero final weight. This avoids keeping four
    # large forests/boosters for every one of the six targets.
    fitted_models: dict[str, Any] = {}
    final_fit_failures: dict[str, dict[str, Any]] = {}
    active_names = [
        name for name, weight in preliminary_weight_map.items()
        if name != "seasonal_baseline" and weight > 1e-12
    ]
    active_names.sort(
        key=lambda name: preliminary_weight_map.get(name, 0.0),
        reverse=True,
    )
    for name in active_names:
        candidate = successful_candidates.get(name)
        if candidate is None:
            continue
        fitted = None
        try:
            gc.collect()
            fitted = clone(candidate)
            fitted.fit(frame, frame[target_column])
            fitted_models[name] = fitted
            fitted = None
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if not settings.stacking_memory_recovery and _is_memory_failure(exc):
                raise
            LOGGER.warning(
                "Final fit for stacking component %s/%s failed; removing it: %s",
                target_column,
                name,
                exc,
            )
            final_fit_failures[name] = {
                "type": type(exc).__name__,
                "memory_related": _is_memory_failure(exc),
                "error": str(exc),
            }
        finally:
            if fitted is not None:
                del fitted
            gc.collect()

    final_weight_map = {
        name: weight
        for name, weight in preliminary_weight_map.items()
        if name == "seasonal_baseline" or name in fitted_models
    }
    final_weight_map = _normalize_weight_map(final_weight_map)

    # Recalculate the surviving global stack, then add leakage-safe city experts
    # and hierarchical residual calibration learned only from development OOF rows.
    final_raw_prediction = np.full(len(oof), float(intercept), dtype=np.float32)
    for name, weight in final_weight_map.items():
        final_raw_prediction += (
            float(weight) * oof[name].to_numpy(dtype=np.float32)
        )

    local_result = fit_local_experts(
        successful_candidates,
        frame,
        target_column,
        oof,
        final_raw_prediction,
        settings=settings,
        horizon_day=horizon_day,
    )
    final_stacked_prediction, calibration_offsets, calibration_report = (
        fit_hierarchical_calibration(
            oof,
            actual,
            local_result.oof_prediction,
            settings=settings,
            horizon_day=horizon_day,
        )
    )
    stacked_metrics = regression_metrics(actual, final_stacked_prediction)
    stacked_city_balanced_rmse = weighted_rmse(
        actual, final_stacked_prediction, city_sample_weight
    )

    model = OOFStackingRegressor(
        base_models=fitted_models,
        weights=final_weight_map,
        intercept=float(intercept),
        target_column=target_column,
        global_bias_offset=float(
            calibration_offsets.get("global", {}).get("__all__", 0.0)
        ),
        province_offsets=calibration_offsets.get("province", {}),
        city_offsets=calibration_offsets.get("city", {}),
        month_offsets=calibration_offsets.get("month", {}),
        city_month_offsets=calibration_offsets.get("city_month", {}),
        recent_city_offsets=calibration_offsets.get("recent_city", {}),
        local_models=local_result.models,
        local_weights=local_result.weights,
        local_algorithms=local_result.algorithms,
        metadata={
            "strategy": strategy,
            "folds": len(baseline_reports),
            "oof_rows": len(oof),
            "memory_safe": True,
            "city_balanced_oof": settings.stacking_city_balanced,
            "recent_fold_weight": recent_fold_weight,
            "horizon_day": horizon_day,
            "day3_extreme_sample_weight": (
                settings.day3_extreme_sample_weight if horizon_day == 3 else 1.0
            ),
            "retained_base_models": list(fitted_models),
            "local_expert_cities": sorted(local_result.models),
            "hierarchical_calibration": True,
        },
    )
    residuals = actual - final_stacked_prediction
    report = {
        "strategy": strategy,
        "weights": final_weight_map,
        "oof_weights_before_memory_recovery": preliminary_weight_map,
        "full_oof_weights_before_sparsification": full_weight_map,
        "intercept": float(intercept),
        "oof_metrics": stacked_metrics.to_dict(),
        "oof_city_balanced_rmse": float(stacked_city_balanced_rmse),
        "city_balanced_oof": settings.stacking_city_balanced,
        "recent_fold_weight": recent_fold_weight,
        "horizon_day": horizon_day,
        "stacking_l2_regularization": stacking_l2,
        "day3_extreme_sample_weight": (
            settings.day3_extreme_sample_weight if horizon_day == 3 else 1.0
        ),
        "best_single_component": best_name,
        "best_single_metrics": best_metrics,
        "relative_rmse_gain_vs_best": float(
            (float(best_metrics["city_balanced_rmse"]) - stacked_city_balanced_rmse)
            / max(float(best_metrics["city_balanced_rmse"]), 1e-9)
        ),
        "optimizer": optimizer,
        "full_optimizer": full_optimizer,
        "global_bias_offset": float(
            calibration_offsets.get("global", {}).get("__all__", 0.0)
        ),
        "province_offsets": calibration_offsets.get("province", {}),
        "city_offsets": calibration_offsets.get("city", {}),
        "month_offsets": calibration_offsets.get("month", {}),
        "city_month_offsets": calibration_offsets.get("city_month", {}),
        "recent_city_offsets": calibration_offsets.get("recent_city", {}),
        "hierarchical_calibration": calibration_report,
        "local_experts": local_result.report,
        "base_component_metrics": base_metrics,
        "base_candidates": candidate_reports,
        "candidate_failures": candidate_failures,
        "final_fit_failures": final_fit_failures,
        "retained_base_models": list(fitted_models),
        "max_base_models": settings.stacking_max_base_models,
        "min_component_weight": settings.stacking_min_component_weight,
    }
    del oof, matrix, sparse_matrix
    gc.collect()
    return model, report, residuals


def _candidate_cv_plain(
    model: Any,
    frame: pd.DataFrame,
    target_column: str,
    *,
    folds: int = 3,
) -> tuple[dict[str, Any], list[float]]:
    fold_reports: list[dict[str, Any]] = []
    residuals: list[float] = []
    for fold_index, (train, validation) in enumerate(rolling_folds(frame, folds=folds), start=1):
        fitted = clone(model)
        fitted.fit(train, train[target_column])
        prediction = np.asarray(fitted.predict(validation), dtype=float)
        metrics = regression_metrics(validation[target_column], prediction)
        residuals.extend(
            (
                pd.to_numeric(validation[target_column], errors="coerce").to_numpy(dtype=float)
                - prediction
            ).tolist()
        )
        fold_reports.append({
            "fold": fold_index,
            "metrics": metrics.to_dict(),
            "train_rows": len(train),
            "validation_rows": len(validation),
        })
    valid = [item for item in fold_reports if np.isfinite(item["metrics"]["rmse"])]
    if not valid:
        return {
            "folds": fold_reports,
            "mean_rmse": np.nan,
            "mean_r2": np.nan,
            "rmse_std": np.nan,
            "selection_score": np.inf,
        }, residuals
    rmses = np.asarray([item["metrics"]["rmse"] for item in valid], dtype=float)
    r2s = np.asarray([item["metrics"]["r2"] for item in valid], dtype=float)
    mean_rmse = float(np.mean(rmses))
    mean_r2 = float(np.mean(r2s))
    stability = float(np.std(rmses))
    penalty = max(0.0, -0.10 - mean_r2) * 10.0
    return {
        "folds": fold_reports,
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "rmse_std": stability,
        "selection_score": mean_rmse + 0.35 * stability + penalty,
    }, residuals


def _clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _grouped_test_metrics(scored: pd.DataFrame, group_column: str) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key, subset in scored.groupby(group_column, observed=True):
        model = regression_metrics(subset["actual"], subset["prediction"])
        baseline = regression_metrics(subset["actual"], subset["baseline"])
        groups[str(key)] = {
            "model": model.to_dict(),
            "baseline": baseline.to_dict(),
            "beats_baseline": bool(model.rmse < baseline.rmse),
        }
    finite_r2 = [
        item["model"]["r2"] for item in groups.values()
        if np.isfinite(item["model"]["r2"])
    ]
    finite_rmse = [
        item["model"]["rmse"] for item in groups.values()
        if np.isfinite(item["model"]["rmse"])
    ]
    return {
        "groups": groups,
        "macro_r2": float(np.mean(finite_r2)) if finite_r2 else np.nan,
        "macro_rmse": float(np.mean(finite_rmse)) if finite_rmse else np.nan,
        "groups_beating_baseline": int(sum(item["beats_baseline"] for item in groups.values())),
        "group_count": int(len(groups)),
    }


def train_project(
    settings: Settings,
    *,
    city_keys: list[str] | None = None,
    allow_promotion: bool = True,
    quick: bool = False,
) -> TrainingResult:
    store = LocalStore(settings)
    selected_keys = city_keys or list(settings.cities)
    histories = []
    for key in selected_keys:
        frame = store.read_city(key)
        if frame.empty:
            LOGGER.warning("Skipping %s because no hourly history is stored", key)
            continue
        histories.append((settings.city(key), frame))
    training = build_national_training_frame(histories, forecast_days=settings.forecast_days)

    # Optional live-provider snapshots are accumulated by the hourly forecast
    # pipeline. Once their target dates have become historical, they are valid
    # issue-time features and can improve future retraining without leakage.
    snapshot_frames: list[pd.DataFrame] = []
    for key in selected_keys:
        snapshot = store.read_provider_snapshots(key)
        if snapshot.empty:
            continue
        snapshot["city"] = key
        snapshot_frames.append(snapshot)
    provider_snapshot_rows = 0
    if snapshot_frames:
        snapshots = pd.concat(snapshot_frames, ignore_index=True, copy=False, sort=False)
        snapshots["issue_date"] = pd.to_datetime(
            snapshots["issue_date"], errors="coerce"
        ).dt.normalize()
        snapshots["horizon_day"] = pd.to_numeric(
            snapshots["horizon_day"], errors="coerce"
        ).astype("Int64")
        snapshots = snapshots.drop(
            columns=["target_date", "collected_at"], errors="ignore"
        )
        provider_snapshot_rows = len(snapshots)
        training = training.merge(
            snapshots,
            on=["city", "issue_date", "horizon_day"],
            how="left",
            validate="many_to_one",
        )
    training = add_provider_snapshot_features(training)

    all_missing_numeric = {
        column: "all_missing"
        for column in training.columns
        if pd.api.types.is_numeric_dtype(training[column])
        and not pd.to_numeric(training[column], errors="coerce").notna().any()
    }
    numeric_features, categorical_features = feature_columns(training)

    # Hundreds of engineered columns are sufficient at float32 precision for
    # AQI forecasting and halve the resident memory of the national frame.
    downcast_columns = list(dict.fromkeys(
        numeric_features + TARGET_COLUMNS + SENSOR_TARGET_COLUMNS
    ))
    for column in downcast_columns:
        if column in training:
            training[column] = pd.to_numeric(
                training[column], errors="coerce", downcast="float"
            )

    required_future = [
        column for column in numeric_features
        if column.startswith("future_temperature_2m")
        or column.startswith("future_relative_humidity_2m")
        or column.startswith("future_surface_pressure")
        or column.startswith("future_wind_speed_10m")
    ]
    future_coverage = (
        float(training[required_future].notna().mean().mean()) if required_future else 0.0
    )
    if future_coverage < 0.65:
        raise ValueError(
            f"Lead-aligned future-weather coverage is only {future_coverage:.1%}. "
            "Complete the historical backfill with lead weather before training."
        )

    train, validation, test = chronological_partitions(training)
    matrix_audit = audit_training_matrix(
        training,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train=train,
        validation=validation,
        test=test,
        selected_cities=selected_keys,
        settings=settings,
    )
    LocalStore(settings).save_report("training_matrix_audit_candidates_v67.json", matrix_audit)
    if settings.matrix_strict and not matrix_audit["ready_for_training"]:
        raise ValueError(
            "Training matrix validation failed: "
            + "; ".join(matrix_audit["errors"][:5])
        )
    development = pd.concat([train, validation], ignore_index=True, copy=False)

    city_selection_result = select_training_cities(
        development,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        candidate_keys=selected_keys,
        settings=settings,
    )
    selected_training_keys = city_selection_result.selected_cities
    save_selected_city_profile(settings, city_selection_result)
    LocalStore(settings).save_report(
        "city_selection_v67.json", city_selection_result.to_dict()
    )
    if set(selected_training_keys) != set(selected_keys):
        LOGGER.info(
            "Development-only city selection retained %s of %s cities: %s",
            len(selected_training_keys),
            len(selected_keys),
            ", ".join(selected_training_keys),
        )
        training = training[training["city"].astype(str).isin(selected_training_keys)].copy()
        train = train[train["city"].astype(str).isin(selected_training_keys)].copy()
        validation = validation[validation["city"].astype(str).isin(selected_training_keys)].copy()
        test = test[test["city"].astype(str).isin(selected_training_keys)].copy()
        development = development[
            development["city"].astype(str).isin(selected_training_keys)
        ].copy()
        selected_matrix_audit = audit_training_matrix(
            training,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            train=train,
            validation=validation,
            test=test,
            selected_cities=selected_training_keys,
            settings=settings,
        )
        LocalStore(settings).save_report(
            "training_matrix_audit_selected_v67.json", selected_matrix_audit
        )
        if settings.matrix_strict and not selected_matrix_audit["ready_for_training"]:
            raise ValueError(
                "Selected-city training matrix validation failed: "
                + "; ".join(selected_matrix_audit["errors"][:5])
            )
    else:
        selected_matrix_audit = matrix_audit

    matrix_manifest = build_matrix_manifest(
        training,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train=train,
        validation=validation,
        test=test,
        selected_cities=selected_training_keys,
        removed_features=all_missing_numeric,
    )
    LocalStore(settings).save_report(
        "training_matrix_manifest_v69.json", matrix_manifest
    )

    models: dict[str, Any] = {}
    selected_algorithms: dict[str, str] = {}
    intervals: dict[str, dict[str, float]] = {}
    sensor_calibrators: dict[str, Any] = {}
    calibrated_cities: set[str] = set()
    sensor_calibration_report: dict[str, Any] = {}
    explainability_reports: dict[str, Any] = {}
    candidate_reports: dict[str, Any] = {}
    test_reports: dict[str, Any] = {}
    test_scored_rows: list[pd.DataFrame] = []

    for horizon_day in range(1, settings.forecast_days + 1):
        dev_h = development[development["horizon_day"] == horizon_day].copy()
        test_h = test[test["horizon_day"] == horizon_day].copy()
        for target_column in TARGET_COLUMNS:
            target_name = "mean" if target_column.endswith("mean") else "max"
            bundle_key = DailyAQIModelBundle.key(horizon_day, target_name)
            LOGGER.info("Selecting %s model for day %s", target_name, horizon_day)
            candidates = build_candidate_models(
                numeric_features,
                categorical_features,
                random_seed=settings.random_seed,
                n_jobs=settings.model_n_jobs,
                horizon_day=horizon_day,
                target=target_name,
            )
            if quick:
                candidates = {
                    key: value for key, value in candidates.items()
                    if key in {"ridge", "hist_gradient"}
                }
            selected, stacking_report, residuals = _fit_oof_stacking_model(
                candidates,
                dev_h,
                target_column,
                settings=settings,
                folds=2 if quick else settings.stacking_cv_folds,
            )
            prediction = selected.predict(test_h)
            metrics = regression_metrics(test_h[target_column], prediction)
            baseline_prediction = seasonal_baseline(test_h, target_column)
            baseline_metrics = regression_metrics(test_h[target_column], baseline_prediction)
            residuals = np.asarray(residuals, dtype=float)
            residuals = residuals[np.isfinite(residuals)]
            if len(residuals):
                lower_error = float(abs(np.quantile(residuals, 0.10)))
                upper_error = float(abs(np.quantile(residuals, 0.90)))
            else:
                lower_error = upper_error = float(metrics.rmse)
            models[bundle_key] = selected
            selected_algorithms[bundle_key] = stacking_report["strategy"]
            intervals[bundle_key] = {
                "lower_error": lower_error,
                "upper_error": upper_error,
            }
            candidate_reports[bundle_key] = stacking_report
            test_reports[bundle_key] = {
                "metrics": metrics.to_dict(),
                "baseline": baseline_metrics.to_dict(),
                "beats_baseline": bool(metrics.rmse < baseline_metrics.rmse),
                "stacking_strategy": stacking_report["strategy"],
                "weights": stacking_report["weights"],
            }
            if settings.generate_explainability and (not quick) and target_name == "mean" and horizon_day in {1, settings.forecast_days}:
                try:
                    dominant = selected.dominant_base_model
                    explanation_model = selected.base_models.get(dominant) if dominant else None
                    if explanation_model is None:
                        raise ValueError("No explainable base learner was selected")
                    explanation_frame = test_h[numeric_features + categorical_features].copy()
                    report = explain_pipeline(
                        explanation_model,
                        explanation_frame,
                        test_h[target_column],
                        max_rows=settings.explainability_sample_rows,
                        random_seed=settings.random_seed + horizon_day,
                    )
                    report["explained_component"] = dominant
                    report["ensemble_weights"] = stacking_report["weights"]
                    explainability_reports[bundle_key] = report
                except Exception as exc:
                    explainability_reports[bundle_key] = {
                        "method": "failed",
                        "error": str(exc),
                        "features": [],
                    }
            scored = test_h[["issue_date", "target_date", "city", "horizon_day"]].copy()
            scored["target_kind"] = target_name
            scored["actual"] = pd.to_numeric(test_h[target_column], errors="coerce").to_numpy()
            scored["prediction"] = prediction
            scored["baseline"] = baseline_prediction
            test_scored_rows.append(scored)
            # Release candidate templates and temporary prediction arrays while
            # retaining only the compact selected ensemble in `models`.
            del candidates, selected, prediction, baseline_prediction, scored
            gc.collect()


    # Optional OpenAQ calibration layer. It learns the residual between the
    # national Open-Meteo target and real sensor AQI only where sufficient
    # hourly OpenAQ coverage exists. The national model remains available for
    # every city, while calibrated cities receive a sensor-domain correction.
    observed_columns_available = all(
        column in training.columns for column in SENSOR_TARGET_COLUMNS
    )
    if observed_columns_available:
        city_counts = (
            training.dropna(subset=["target_observed_aqi_mean"])
            .groupby("city", observed=True)
            .size()
        )
        calibrated_cities = set(city_counts[city_counts >= 90].index.astype(str))
        for horizon_day in range(1, settings.forecast_days + 1):
            for target_name, observed_column, base_column in (
                ("mean", "target_observed_aqi_mean", "target_aqi_mean"),
                ("max", "target_observed_aqi_max", "target_aqi_max"),
            ):
                key = DailyAQIModelBundle.key(horizon_day, target_name)
                dev_sensor = development[
                    (development["horizon_day"] == horizon_day)
                    & development["city"].astype(str).isin(calibrated_cities)
                    & development[observed_column].notna()
                ].copy()
                test_sensor = test[
                    (test["horizon_day"] == horizon_day)
                    & test["city"].astype(str).isin(calibrated_cities)
                    & test[observed_column].notna()
                ].copy()
                if len(dev_sensor) < 240 or len(test_sensor) < 40:
                    sensor_calibration_report[key] = {
                        "enabled": False,
                        "reason": "insufficient observed sensor rows",
                        "development_rows": len(dev_sensor),
                        "test_rows": len(test_sensor),
                    }
                    continue
                residual_column = "_sensor_residual_target"
                dev_sensor[residual_column] = (
                    pd.to_numeric(dev_sensor[observed_column], errors="coerce")
                    - pd.to_numeric(dev_sensor[base_column], errors="coerce")
                )
                test_sensor[residual_column] = (
                    pd.to_numeric(test_sensor[observed_column], errors="coerce")
                    - pd.to_numeric(test_sensor[base_column], errors="coerce")
                )
                candidates = build_candidate_models(
                    numeric_features,
                    categorical_features,
                    random_seed=settings.random_seed + 9000,
                    n_jobs=settings.model_n_jobs,
                    horizon_day=horizon_day,
                    target=target_name,
                )
                candidates = {
                    name: candidate
                    for name, candidate in candidates.items()
                    if name in {"ridge", "hist_gradient"}
                }
                candidate_results: dict[str, Any] = {}
                for name, candidate in candidates.items():
                    report, _ = _candidate_cv_plain(
                        candidate,
                        dev_sensor,
                        residual_column,
                        folds=2 if quick else 3,
                    )
                    candidate_results[name] = report
                eligible = [
                    (name, report)
                    for name, report in candidate_results.items()
                    if np.isfinite(report["selection_score"])
                ]
                if not eligible:
                    sensor_calibration_report[key] = {
                        "enabled": False,
                        "reason": "no calibration candidate completed",
                    }
                    continue
                selected_name, _ = min(
                    eligible, key=lambda item: item[1]["selection_score"]
                )
                calibrator = clone(candidates[selected_name])
                calibrator.fit(dev_sensor, dev_sensor[residual_column])
                predicted_residual = np.asarray(calibrator.predict(test_sensor), dtype=float)
                corrected = np.clip(
                    pd.to_numeric(test_sensor[base_column], errors="coerce").to_numpy(dtype=float)
                    + predicted_residual,
                    0,
                    500,
                )
                uncalibrated = pd.to_numeric(
                    test_sensor[base_column], errors="coerce"
                ).to_numpy(dtype=float)
                observed = pd.to_numeric(
                    test_sensor[observed_column], errors="coerce"
                ).to_numpy(dtype=float)
                calibrated_metrics = regression_metrics(observed, corrected)
                uncalibrated_metrics = regression_metrics(observed, uncalibrated)
                if calibrated_metrics.rmse < uncalibrated_metrics.rmse:
                    sensor_calibrators[key] = calibrator
                    enabled = True
                else:
                    enabled = False
                sensor_calibration_report[key] = {
                    "enabled": enabled,
                    "selected": selected_name,
                    "development_rows": len(dev_sensor),
                    "test_rows": len(test_sensor),
                    "calibrated_metrics": calibrated_metrics.to_dict(),
                    "uncalibrated_metrics": uncalibrated_metrics.to_dict(),
                    "candidates": candidate_results,
                }

    scored = pd.concat(test_scored_rows, ignore_index=True, copy=False)
    mean_scored = scored[scored["target_kind"] == "mean"]
    max_scored = scored[scored["target_kind"] == "max"]
    overall_mean = regression_metrics(mean_scored["actual"], mean_scored["prediction"])
    overall_max = regression_metrics(max_scored["actual"], max_scored["prediction"])
    mean_baseline = regression_metrics(mean_scored["actual"], mean_scored["baseline"])
    day_metrics: dict[str, Any] = {}
    for horizon_day in range(1, settings.forecast_days + 1):
        subset = mean_scored[mean_scored["horizon_day"] == horizon_day]
        baseline_subset = subset["baseline"].to_numpy(dtype=float)
        day_metrics[f"day{horizon_day}"] = {
            "model": regression_metrics(subset["actual"], subset["prediction"]).to_dict(),
            "baseline": regression_metrics(subset["actual"], baseline_subset).to_dict(),
        }

    city_metrics = _grouped_test_metrics(mean_scored, "city")
    day3_r2 = float(day_metrics.get("day3", {}).get("model", {}).get("r2", -np.inf))
    all_mean_targets_beat_baseline = all(
        test_reports[DailyAQIModelBundle.key(day, "mean")]["beats_baseline"]
        for day in range(1, settings.forecast_days + 1)
    )
    quality_gate = bool(
        overall_mean.r2 >= settings.minimum_test_r2
        and overall_mean.rmse <= settings.maximum_test_rmse
        and day3_r2 >= settings.minimum_day3_r2
        and all_mean_targets_beat_baseline
    )
    registry = LocalModelRegistry(settings)
    comparison_payload = {
        "test_metrics": {
            "daily_mean": overall_mean.to_dict(),
            "by_day": day_metrics,
            "macro_city": {
                "r2": city_metrics["macro_r2"],
                "rmse": city_metrics["macro_rmse"],
            },
        }
    }
    champion_comparison = registry.compare_with_production(comparison_payload)
    promoted = bool(
        allow_promotion
        and not quick
        and quality_gate
        and champion_comparison["candidate_is_better"]
    )

    bundle = DailyAQIModelBundle(
        models=models,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        selected_algorithms=selected_algorithms,
        intervals=intervals,
        sensor_calibrators=sensor_calibrators,
        calibrated_cities=calibrated_cities,
        metadata={
            "trained_at": datetime.now(UTC).isoformat(),
            "forecast_days": settings.forecast_days,
            "target_contract": "daily_mean_and_daily_peak_us_aqi",
            "training_rows": len(training),
            "cities": sorted(training["city"].unique().tolist()),
            "candidate_cities": selected_keys,
            "mandatory_cities": list(settings.city_selection_mandatory),
            "model_selection": "day3_guarded_champion_challenger_oof_stacking",
            "training_profile": "day3_horizon_specific_precision_with_bias_guard",
        },
    )

    report = _clean_for_json({
        "project_version": "6.9.0",
        "model_name": settings.model_name,
        "mode": "quick_smoke_test" if quick else "full_three_day_training",
        "training_profile": "core8_day3_horizon_specific_bias_guard_champion_challenger",
        "target_contract": {
            "primary": "daily mean US AQI for day 1, day 2 and day 3",
            "secondary": "daily peak US AQI for day 1, day 2 and day 3",
            "hourly_dashboard_curve": "provider forecast calibrated to daily ML predictions",
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "training_data": {
            "rows": len(training),
            "cities": int(training["city"].nunique()),
            "candidate_city_count": int(len(selected_keys)),
            "selected_cities": selected_training_keys,
            "issue_dates": int(training["issue_date"].nunique()),
            "future_weather_coverage": future_coverage,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "provider_snapshot_rows": provider_snapshot_rows,
            "provider_snapshot_feature_coverage": (
                float(training.filter(regex=r"^provider_").notna().mean().mean())
                if any(column.startswith("provider_") for column in training.columns)
                else 0.0
            ),
        },
        "matrix_audit": selected_matrix_audit,
        "candidate_matrix_audit": matrix_audit,
        "matrix_manifest": matrix_manifest,
        "city_selection": city_selection_result.to_dict(),
        "features": {
            "numeric_count": len(numeric_features),
            "categorical": categorical_features,
            "numeric": numeric_features,
        },
        "selected_algorithms": selected_algorithms,
        "candidate_evaluation": candidate_reports,
        "explainability": explainability_reports,
        "sensor_calibration": {
            "calibrated_cities": sorted(calibrated_cities),
            "models_enabled": sorted(sensor_calibrators),
            "evaluation": sensor_calibration_report,
        },
        "test_metrics": {
            "daily_mean": overall_mean.to_dict(),
            "daily_peak": overall_max.to_dict(),
            "daily_mean_baseline": mean_baseline.to_dict(),
            "by_day": day_metrics,
            "by_city": city_metrics["groups"],
            "macro_city": {
                "r2": city_metrics["macro_r2"],
                "rmse": city_metrics["macro_rmse"],
                "cities_beating_baseline": city_metrics["groups_beating_baseline"],
                "city_count": city_metrics["group_count"],
            },
            "by_model_key": test_reports,
        },
        "day3_precision_contract": {
            "recent_fold_weight": settings.day3_recent_fold_weight,
            "extreme_sample_weight": settings.day3_extreme_sample_weight,
            "extreme_threshold": settings.day3_extreme_threshold,
            "stacking_l2_regularization": settings.day3_stacking_l2_regularization,
            "local_expert_max_weight": settings.day3_local_expert_max_weight,
            "local_expert_max_cities": settings.day3_local_expert_max_cities,
            "guarded_calibration": settings.day3_calibration_enabled,
        },
        "quality_gate": {
            "minimum_test_r2": settings.minimum_test_r2,
            "maximum_test_rmse": settings.maximum_test_rmse,
            "minimum_day3_r2": settings.minimum_day3_r2,
            "all_days_beat_baseline": all_mean_targets_beat_baseline,
            "passed": quality_gate,
        },
        "champion_comparison": champion_comparison,
        "promotion": {
            "promoted": promoted,
            "reason": (
                "Passed quality gates and outperformed the existing champion."
                if promoted
                else (
                    "Quick mode never promotes a model."
                    if quick
                    else (
                        champion_comparison["reason"]
                        if quality_gate
                        else (
                            f"Failed gate: daily mean R²={overall_mean.r2:.4f}, "
                            f"RMSE={overall_mean.rmse:.4f}, day-3 R²={day3_r2:.4f}, "
                            f"beats baseline={all_mean_targets_beat_baseline}"
                        )
                    )
                )
            ),
        },
    })
    version_path = registry.register(bundle, report, promote=promoted)
    report["version_path"] = str(version_path)
    (version_path / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (version_path / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path = LocalStore(settings).save_report("training_report_v6.json", report)
    try:
        scored.to_parquet(
            settings.reports_dir / "test_predictions_v6.parquet",
            index=False,
            compression="zstd",
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        scored.to_csv(
            settings.reports_dir / "test_predictions_v6.csv.gz",
            index=False,
            compression="gzip",
        )
    return TrainingResult(
        version_path=version_path,
        promoted=promoted,
        quality_gate_passed=quality_gate,
        report_path=report_path,
        test_metrics=report["test_metrics"],
    )
