from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import Settings
from .models import regression_metrics, shrunken_group_offsets

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalExpertResult:
    models: dict[str, Any]
    weights: dict[str, float]
    algorithms: dict[str, str]
    oof_prediction: np.ndarray
    report: dict[str, Any]


def _rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(actual[valid] - prediction[valid]))))


def _local_oof_predictions(
    candidate: Any,
    city_frame: pd.DataFrame,
    city_oof: pd.DataFrame,
    target_column: str,
    *,
    embargo_days: int = 3,
) -> np.ndarray:
    """Generate predictions on the exact global OOF validation dates."""
    output = np.full(len(city_oof), np.nan, dtype=np.float32)
    issue_dates = pd.to_datetime(city_frame["issue_date"], errors="coerce").dt.normalize()
    frame = city_frame.copy()
    frame["issue_date"] = issue_dates

    for fold in sorted(
        pd.to_numeric(city_oof["oof_fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    ):
        positions = np.flatnonzero(
            pd.to_numeric(city_oof["oof_fold"], errors="coerce").to_numpy(dtype=float)
            == fold
        )
        if not len(positions):
            continue
        validation_dates = pd.to_datetime(
            city_oof.iloc[positions]["issue_date"], errors="coerce"
        ).dt.normalize()
        validation_dates = validation_dates.dropna().drop_duplicates().sort_values()
        if validation_dates.empty:
            continue
        cutoff = validation_dates.min() - pd.Timedelta(days=int(embargo_days))
        train = frame[frame["issue_date"] < cutoff]
        validation = frame[frame["issue_date"].isin(set(validation_dates))]
        if len(train) < 120 or validation.empty:
            continue

        fitted = clone(candidate)
        try:
            fitted.fit(train, train[target_column])
            predicted = np.clip(
                np.asarray(fitted.predict(validation), dtype=np.float32), 0, 500
            )
            mapping = dict(zip(validation["issue_date"], predicted, strict=True))
            output[positions] = [
                mapping.get(pd.Timestamp(value).normalize(), np.nan)
                for value in city_oof.iloc[positions]["issue_date"]
            ]
        finally:
            del fitted, train, validation
            gc.collect()
    return output


def fit_local_experts(
    candidate_templates: dict[str, Any],
    frame: pd.DataFrame,
    target_column: str,
    oof: pd.DataFrame,
    global_prediction: np.ndarray,
    *,
    settings: Settings,
    horizon_day: int = 1,
) -> LocalExpertResult:
    """Fit conservative city experts using development OOF predictions only."""
    blended = np.asarray(global_prediction, dtype=np.float32).copy()
    actual = oof["actual"].to_numpy(dtype=np.float32)
    candidates = {
        name: candidate_templates[name]
        for name in settings.local_expert_candidates
        if name in candidate_templates
    }
    if not settings.local_experts_enabled or not candidates:
        return LocalExpertResult({}, {}, {}, blended, {
            "enabled": False,
            "reason": "disabled or no configured candidate was available",
            "cities": {},
        })

    max_weight = (
        settings.day3_local_expert_max_weight
        if int(horizon_day) == 3
        else settings.local_expert_max_weight
    )
    max_cities = (
        settings.day3_local_expert_max_cities
        if int(horizon_day) == 3
        else settings.local_expert_max_cities_per_target
    )

    provisional: list[dict[str, Any]] = []
    city_reports: dict[str, Any] = {}
    for city in sorted(oof["city"].astype(str).unique()):
        city_mask = oof["city"].astype(str).to_numpy() == city
        city_positions = np.flatnonzero(city_mask)
        city_frame = frame[frame["city"].astype(str) == city].copy()
        city_oof = oof.iloc[city_positions].reset_index(drop=True)
        if len(city_frame) < settings.local_expert_min_rows or len(city_oof) < 120:
            city_reports[city] = {
                "enabled": False,
                "reason": "insufficient development rows",
                "development_rows": len(city_frame),
                "oof_rows": len(city_oof),
            }
            continue

        actual_city = actual[city_positions]
        global_city = blended[city_positions]
        global_rmse = _rmse(actual_city, global_city)
        candidate_results: dict[str, Any] = {}
        best: tuple[str, Any, np.ndarray, float] | None = None
        for name, template in candidates.items():
            try:
                local_oof = _local_oof_predictions(
                    template,
                    city_frame,
                    city_oof,
                    target_column,
                )
                valid = (
                    np.isfinite(local_oof)
                    & np.isfinite(actual_city)
                    & np.isfinite(global_city)
                )
                if int(valid.sum()) < settings.local_expert_min_oof_rows:
                    candidate_results[name] = {
                        "valid_rows": int(valid.sum()),
                        "accepted": False,
                        "reason": "insufficient finite OOF predictions",
                    }
                    continue
                local_rmse = _rmse(actual_city[valid], local_oof[valid])
                candidate_results[name] = {
                    "valid_rows": int(valid.sum()),
                    "local_rmse": local_rmse,
                }
                if best is None or local_rmse < best[3]:
                    best = (name, template, local_oof, local_rmse)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                candidate_results[name] = {
                    "accepted": False,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
                LOGGER.warning("Local expert %s/%s failed: %s", city, name, exc)
            finally:
                gc.collect()

        if best is None or not np.isfinite(global_rmse):
            city_reports[city] = {
                "enabled": False,
                "reason": "no local candidate completed",
                "candidates": candidate_results,
            }
            continue

        best_name, best_template, local_oof, local_rmse = best
        valid = (
            np.isfinite(local_oof)
            & np.isfinite(actual_city)
            & np.isfinite(global_city)
        )
        delta = local_oof[valid].astype(float) - global_city[valid].astype(float)
        residual = actual_city[valid].astype(float) - global_city[valid].astype(float)
        denominator = float(np.dot(delta, delta))
        raw_weight = (
            float(np.dot(delta, residual) / denominator)
            if denominator > 1e-9
            else 0.0
        )
        raw_weight = float(np.clip(raw_weight, 0.0, max_weight))
        shrink = float(
            valid.sum() / (valid.sum() + settings.local_expert_weight_shrinkage)
        )
        weight = raw_weight * shrink
        proposed = global_city.copy()
        proposed[valid] = global_city[valid] + weight * (
            local_oof[valid] - global_city[valid]
        )
        blended_rmse = _rmse(actual_city, proposed)
        relative_gain = (global_rmse - blended_rmse) / max(global_rmse, 1e-9)
        city_reports[city] = {
            "enabled": bool(
                relative_gain >= settings.local_expert_min_gain and weight > 0
            ),
            "selected_algorithm": best_name,
            "development_rows": len(city_frame),
            "oof_rows": len(city_oof),
            "valid_oof_rows": int(valid.sum()),
            "global_rmse": global_rmse,
            "local_rmse": local_rmse,
            "blended_rmse": blended_rmse,
            "relative_gain": relative_gain,
            "raw_weight": raw_weight,
            "shrunken_weight": weight,
            "horizon_day": int(horizon_day),
            "candidates": candidate_results,
        }
        if relative_gain >= settings.local_expert_min_gain and weight > 0:
            provisional.append({
                "city": city,
                "template": best_template,
                "algorithm": best_name,
                "weight": weight,
                "gain": relative_gain,
                "prediction": proposed,
                "positions": city_positions,
                "frame": city_frame,
            })

    provisional.sort(key=lambda item: (item["gain"], item["weight"]), reverse=True)
    retained = provisional[:max_cities]
    models: dict[str, Any] = {}
    weights: dict[str, float] = {}
    algorithms: dict[str, str] = {}
    for item in retained:
        fitted = clone(item["template"])
        city = str(item["city"])
        try:
            fitted.fit(item["frame"], item["frame"][target_column])
            models[city] = fitted
            weights[city] = float(item["weight"])
            algorithms[city] = str(item["algorithm"])
            blended[item["positions"]] = item["prediction"]
            city_reports[city]["retained"] = True
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            city_reports[city]["retained"] = False
            city_reports[city]["final_fit_error"] = str(exc)
            LOGGER.warning("Final local expert fit %s failed: %s", city, exc)
        finally:
            gc.collect()

    retained_cities = set(models)
    for item in provisional:
        city = str(item["city"])
        if city not in retained_cities and "retained" not in city_reports[city]:
            city_reports[city]["retained"] = False
            city_reports[city]["reason"] = (
                "positive OOF gain but outside retained-city cap"
            )

    return LocalExpertResult(
        models=models,
        weights=weights,
        algorithms=algorithms,
        oof_prediction=blended,
        report={
            "enabled": True,
            "horizon_day": int(horizon_day),
            "candidate_algorithms": list(candidates),
            "retained_cities": sorted(models),
            "max_cities_per_target": int(max_cities),
            "maximum_local_weight": float(max_weight),
            "cities": city_reports,
            "oof_metrics_before": regression_metrics(
                actual, global_prediction
            ).to_dict(),
            "oof_metrics_after": regression_metrics(actual, blended).to_dict(),
        },
    )


def _labels(oof: pd.DataFrame) -> dict[str, pd.Series]:
    issue_month = (
        pd.to_datetime(oof["issue_date"], errors="coerce")
        .dt.month.astype("Int64").astype(str)
    )
    city = oof["city"].astype(str)
    province = oof["province"].astype(str)
    return {
        "global": pd.Series("__all__", index=oof.index, dtype="object"),
        "province": province,
        "city": city,
        "month": issue_month,
        "city_month": city + "|" + issue_month,
    }


def _offset(values: pd.Series, mapping: dict[str, float]) -> np.ndarray:
    return values.astype(str).map(mapping).fillna(0.0).to_numpy(dtype=np.float32)


def _relative_rmse_gain(before: Any, after: Any) -> float:
    if not np.isfinite(before.rmse) or before.rmse <= 0 or not np.isfinite(after.rmse):
        return float("-inf")
    return float((before.rmse - after.rmse) / before.rmse)


def fit_hierarchical_calibration(
    oof: pd.DataFrame,
    actual: np.ndarray,
    base_prediction: np.ndarray,
    *,
    settings: Settings,
    horizon_day: int = 1,
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, Any]]:
    """Select residual corrections on a temporal OOF holdout, then refit them.

    The most recent OOF fold is reserved to decide whether each correction is
    safe. Accepted mappings are then refit on all development OOF rows for the
    serialised model. This avoids blindly adding city/month offsets that merely
    reduce in-sample bias.
    """
    base = np.asarray(base_prediction, dtype=np.float32)
    actual_array = np.asarray(actual, dtype=np.float32)
    labels = _labels(oof)
    folds = pd.to_numeric(oof.get("oof_fold"), errors="coerce")
    finite_folds = folds.dropna()

    if finite_folds.nunique() >= 2:
        latest_fold = int(finite_folds.max())
        selection_train = (folds < latest_fold).fillna(False).to_numpy()
        selection_eval = (folds == latest_fold).fillna(False).to_numpy()
    else:
        order = np.argsort(
            pd.to_datetime(oof["issue_date"], errors="coerce")
            .fillna(pd.Timestamp.min)
            .to_numpy()
        )
        split = max(1, int(len(order) * 0.8))
        selection_train = np.zeros(len(oof), dtype=bool)
        selection_eval = np.zeros(len(oof), dtype=bool)
        selection_train[order[:split]] = True
        selection_eval[order[split:]] = True
        latest_fold = -1

    if int(selection_train.sum()) < 120 or int(selection_eval.sum()) < 60:
        return base.copy(), {
            "global": {}, "province": {}, "city": {}, "month": {},
            "city_month": {}, "recent_city": {},
        }, {
            "enabled": False,
            "reason": "insufficient temporal calibration holdout rows",
            "metrics_before": regression_metrics(actual_array, base).to_dict(),
            "metrics_after": regression_metrics(actual_array, base).to_dict(),
        }

    train_prediction = base[selection_train].copy()
    eval_prediction = base[selection_eval].copy()
    train_actual = actual_array[selection_train]
    eval_actual = actual_array[selection_eval]

    base_shrinkage = {
        "global": settings.calibration_bias_shrinkage,
        "province": settings.stacking_province_shrinkage,
        "city": settings.stacking_city_shrinkage,
        "month": settings.calibration_month_shrinkage,
        "city_month": settings.calibration_city_month_shrinkage,
    }
    step_order = ["global", "city", "province", "month", "city_month"]
    accepted: list[dict[str, Any]] = []
    step_reports: dict[str, Any] = {}
    min_gain = (
        settings.day3_calibration_min_rmse_gain
        if int(horizon_day) == 3 and settings.day3_calibration_enabled
        else 0.0
    )
    bias_tolerance = (
        settings.day3_calibration_max_abs_bias_regression
        if int(horizon_day) == 3
        else 3.0
    )

    factors = (0.5, 1.0, 2.0) if int(horizon_day) == 3 else (1.0,)
    for name in step_order:
        before = regression_metrics(eval_actual, eval_prediction)
        best: dict[str, Any] | None = None
        for factor in factors:
            shrinkage = float(base_shrinkage[name] * factor)
            mapping = shrunken_group_offsets(
                labels[name][selection_train],
                train_actual - train_prediction,
                shrinkage=shrinkage,
            )
            proposed_eval = eval_prediction + _offset(
                labels[name][selection_eval], mapping
            )
            after = regression_metrics(eval_actual, proposed_eval)
            gain = _relative_rmse_gain(before, after)
            bias_regression = float(abs(after.bias) - abs(before.bias))
            # Bias-only corrections may be retained when RMSE is effectively
            # unchanged; all other corrections must improve temporal holdout RMSE.
            bias_improvement = float(abs(before.bias) - abs(after.bias))
            accepted_candidate = bool(
                (gain >= min_gain and bias_regression <= bias_tolerance)
                or (bias_improvement >= 0.10 and gain >= -0.0005)
            )
            score = float(after.rmse + 0.10 * abs(after.bias))
            candidate = {
                "mapping": mapping,
                "shrinkage": shrinkage,
                "metrics": after,
                "rmse_gain": gain,
                "bias_regression": bias_regression,
                "bias_improvement": bias_improvement,
                "accepted": accepted_candidate,
                "score": score,
            }
            if best is None or (
                candidate["accepted"] and not best["accepted"]
            ) or (
                candidate["accepted"] == best["accepted"]
                and candidate["score"] < best["score"]
            ):
                best = candidate

        assert best is not None
        step_reports[name] = {
            "accepted": bool(best["accepted"]),
            "selected_shrinkage": float(best["shrinkage"]),
            "evaluation_before": before.to_dict(),
            "evaluation_after": best["metrics"].to_dict(),
            "relative_rmse_gain": float(best["rmse_gain"]),
            "absolute_bias_regression": float(best["bias_regression"]),
            "mapping_size": int(len(best["mapping"])),
        }
        if best["accepted"] and best["mapping"]:
            train_prediction += _offset(
                labels[name][selection_train], best["mapping"]
            )
            eval_prediction += _offset(
                labels[name][selection_eval], best["mapping"]
            )
            accepted.append({
                "name": name,
                "shrinkage": float(best["shrinkage"]),
            })

    # A recent-city correction is accepted only when the penultimate OOF fold
    # predicts the direction of the latest fold. The deployed mapping is then
    # refit from the latest development fold, never from final test rows.
    recent_accepted = False
    recent_shrinkage = float(settings.calibration_recent_city_shrinkage)
    if settings.calibration_recent_city_enabled and finite_folds.nunique() >= 3:
        previous_fold = latest_fold - 1
        recent_train = (folds == previous_fold).fillna(False).to_numpy()
        recent_eval = selection_eval
        if int(recent_train.sum()) >= 60 and int(recent_eval.sum()) >= 60:
            before = regression_metrics(eval_actual, eval_prediction)
            recent_mapping = shrunken_group_offsets(
                labels["city"][recent_train],
                actual_array[recent_train] - base[recent_train],
                shrinkage=recent_shrinkage,
            )
            proposed = eval_prediction + _offset(
                labels["city"][recent_eval], recent_mapping
            )
            after = regression_metrics(eval_actual, proposed)
            gain = _relative_rmse_gain(before, after)
            bias_regression = float(abs(after.bias) - abs(before.bias))
            recent_accepted = bool(
                gain >= min_gain and bias_regression <= bias_tolerance
            )
            step_reports["recent_city"] = {
                "accepted": recent_accepted,
                "selected_shrinkage": recent_shrinkage,
                "evaluation_before": before.to_dict(),
                "evaluation_after": after.to_dict(),
                "relative_rmse_gain": gain,
                "absolute_bias_regression": bias_regression,
                "mapping_size": int(len(recent_mapping)),
            }
        else:
            step_reports["recent_city"] = {
                "accepted": False,
                "reason": "insufficient penultimate/latest fold rows",
            }
    else:
        step_reports["recent_city"] = {
            "accepted": False,
            "reason": "recent-city calibration disabled or fewer than three folds",
        }

    prediction = base.copy()
    offsets: dict[str, dict[str, float]] = {
        "global": {}, "province": {}, "city": {}, "month": {},
        "city_month": {}, "recent_city": {},
    }
    for item in accepted:
        name = str(item["name"])
        mapping = shrunken_group_offsets(
            labels[name],
            actual_array - prediction,
            shrinkage=float(item["shrinkage"]),
        )
        offsets[name] = mapping
        prediction += _offset(labels[name], mapping)

    if recent_accepted:
        latest = selection_eval
        mapping = shrunken_group_offsets(
            labels["city"][latest],
            (actual_array - prediction)[latest],
            shrinkage=recent_shrinkage,
        )
        offsets["recent_city"] = mapping
        prediction += _offset(labels["city"], mapping)

    prediction = np.clip(prediction, 0, 500)
    return prediction, offsets, {
        "enabled": True,
        "horizon_day": int(horizon_day),
        "selection_latest_oof_fold": int(latest_fold),
        "selection_train_rows": int(selection_train.sum()),
        "selection_evaluation_rows": int(selection_eval.sum()),
        "accepted_steps": [item["name"] for item in accepted]
        + (["recent_city"] if recent_accepted else []),
        "steps": step_reports,
        "selection_metrics_before": regression_metrics(
            eval_actual, base[selection_eval]
        ).to_dict(),
        "selection_metrics_after": regression_metrics(
            eval_actual, eval_prediction
        ).to_dict(),
        "metrics_before": regression_metrics(actual_array, base).to_dict(),
        "metrics_after": regression_metrics(actual_array, prediction).to_dict(),
        "offset_counts": {
            name: len(mapping) for name, mapping in offsets.items()
        },
        "bias_guard_enabled": True,
    }
