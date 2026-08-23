from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float
    bias: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2": float(self.r2),
            "bias": float(self.bias),
            "sample_count": int(self.sample_count),
        }


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Metrics:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if not len(true):
        return Metrics(np.nan, np.nan, np.nan, np.nan, 0)
    return Metrics(
        mae=float(mean_absolute_error(true, pred)),
        rmse=float(np.sqrt(mean_squared_error(true, pred))),
        r2=float(r2_score(true, pred)) if len(true) > 1 else np.nan,
        bias=float(np.mean(pred - true)),
        sample_count=len(true),
    )


def weighted_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        weights = np.ones(len(true), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    true = true[valid]
    pred = pred[valid]
    weights = weights[valid]
    return float(np.sqrt(np.sum(weights * np.square(true - pred)) / weights.sum()))


def seasonal_baseline(frame: pd.DataFrame, target: str) -> np.ndarray:
    mean_column = (
        "us_aqi__mean__lag1" if target.endswith("mean")
        else "us_aqi__max__lag1"
    )
    week_column = (
        "us_aqi__mean__lag7" if target.endswith("mean")
        else "us_aqi__max__lag7"
    )

    def values(name: str, default: float = np.nan) -> np.ndarray:
        if name not in frame:
            return np.full(len(frame), default, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)

    last = values(mean_column)
    week = values(week_column)
    fallback = values("us_aqi__mean" if target.endswith("mean") else "us_aqi__max")
    last = np.where(np.isfinite(last), last, fallback)
    week = np.where(np.isfinite(week), week, last)
    if "horizon_day" in frame:
        horizon = pd.to_numeric(
            frame["horizon_day"], errors="coerce"
        ).fillna(1).to_numpy(dtype=float)
    else:
        horizon = np.ones(len(frame), dtype=float)
    week_weight = np.clip(0.25 + 0.12 * horizon, 0.25, 0.62)
    return np.clip((1 - week_weight) * last + week_weight * week, 0, 500)


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", RobustScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            min_frequency=2,
                        ),
                    ),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_seed: int,
    n_jobs: int,
    horizon_day: int,
    target: str,
) -> dict[str, Pipeline]:
    """Accuracy-oriented candidates required by the project brief.

    Ridge and Random Forest are explicitly named in the PDF. Histogram Gradient
    Boosting and Extra Trees provide robust nonlinear alternatives.
    """
    jobs = max(1, int(n_jobs))
    max_depth = {1: 18, 2: 16, 3: 14}.get(horizon_day, 14)
    min_leaf = {1: 2, 2: 3, 3: 4}.get(horizon_day, 4)
    target_seed = random_seed + horizon_day * 100 + (0 if target.endswith("mean") else 17)

    def pipeline(model: Any, *, scaled: bool = False) -> Pipeline:
        return Pipeline([
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=scaled,
                ),
            ),
            ("model", model),
        ])

    candidates: dict[str, Pipeline] = {
        "ridge": pipeline(Ridge(alpha=18.0), scaled=True),
        "hist_gradient": pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=23,
                max_depth=7 if horizon_day == 1 else 6,
                max_bins=64,
                min_samples_leaf=24 if target.endswith("mean") else 28,
                l2_regularization=4.0,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                random_state=target_seed,
            )
        ),
        "random_forest": pipeline(
            RandomForestRegressor(
                n_estimators=130,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                max_features=0.72,
                bootstrap=True,
                max_samples=0.82,
                n_jobs=jobs,
                random_state=target_seed + 1,
            )
        ),
        "extra_trees": pipeline(
            ExtraTreesRegressor(
                n_estimators=145,
                max_depth=max_depth + 2,
                min_samples_leaf=max(2, min_leaf - 1),
                max_features=0.78,
                bootstrap=False,
                n_jobs=jobs,
                random_state=target_seed + 2,
            )
        ),
    }
    if horizon_day == 3:
        # Longer-horizon candidates use slower learning and stronger averaging.
        # They are accepted only when chronological OOF scoring beats the
        # standard candidates, so the extra complexity cannot force promotion.
        candidates["hist_gradient_long"] = pipeline(
            HistGradientBoostingRegressor(
                max_iter=480,
                learning_rate=0.025,
                max_leaf_nodes=31,
                max_depth=6,
                max_bins=96,
                min_samples_leaf=20 if target.endswith("mean") else 24,
                l2_regularization=5.5,
                early_stopping=True,
                validation_fraction=0.14,
                n_iter_no_change=35,
                random_state=target_seed + 11,
            )
        )
        candidates["extra_trees_long"] = pipeline(
            ExtraTreesRegressor(
                n_estimators=220,
                max_depth=18 if target.endswith("mean") else 16,
                min_samples_leaf=3,
                max_features=0.82,
                bootstrap=False,
                n_jobs=jobs,
                random_state=target_seed + 12,
            )
        )
    return candidates



def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the non-negative unit simplex."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not len(vector):
        return vector
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(vector) + 1, dtype=float)
    positive = ordered - cumulative / indices > 0
    if not np.any(positive):
        return np.full(len(vector), 1.0 / len(vector), dtype=float)
    rho = int(np.flatnonzero(positive)[-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    total = float(projected.sum())
    return projected / total if total > 0 else np.full(len(vector), 1.0 / len(vector))


def fit_convex_blend(
    predictions: np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    prior: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    l2_regularization: float = 2.0,
    max_iterations: int = 2500,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    """Fit leakage-safe non-negative weights that sum to one.

    The optimization is performed only on out-of-fold predictions. A centered
    least-squares objective with a small prior penalty is solved by projected
    gradient descent, avoiding an additional SciPy dependency.
    """
    matrix = np.asarray(predictions, dtype=float)
    truth = np.asarray(target, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(truth):
        raise ValueError("Prediction matrix and target have incompatible shapes")
    if sample_weight is None:
        weights_array = np.ones(len(truth), dtype=float)
    else:
        weights_array = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights_array) != len(truth):
            raise ValueError("sample_weight must match the target length")
    valid = (
        np.isfinite(truth)
        & np.all(np.isfinite(matrix), axis=1)
        & np.isfinite(weights_array)
        & (weights_array > 0)
    )
    matrix = matrix[valid]
    truth = truth[valid]
    weights_array = weights_array[valid]
    if len(truth) < max(20, matrix.shape[1] * 4):
        raise ValueError("Insufficient finite out-of-fold rows for stacking")
    weights_array = weights_array * (len(weights_array) / weights_array.sum())
    total_sample_weight = float(weights_array.sum())

    feature_count = matrix.shape[1]
    if prior is None:
        weights = np.full(feature_count, 1.0 / feature_count, dtype=float)
    else:
        weights = _project_to_simplex(np.asarray(prior, dtype=float))
    prior_weights = weights.copy()

    x_mean = np.average(matrix, axis=0, weights=weights_array)
    y_mean = float(np.average(truth, weights=weights_array))
    centered_x = matrix - x_mean
    centered_y = truth - y_mean
    weighted_x = centered_x * np.sqrt(weights_array)[:, None]
    spectral = float(np.linalg.norm(weighted_x, ord=2))
    lipschitz = 2.0 * (
        spectral * spectral / total_sample_weight
        + max(0.0, l2_regularization)
    )
    step = 1.0 / max(lipschitz, 1e-9)

    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        residual = centered_x @ weights - centered_y
        gradient = (2.0 / total_sample_weight) * (
            centered_x.T @ (weights_array * residual)
        )
        gradient += 2.0 * max(0.0, l2_regularization) * (weights - prior_weights)
        updated = _project_to_simplex(weights - step * gradient)
        if float(np.max(np.abs(updated - weights))) <= tolerance:
            weights = updated
            break
        weights = updated

    intercept = float(y_mean - x_mean @ weights)
    fitted = matrix @ weights + intercept
    rmse = float(np.sqrt(
        np.sum(weights_array * np.square(truth - fitted)) / total_sample_weight
    ))
    return weights, intercept, {
        "rows": int(len(truth)),
        "iterations": int(iterations),
        "objective_rmse": rmse,
        "weight_sum": float(weights.sum()),
        "sample_weighted": bool(sample_weight is not None),
        "effective_weight_sum": total_sample_weight,
    }


def shrunken_group_offsets(
    labels: pd.Series,
    residuals: np.ndarray,
    *,
    shrinkage: float,
) -> dict[str, float]:
    """Estimate conservative group residual offsets from OOF residuals."""
    frame = pd.DataFrame({
        "label": labels.astype(str).to_numpy(),
        "residual": np.asarray(residuals, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {}
    grouped = frame.groupby("label", observed=True)["residual"].agg(["mean", "count"])
    factor = grouped["count"] / (grouped["count"] + max(0.0, float(shrinkage)))
    return {
        str(index): float(row["mean"] * factor.loc[index])
        for index, row in grouped.iterrows()
    }


@dataclass
class OOFStackingRegressor:
    """Serializable convex ensemble trained only from OOF predictions."""

    base_models: dict[str, Any]
    weights: dict[str, float]
    intercept: float
    target_column: str
    global_bias_offset: float = 0.0
    province_offsets: dict[str, float] = field(default_factory=dict)
    city_offsets: dict[str, float] = field(default_factory=dict)
    month_offsets: dict[str, float] = field(default_factory=dict)
    city_month_offsets: dict[str, float] = field(default_factory=dict)
    recent_city_offsets: dict[str, float] = field(default_factory=dict)
    local_models: dict[str, Any] = field(default_factory=dict)
    local_weights: dict[str, float] = field(default_factory=dict)
    local_algorithms: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        components: dict[str, np.ndarray] = {}
        for name in self.weights:
            if name == "seasonal_baseline":
                components[name] = seasonal_baseline(frame, self.target_column)
            else:
                model = self.base_models.get(name)
                if model is None:
                    raise KeyError(f"Stacking model is missing base learner {name}")
                components[name] = np.asarray(model.predict(frame), dtype=float)
        return pd.DataFrame(components, index=frame.index)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        components = self.predict_components(frame)
        prediction = np.full(len(frame), float(self.intercept), dtype=float)
        for name, weight in self.weights.items():
            prediction += float(weight) * components[name].to_numpy(dtype=float)
        prediction += float(getattr(self, "global_bias_offset", 0.0))

        local_models = getattr(self, "local_models", {})
        local_weights = getattr(self, "local_weights", {})
        if "city" in frame and local_models:
            cities = frame["city"].astype(str)
            for city, local_model in local_models.items():
                mask = cities == str(city)
                if not mask.any():
                    continue
                local_prediction = np.asarray(local_model.predict(frame.loc[mask]), dtype=float)
                weight = float(local_weights.get(str(city), 0.0))
                prediction[mask.to_numpy()] = (
                    (1.0 - weight) * prediction[mask.to_numpy()]
                    + weight * local_prediction
                )

        province_offsets = getattr(self, "province_offsets", {})
        city_offsets = getattr(self, "city_offsets", {})
        month_offsets = getattr(self, "month_offsets", {})
        city_month_offsets = getattr(self, "city_month_offsets", {})
        recent_city_offsets = getattr(self, "recent_city_offsets", {})
        if "province" in frame and province_offsets:
            prediction += frame["province"].astype(str).map(province_offsets).fillna(0.0).to_numpy(dtype=float)
        if "city" in frame and city_offsets:
            prediction += frame["city"].astype(str).map(city_offsets).fillna(0.0).to_numpy(dtype=float)
        if "issue_date" in frame and month_offsets:
            month = pd.to_datetime(frame["issue_date"], errors="coerce").dt.month.astype("Int64").astype(str)
            prediction += month.map(month_offsets).fillna(0.0).to_numpy(dtype=float)
        if "city" in frame and "issue_date" in frame and city_month_offsets:
            month = pd.to_datetime(frame["issue_date"], errors="coerce").dt.month.astype("Int64").astype(str)
            city_month = frame["city"].astype(str) + "|" + month
            prediction += city_month.map(city_month_offsets).fillna(0.0).to_numpy(dtype=float)
        if "city" in frame and recent_city_offsets:
            prediction += frame["city"].astype(str).map(recent_city_offsets).fillna(0.0).to_numpy(dtype=float)
        return np.clip(prediction, 0, 500)

    @property
    def dominant_base_model(self) -> str | None:
        candidates = {name: weight for name, weight in self.weights.items() if name != "seasonal_baseline"}
        return max(candidates, key=candidates.get) if candidates else None

def chronological_partitions(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issue_dates = pd.Series(pd.to_datetime(frame["issue_date"]).dropna().unique()).sort_values()
    if len(issue_dates) < 180:
        raise ValueError("At least 180 unique issue dates are required for robust training")
    train_cut = issue_dates.iloc[int(len(issue_dates) * train_fraction) - 1]
    validation_cut = issue_dates.iloc[
        int(len(issue_dates) * (train_fraction + validation_fraction)) - 1
    ]
    train = frame[frame["issue_date"] <= train_cut].copy()
    validation_start = train_cut + pd.Timedelta(days=int(embargo_days) + 1)
    validation = frame[
        (frame["issue_date"] >= validation_start)
        & (frame["issue_date"] <= validation_cut)
    ].copy()
    test_start = validation_cut + pd.Timedelta(days=int(embargo_days) + 1)
    test = frame[frame["issue_date"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, validation, test


def iter_rolling_folds(
    frame: pd.DataFrame,
    *,
    folds: int = 3,
    embargo_days: int = 3,
):
    """Yield chronological folds one at a time to bound peak memory.

    The previous implementation materialized every expanding train/validation
    DataFrame pair at once. With hundreds of engineered columns and four OOF
    folds, that retained several large copies of the development frame.
    """
    dates = (
        pd.Series(pd.to_datetime(frame["issue_date"]).dropna().unique())
        .sort_values()
        .reset_index(drop=True)
    )
    if len(dates) < 120:
        return
    blocks = [block for block in np.array_split(dates.to_numpy(), folds + 1) if len(block)]
    for index in range(1, len(blocks)):
        train_end = pd.Timestamp(blocks[index - 1][-1])
        validation_start = train_end + pd.Timedelta(days=int(embargo_days) + 1)
        validation_dates = pd.to_datetime(blocks[index])
        validation_end = pd.Timestamp(validation_dates[-1])
        train_mask = frame["issue_date"] <= train_end
        validation_mask = (
            (frame["issue_date"] >= validation_start)
            & (frame["issue_date"] <= validation_end)
        )
        train = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        if len(train) >= 100 and len(validation) >= 30:
            yield train, validation


def rolling_folds(
    frame: pd.DataFrame,
    *,
    folds: int = 3,
    embargo_days: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Backward-compatible materialized folds for tests and small callers."""
    return list(iter_rolling_folds(frame, folds=folds, embargo_days=embargo_days))


@dataclass
class DailyAQIModelBundle:
    models: dict[str, Any]
    numeric_features: list[str]
    categorical_features: list[str]
    selected_algorithms: dict[str, str]
    intervals: dict[str, dict[str, float]]
    sensor_calibrators: dict[str, Any] = field(default_factory=dict)
    calibrated_cities: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def key(horizon_day: int, target: str) -> str:
        return f"day{horizon_day}_{target}"

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for horizon_day in sorted(pd.to_numeric(frame["horizon_day"], errors="coerce").dropna().astype(int).unique()):
            subset = frame[pd.to_numeric(frame["horizon_day"], errors="coerce") == horizon_day]
            if subset.empty:
                continue
            record: dict[str, Any] = {"horizon_day": horizon_day}
            for target in ("mean", "max"):
                key = self.key(horizon_day, target)
                model = self.models.get(key)
                if model is None:
                    raise KeyError(f"Model bundle is missing {key}")
                prediction = np.asarray(model.predict(subset), dtype=float)
                value = float(np.clip(prediction[0], 0, 500))
                city_value = str(subset["city"].iloc[0]) if "city" in subset else ""
                calibrator = self.sensor_calibrators.get(key)
                if calibrator is not None and city_value in self.calibrated_cities:
                    residual = float(np.asarray(calibrator.predict(subset), dtype=float)[0])
                    value = float(np.clip(value + residual, 0, 500))
                record[f"aqi_{target}"] = value
                interval = self.intervals.get(key, {})
                record[f"aqi_{target}_lower"] = float(np.clip(value - interval.get("lower_error", 0), 0, 500))
                record[f"aqi_{target}_upper"] = float(np.clip(value + interval.get("upper_error", 0), 0, 500))
            rows.append(record)
        return pd.DataFrame(rows).sort_values("horizon_day").reset_index(drop=True)
