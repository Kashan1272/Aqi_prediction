from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def explain_pipeline(
    pipeline: Any,
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    max_rows: int = 500,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Return SHAP feature importance when available, otherwise permutation importance."""
    if frame.empty:
        return {"method": "unavailable", "features": []}
    sample_size = min(max_rows, len(frame))
    sample = frame.sample(sample_size, random_state=random_seed)
    y = target.loc[sample.index]

    preprocessor = pipeline.named_steps.get("preprocessor")
    estimator = pipeline.named_steps.get("model")
    if preprocessor is not None and estimator is not None:
        try:
            import shap  # type: ignore

            transformed = preprocessor.transform(sample)
            names = preprocessor.get_feature_names_out().tolist()
            if estimator.__class__.__name__ in {
                "RandomForestRegressor",
                "ExtraTreesRegressor",
                "HistGradientBoostingRegressor",
            }:
                explainer = shap.TreeExplainer(estimator)
                values = explainer.shap_values(transformed)
            else:
                explainer = shap.LinearExplainer(estimator, transformed)
                values = explainer.shap_values(transformed)
            array = np.asarray(values, dtype=float)
            if array.ndim > 2:
                array = np.squeeze(array)
            importance = np.nanmean(np.abs(array), axis=0)
            order = np.argsort(importance)[::-1][:30]
            return {
                "method": "shap",
                "features": [
                    {"feature": names[index], "importance": float(importance[index])}
                    for index in order
                ],
            }
        except Exception:
            pass

    # The pipeline itself accepts the original named columns. Permutation
    # importance is deterministic, model-agnostic, and requires no extra package.
    result = permutation_importance(
        pipeline,
        sample,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=2,
        random_state=random_seed,
        n_jobs=1,
    )
    names = sample.columns.tolist()
    order = np.argsort(result.importances_mean)[::-1][:30]
    return {
        "method": "permutation_rmse",
        "features": [
            {
                "feature": names[index],
                "importance": float(result.importances_mean[index]),
                "std": float(result.importances_std[index]),
            }
            for index in order
        ],
    }
