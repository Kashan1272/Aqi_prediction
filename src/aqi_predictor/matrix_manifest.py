from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    values = pd.to_datetime(frame.get("issue_date"), errors="coerce").dropna()
    return {
        "start": values.min().isoformat() if len(values) else None,
        "end": values.max().isoformat() if len(values) else None,
    }


def build_matrix_manifest(
    training: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected_cities: list[str],
    removed_features: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe and fingerprint the exact matrix used by the training run."""
    feature_schema = []
    for column in numeric_features:
        values = pd.to_numeric(training[column], errors="coerce")
        finite = values[np.isfinite(values)]
        feature_schema.append({
            "name": column,
            "kind": "numeric",
            "dtype": str(training[column].dtype),
            "missing_ratio": float(values.isna().mean()),
            "finite_count": int(np.isfinite(values).sum()),
            "unique_finite": int(finite.nunique()),
        })
    for column in categorical_features:
        values = training[column].astype("string")
        feature_schema.append({
            "name": column,
            "kind": "categorical",
            "dtype": str(training[column].dtype),
            "missing_ratio": float(values.isna().mean()),
            "unique": int(values.dropna().nunique()),
        })

    split_payload = {}
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        split_payload[name] = {
            "rows": int(len(frame)),
            "issue_dates": int(
                pd.to_datetime(frame.get("issue_date"), errors="coerce").nunique()
            ),
            "date_range": _date_range(frame),
            "city_rows": {
                str(key): int(value)
                for key, value in frame.groupby("city", observed=True).size().items()
            },
            "horizon_rows": {
                str(int(key)): int(value)
                for key, value in frame.groupby("horizon_day", observed=True).size().items()
            },
        }

    provider_columns = [
        column for column in training.columns if column.startswith("provider_")
    ]
    provider_coverage = {
        column: float(pd.to_numeric(training[column], errors="coerce").notna().mean())
        for column in provider_columns
        if pd.api.types.is_numeric_dtype(training[column])
    }
    provider_by_horizon = {}
    for horizon, subset in training.groupby("horizon_day", observed=True):
        provider_by_horizon[str(int(horizon))] = {
            column: float(
                pd.to_numeric(subset[column], errors="coerce").notna().mean()
            )
            for column in provider_columns
            if pd.api.types.is_numeric_dtype(subset[column])
        }

    schema_contract = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_schema": feature_schema,
        "selected_cities": sorted(selected_cities),
        "horizons": sorted(
            int(value)
            for value in pd.to_numeric(
                training["horizon_day"], errors="coerce"
            ).dropna().unique()
        ),
    }
    data_contract = {
        "rows": int(len(training)),
        "issue_dates": int(
            pd.to_datetime(training["issue_date"], errors="coerce").nunique()
        ),
        "city_rows": {
            str(key): int(value)
            for key, value in training.groupby("city", observed=True).size().items()
        },
        "horizon_rows": {
            str(int(key)): int(value)
            for key, value in training.groupby("horizon_day", observed=True).size().items()
        },
        "splits": split_payload,
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": "v6.9_day3_precision_matrix_contract",
        "schema_sha256": _stable_hash(schema_contract),
        "data_contract_sha256": _stable_hash(data_contract),
        "schema": schema_contract,
        "data": data_contract,
        "provider_snapshot_coverage": provider_coverage,
        "provider_snapshot_coverage_by_horizon": provider_by_horizon,
        "removed_features": removed_features or {},
    }
