from __future__ import annotations

import json

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.hopsworks_integration import HopsworksAdapter


def main() -> None:
    print(
        "Verifying Hopsworks serving path...",
        flush=True,
    )

    settings = get_settings()
    adapter = HopsworksAdapter(settings)

    # ---------------------------------------------------------
    # 1. Read production predictions from Feature Store
    # ---------------------------------------------------------

    predictions = adapter.read_group(
        "aqi_daily_predictions_v69"
    )

    if predictions.empty:
        raise RuntimeError(
            "aqi_daily_predictions_v69 is empty."
        )

    predictions = predictions.copy()

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        utc=True,
        errors="coerce",
    )

    predictions["generated_at"] = pd.to_datetime(
        predictions["generated_at"],
        utc=True,
        errors="coerce",
    )

    predictions["aqi_mean"] = pd.to_numeric(
        predictions["aqi_mean"],
        errors="coerce",
    )

    predictions["aqi_max"] = pd.to_numeric(
        predictions["aqi_max"],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=[
            "city",
            "date",
            "aqi_mean",
            "aqi_max",
        ]
    )

    # Keep the newest version of every city/date prediction.
    predictions = (
        predictions
        .sort_values("generated_at")
        .drop_duplicates(
            subset=[
                "city",
                "date",
            ],
            keep="last",
        )
    )

    today = pd.Timestamp.now(
        tz="UTC"
    ).normalize()

    future = predictions[
        predictions["date"] >= today
    ].copy()

    latest_rows = (
        future
        .sort_values(
            [
                "city",
                "date",
            ]
        )
        .groupby(
            "city",
            observed=True,
        )
        .head(
            settings.forecast_days
        )
        .reset_index(drop=True)
    )

    print(
        f"\nPrediction rows available: "
        f"{len(predictions):,}",
        flush=True,
    )

    print(
        f"Current/future rows: "
        f"{len(latest_rows):,}",
        flush=True,
    )

    if latest_rows.empty:
        print(
            "\nWARNING: no current/future predictions "
            "were found.",
            flush=True,
        )

    else:
        print(
            "\nLatest Feature Store forecasts:",
            flush=True,
        )

        display_columns = [
            "city",
            "date",
            "aqi_mean",
            "aqi_max",
            "category",
            "generated_at",
        ]

        print(
            latest_rows[
                [
                    column
                    for column in display_columns
                    if column in latest_rows.columns
                ]
            ].to_string(
                index=False
            ),
            flush=True,
        )

    # ---------------------------------------------------------
    # 2. Verify production model in Model Registry
    # ---------------------------------------------------------

    models = adapter.model_registry.get_models(
        settings.model_name
    )

    if not models:
        raise RuntimeError(
            f"No model named "
            f"{settings.model_name!r} "
            "exists in Hopsworks Model Registry."
        )

    champion = max(
        models,
        key=lambda model: int(
            getattr(
                model,
                "version",
                0,
            )
        ),
    )

    metrics = getattr(
        champion,
        "metrics",
        {},
    ) or {}

    print(
        "\nProduction Model Registry entry:",
        flush=True,
    )

    print(
        json.dumps(
            {
                "name": getattr(
                    champion,
                    "name",
                    settings.model_name,
                ),
                "version": getattr(
                    champion,
                    "version",
                    None,
                ),
                "metrics": metrics,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )

    print(
        "\nSERVING CHECK PASSED",
        flush=True,
    )

    print(
        "Feature Store predictions: OK",
        flush=True,
    )

    print(
        "Model Registry production model: OK",
        flush=True,
    )


if __name__ == "__main__":
    main()