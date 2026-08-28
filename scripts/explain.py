from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import ROOT

from aqi_predictor.config import get_settings
from aqi_predictor.explainability import explain_pipeline
from aqi_predictor.features import (
    add_provider_snapshot_features,
    build_national_training_frame,
)
from aqi_predictor.models import chronological_partitions
from aqi_predictor.registry import LocalModelRegistry
from aqi_predictor.storage import LocalStore


REPORT_DIR = ROOT / "reports" / "explainability"


def rebuild_training_frame(
    settings,
    store: LocalStore,
    city_keys: list[str],
) -> pd.DataFrame:
    histories = []

    for key in city_keys:
        frame = store.read_city(key)

        if frame.empty:
            print(
                f"Skipping {key}: no hourly data",
                flush=True,
            )
            continue

        histories.append(
            (
                settings.city(key),
                frame,
            )
        )

    if not histories:
        raise RuntimeError(
            "No historical city data is available."
        )

    training = build_national_training_frame(
        histories,
        forecast_days=settings.forecast_days,
    )

    # ----------------------------------------------------------
    # Reproduce provider snapshot merge used by training.py
    # ----------------------------------------------------------
    snapshot_frames: list[pd.DataFrame] = []

    for key in city_keys:
        snapshot = store.read_provider_snapshots(key)

        if snapshot.empty:
            continue

        snapshot = snapshot.copy()
        snapshot["city"] = key

        snapshot_frames.append(snapshot)

    if snapshot_frames:
        snapshots = pd.concat(
            snapshot_frames,
            ignore_index=True,
            copy=False,
            sort=False,
        )

        snapshots["issue_date"] = (
            pd.to_datetime(
                snapshots["issue_date"],
                utc=True,
                errors="coerce",
            )
            .dt.tz_localize(None)
            .dt.normalize()
        )

        snapshots["horizon_day"] = pd.to_numeric(
            snapshots["horizon_day"],
            errors="coerce",
        ).astype("Int64")

        snapshots = snapshots.drop(
            columns=[
                "target_date",
                "collected_at",
            ],
            errors="ignore",
        )

        training = training.merge(
            snapshots,
            on=[
                "city",
                "issue_date",
                "horizon_day",
            ],
            how="left",
            validate="many_to_one",
        )

    training = add_provider_snapshot_features(
        training
    )

    return (
        training
        .sort_values(
            [
                "issue_date",
                "city",
                "horizon_day",
            ]
        )
        .reset_index(drop=True)
    )


def save_importance_plot(
    frame: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    if frame.empty:
        return

    top = (
        frame
        .head(20)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        top["feature"],
        top["importance"],
    )

    ax.set_title(title)
    ax.set_xlabel(
        "Mean absolute SHAP value"
    )
    ax.set_ylabel("Feature")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def explain_horizon(
    bundle,
    test: pd.DataFrame,
    *,
    horizon_day: int,
    max_rows: int,
    random_seed: int,
) -> dict:
    key = f"day{horizon_day}_mean"

    stacking_model = bundle.models.get(key)

    if stacking_model is None:
        raise KeyError(
            f"Production bundle does not contain {key}"
        )

    dominant = (
        stacking_model.dominant_base_model
    )

    if not dominant:
        raise RuntimeError(
            f"No dominant model exists for {key}"
        )

    explanation_model = (
        stacking_model.base_models.get(
            dominant
        )
    )

    if explanation_model is None:
        raise RuntimeError(
            f"Dominant base model {dominant!r} "
            f"is missing for {key}"
        )

    subset = test[
        pd.to_numeric(
            test["horizon_day"],
            errors="coerce",
        )
        == horizon_day
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No test rows found for horizon {horizon_day}"
        )

    feature_columns = (
        list(bundle.numeric_features)
        + list(bundle.categorical_features)
    )

    missing = [
        column
        for column in feature_columns
        if column not in subset.columns
    ]

    if missing:
        raise ValueError(
            "Explainability matrix is missing features: "
            + ", ".join(missing[:20])
        )

    explanation_frame = subset[
        feature_columns
    ].copy()

    target = pd.to_numeric(
        subset["target_aqi_mean"],
        errors="coerce",
    )

    valid = target.notna()

    explanation_frame = (
        explanation_frame.loc[valid]
    )

    target = target.loc[valid]

    report = explain_pipeline(
        explanation_model,
        explanation_frame,
        target,
        max_rows=max_rows,
        random_seed=random_seed,
    )

    report["model_key"] = key
    report["horizon_day"] = horizon_day
    report["target"] = "daily mean AQI"
    report["explained_component"] = (
        dominant
    )
    report["ensemble_weights"] = (
        stacking_model.weights
    )

    return report


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Starting production-model SHAP analysis",
        flush=True,
    )

    settings = get_settings()

    store = LocalStore(settings)

    registry = LocalModelRegistry(
        settings
    )

    bundle, production_report, version_dir = (
        registry.load_production()
    )

    print(
        f"Production model: {version_dir.name}",
        flush=True,
    )

    # Use exactly the cities recorded in the
    # production model when possible.
    city_keys = list(
        bundle.metadata.get(
            "cities",
            [],
        )
    )

    if not city_keys:
        city_keys = list(
            settings.cities
        )

    city_keys = [
        key
        for key in city_keys
        if key in settings.cities
    ]

    print(
        "Production cities:",
        ", ".join(city_keys),
        flush=True,
    )

    training = rebuild_training_frame(
        settings,
        store,
        city_keys,
    )

    print(
        f"Training matrix rows rebuilt: "
        f"{len(training):,}",
        flush=True,
    )

    _, _, test = chronological_partitions(
        training
    )

    print(
        f"Explainability test rows: "
        f"{len(test):,}",
        flush=True,
    )

    reports: dict[str, dict] = {}

    for horizon_day in (
        1,
        settings.forecast_days,
    ):
        print(
            f"\nExplaining Day {horizon_day} mean AQI...",
            flush=True,
        )

        report = explain_horizon(
            bundle,
            test,
            horizon_day=horizon_day,
            max_rows=min(
                settings.explainability_sample_rows,
                180,
            ),
            random_seed=(
                settings.random_seed
                + horizon_day
            ),
        )

        method = str(
            report.get(
                "method",
                "unknown",
            )
        )

        print(
            f"Method: {method}",
            flush=True,
        )

        if method != "shap":
            raise RuntimeError(
                "SHAP was not used. "
                f"Explainability returned {method!r}. "
                "Confirm the shap package is installed."
            )

        importance = pd.DataFrame(
            report.get(
                "features",
                [],
            )
        )

        if importance.empty:
            raise RuntimeError(
                f"No SHAP features produced "
                f"for Day {horizon_day}."
            )

        importance = (
            importance
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        importance.to_csv(
            REPORT_DIR
            / (
                f"shap_day{horizon_day}"
                "_importance.csv"
            ),
            index=False,
        )

        save_importance_plot(
            importance,
            (
                f"Day {horizon_day} AQI "
                "SHAP Feature Importance"
            ),
            (
                f"shap_day{horizon_day}"
                "_importance.png"
            ),
        )

        reports[
            f"day{horizon_day}_mean"
        ] = report

        print(
            "\nTop SHAP features:",
            flush=True,
        )

        print(
            importance.head(10).to_string(
                index=False
            ),
            flush=True,
        )

    payload = {
        "production_version":
            version_dir.name,
        "model_name":
            settings.model_name,
        "method":
            "SHAP",
        "explained_targets": [
            "day1_mean",
            f"day{settings.forecast_days}_mean",
        ],
        "production_metrics":
            production_report.get(
                "test_metrics",
                {},
            ).get(
                "daily_mean",
                {},
            ),
        "reports":
            reports,
    }

    (
        REPORT_DIR
        / "shap_report.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Pearls AQI Predictor — SHAP Explainability",
        "",
        f"- Production version: `{version_dir.name}`",
        f"- Model name: `{settings.model_name}`",
        "- Explainability method: SHAP",
        "- Targets explained: Day 1 mean AQI and Day 3 mean AQI",
        "",
    ]

    for key, report in reports.items():
        lines.extend(
            [
                f"## {key}",
                "",
                f"- Dominant ensemble component: "
                f"`{report.get('explained_component')}`",
                "",
                "### Top features",
                "",
            ]
        )

        for item in report.get(
            "features",
            [],
        )[:10]:
            lines.append(
                "- "
                f"{item['feature']}: "
                f"{item['importance']:.6f}"
            )

        lines.append("")

    (
        REPORT_DIR
        / "SHAP_SUMMARY.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "\nSHAP analysis completed successfully.",
        flush=True,
    )

    print(
        f"Reports written to: {REPORT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()