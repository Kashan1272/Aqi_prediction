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
from aqi_predictor.features import POLLUTANT_COLUMNS, WEATHER_COLUMNS
from aqi_predictor.storage import LocalStore


REPORT_DIR = ROOT / "reports" / "eda"


def save_figure(fig: plt.Figure, name: str) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / name,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def choose_aqi(frame: pd.DataFrame) -> pd.Series:
    """
    Prefer real observed AQI where available.
    Fall back to US AQI when observed sensor AQI is missing.
    """

    output = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    if "observed_us_aqi" in frame.columns:
        output = pd.to_numeric(
            frame["observed_us_aqi"],
            errors="coerce",
        )

    if "us_aqi" in frame.columns:
        fallback = pd.to_numeric(
            frame["us_aqi"],
            errors="coerce",
        )

        output = output.combine_first(
            fallback
        )

    return output


def season_from_month(month: pd.Series) -> pd.Series:
    """
    Match the seasonal definitions used by the AQI feature pipeline.
    """

    result = pd.Series(
        "Other",
        index=month.index,
        dtype="object",
    )

    result.loc[
        month.isin([11, 12, 1, 2])
    ] = "Winter Smog"

    result.loc[
        month.isin([4, 5, 6])
    ] = "Pre-Monsoon"

    result.loc[
        month.isin([7, 8, 9])
    ] = "Monsoon"

    return result


def load_hourly_data() -> pd.DataFrame:
    settings = get_settings()
    store = LocalStore(settings)

    frames: list[pd.DataFrame] = []

    for city_key, city in settings.cities.items():
        frame = store.read_city(city_key)

        if frame.empty:
            print(
                f"Skipping {city_key}: no hourly data",
                flush=True,
            )
            continue

        frame = frame.copy()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        frame["aqi"] = choose_aqi(frame)

        frame = frame.dropna(
            subset=[
                "timestamp",
                "aqi",
            ]
        )

        if frame.empty:
            print(
                f"Skipping {city_key}: no usable AQI rows",
                flush=True,
            )
            continue

        frame["city"] = city_key

        local_time = (
            frame["timestamp"]
            .dt.tz_convert(city.timezone)
        )

        frame["local_date"] = pd.to_datetime(
            local_time.dt.date
        )

        frame["year"] = local_time.dt.year
        frame["month"] = local_time.dt.month
        frame["hour"] = local_time.dt.hour
        frame["weekday"] = local_time.dt.weekday

        frame["season"] = season_from_month(
            frame["month"]
        )

        frames.append(frame)

        print(
            f"Loaded {city_key}: "
            f"{len(frame):,} hourly rows",
            flush=True,
        )

    if not frames:
        raise RuntimeError(
            "No hourly AQI data found. "
            "Run the backfill/hydration pipeline first."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return (
        result
        .sort_values(
            ["city", "timestamp"]
        )
        .reset_index(drop=True)
    )


def build_daily_data(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    candidates = [
        "aqi",
        *POLLUTANT_COLUMNS,
        *WEATHER_COLUMNS,
    ]

    numeric_columns = [
        column
        for column in candidates
        if column in hourly.columns
    ]

    working = hourly[
        [
            "city",
            "local_date",
            *numeric_columns,
        ]
    ].copy()

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    daily = (
        working
        .groupby(
            ["city", "local_date"],
            as_index=False,
        )[numeric_columns]
        .mean()
        .sort_values(
            ["city", "local_date"]
        )
        .reset_index(drop=True)
    )

    daily["year"] = (
        daily["local_date"].dt.year
    )

    daily["month"] = (
        daily["local_date"].dt.month
    )

    daily["season"] = season_from_month(
        daily["month"]
    )

    daily["aqi_daily_change"] = (
        daily
        .groupby("city")["aqi"]
        .diff()
    )

    previous = (
        daily
        .groupby("city")["aqi"]
        .shift(1)
        .replace(0, np.nan)
    )

    daily["aqi_change_rate_pct"] = (
        daily["aqi_daily_change"]
        / previous
        * 100.0
    )

    daily["aqi_change_rate_pct"] = (
        daily["aqi_change_rate_pct"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return daily


def city_analysis(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        daily
        .groupby("city")
        .agg(
            days=("aqi", "count"),
            mean_aqi=("aqi", "mean"),
            median_aqi=("aqi", "median"),
            max_aqi=("aqi", "max"),
            p95_aqi=(
                "aqi",
                lambda values:
                    values.quantile(0.95),
            ),
        )
        .sort_values(
            "mean_aqi",
            ascending=False,
        )
    )

    summary.to_csv(
        REPORT_DIR / "city_aqi_summary.csv"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    summary["mean_aqi"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Average AQI by City"
    )
    ax.set_xlabel("City")
    ax.set_ylabel("Mean AQI")
    ax.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        fig,
        "01_city_mean_aqi.png",
    )

    return summary


def monthly_analysis(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    working = daily.copy()

    working["year_month"] = (
        working["local_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        working
        .groupby("year_month")
        .agg(
            mean_aqi=("aqi", "mean"),
            max_aqi=("aqi", "max"),
            observations=("aqi", "count"),
        )
        .reset_index()
    )

    monthly.to_csv(
        REPORT_DIR / "monthly_aqi.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        monthly["year_month"],
        monthly["mean_aqi"],
        marker="o",
    )

    ax.set_title(
        "Monthly AQI Trend"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQI")
    ax.tick_params(
        axis="x",
        rotation=60,
    )

    save_figure(
        fig,
        "02_monthly_aqi_trend.png",
    )

    return monthly


def seasonal_analysis(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    seasonal = (
        daily
        .groupby("season")
        .agg(
            mean_aqi=("aqi", "mean"),
            median_aqi=("aqi", "median"),
            max_aqi=("aqi", "max"),
            days=("aqi", "count"),
        )
        .sort_values(
            "mean_aqi",
            ascending=False,
        )
    )

    seasonal.to_csv(
        REPORT_DIR / "seasonal_aqi.csv"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    seasonal["mean_aqi"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "AQI by Seasonal Regime"
    )
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean AQI")
    ax.tick_params(
        axis="x",
        rotation=30,
    )

    save_figure(
        fig,
        "03_seasonal_aqi.png",
    )

    return seasonal


def distribution_analysis(
    daily: pd.DataFrame,
) -> None:
    values = (
        pd.to_numeric(
            daily["aqi"],
            errors="coerce",
        )
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=40,
    )

    ax.set_title(
        "Distribution of Daily AQI"
    )
    ax.set_xlabel("AQI")
    ax.set_ylabel("Number of city-days")

    save_figure(
        fig,
        "04_aqi_distribution.png",
    )


def change_rate_analysis(
    daily: pd.DataFrame,
) -> None:
    changes = (
        pd.to_numeric(
            daily["aqi_change_rate_pct"],
            errors="coerce",
        )
        .dropna()
    )

    if changes.empty:
        return

    lower = changes.quantile(0.01)
    upper = changes.quantile(0.99)

    clipped = changes.clip(
        lower=lower,
        upper=upper,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        clipped,
        bins=50,
    )

    ax.set_title(
        "Daily AQI Change Rate"
    )
    ax.set_xlabel(
        "AQI change from previous day (%)"
    )
    ax.set_ylabel("Frequency")

    save_figure(
        fig,
        "05_aqi_change_rate.png",
    )


def missingness_analysis(
    hourly: pd.DataFrame,
) -> pd.Series:
    missing = (
        hourly
        .isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    missing.rename(
        "missing_percent"
    ).to_csv(
        REPORT_DIR / "missingness.csv"
    )

    top = missing.head(20)

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    top.sort_values().plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Top 20 Features by Missing Data"
    )
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")

    save_figure(
        fig,
        "06_missingness_top20.png",
    )

    return missing


def correlation_analysis(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    selected = [
        "aqi",
        *POLLUTANT_COLUMNS,
        *WEATHER_COLUMNS,
    ]

    selected = [
        column
        for column in selected
        if column in hourly.columns
    ]

    numeric = hourly[selected].copy()

    for column in selected:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

    correlation = numeric.corr(
        min_periods=100
    )

    correlation.to_csv(
        REPORT_DIR / "correlation_matrix.csv"
    )

    if correlation.empty:
        return correlation

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    image = ax.imshow(
        correlation.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(correlation.columns)
        )
    )

    ax.set_yticks(
        np.arange(
            len(correlation.index)
        )
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=90,
    )

    ax.set_yticklabels(
        correlation.index
    )

    ax.set_title(
        "AQI, Pollutant and Weather Correlation"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    save_figure(
        fig,
        "07_correlation_heatmap.png",
    )

    return correlation


def relationship_plot(
    hourly: pd.DataFrame,
    feature: str,
    filename: str,
    title: str,
) -> None:
    if feature not in hourly.columns:
        return

    sample = hourly[
        [
            feature,
            "aqi",
        ]
    ].copy()

    sample[feature] = pd.to_numeric(
        sample[feature],
        errors="coerce",
    )

    sample["aqi"] = pd.to_numeric(
        sample["aqi"],
        errors="coerce",
    )

    sample = sample.dropna()

    if sample.empty:
        return

    if len(sample) > 8000:
        sample = sample.sample(
            n=8000,
            random_state=42,
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        sample[feature],
        sample["aqi"],
        alpha=0.20,
        s=10,
    )

    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel("AQI")

    save_figure(
        fig,
        filename,
    )


def write_summary(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    city_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    seasonal: pd.DataFrame,
    missing: pd.Series,
    correlation: pd.DataFrame,
) -> None:
    strongest_positive = None
    strongest_negative = None

    if (
        "aqi" in correlation.columns
        and len(correlation) > 1
    ):
        aqicorr = (
            correlation["aqi"]
            .drop(
                labels=["aqi"],
                errors="ignore",
            )
            .dropna()
            .sort_values()
        )

        if not aqicorr.empty:
            strongest_negative = {
                "feature":
                    str(aqicorr.index[0]),
                "correlation":
                    float(aqicorr.iloc[0]),
            }

            strongest_positive = {
                "feature":
                    str(aqicorr.index[-1]),
                "correlation":
                    float(aqicorr.iloc[-1]),
            }

    unhealthy_share = float(
        (
            pd.to_numeric(
                daily["aqi"],
                errors="coerce",
            )
            >= 151
        ).mean()
        * 100
    )

    worst_city = (
        city_summary.index[0]
        if not city_summary.empty
        else None
    )

    worst_month = None

    if not monthly.empty:
        row = monthly.loc[
            monthly["mean_aqi"].idxmax()
        ]

        worst_month = str(
            row["year_month"]
        )

    worst_season = (
        seasonal.index[0]
        if not seasonal.empty
        else None
    )

    payload = {
        "hourly_rows": int(
            len(hourly)
        ),
        "daily_rows": int(
            len(daily)
        ),
        "cities": sorted(
            hourly["city"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "start_timestamp": str(
            hourly["timestamp"].min()
        ),
        "end_timestamp": str(
            hourly["timestamp"].max()
        ),
        "overall_mean_aqi": float(
            daily["aqi"].mean()
        ),
        "overall_median_aqi": float(
            daily["aqi"].median()
        ),
        "overall_max_aqi": float(
            daily["aqi"].max()
        ),
        "unhealthy_or_worse_percent":
            unhealthy_share,
        "highest_mean_aqi_city":
            worst_city,
        "highest_mean_aqi_month":
            worst_month,
        "highest_mean_aqi_season":
            worst_season,
        "strongest_positive_aqi_correlation":
            strongest_positive,
        "strongest_negative_aqi_correlation":
            strongest_negative,
        "highest_missing_feature":
            (
                str(missing.index[0])
                if not missing.empty
                else None
            ),
        "highest_missing_percent":
            (
                float(missing.iloc[0])
                if not missing.empty
                else None
            ),
    }

    (
        REPORT_DIR
        / "eda_summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Pearls AQI Predictor — EDA Summary",
        "",
        "## Dataset",
        "",
        f"- Hourly rows analyzed: "
        f"{payload['hourly_rows']:,}",
        f"- Daily city observations: "
        f"{payload['daily_rows']:,}",
        f"- Cities analyzed: "
        f"{len(payload['cities'])}",
        f"- Data range: "
        f"{payload['start_timestamp']} "
        f"to {payload['end_timestamp']}",
        "",
        "## Main AQI Findings",
        "",
        f"- Overall mean AQI: "
        f"{payload['overall_mean_aqi']:.2f}",
        f"- Overall median AQI: "
        f"{payload['overall_median_aqi']:.2f}",
        f"- Maximum daily AQI: "
        f"{payload['overall_max_aqi']:.2f}",
        f"- AQI >= 151: "
        f"{payload['unhealthy_or_worse_percent']:.2f}% "
        f"of city-days",
        f"- Highest average AQI city: "
        f"{payload['highest_mean_aqi_city']}",
        f"- Highest average AQI month: "
        f"{payload['highest_mean_aqi_month']}",
        f"- Highest average AQI season: "
        f"{payload['highest_mean_aqi_season']}",
        "",
        "## Relationships",
        "",
    ]

    if strongest_positive:
        lines.append(
            "- Strongest positive AQI correlation: "
            f"{strongest_positive['feature']} "
            f"({strongest_positive['correlation']:.3f})"
        )

    if strongest_negative:
        lines.append(
            "- Strongest negative AQI correlation: "
            f"{strongest_negative['feature']} "
            f"({strongest_negative['correlation']:.3f})"
        )

    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            "- Highest missing feature: "
            f"{payload['highest_missing_feature']} "
            f"({payload['highest_missing_percent']:.2f}%)",
            "",
            "Generated automatically by "
            "`scripts/eda.py`.",
            "",
        ]
    )

    (
        REPORT_DIR
        / "EDA_SUMMARY.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Starting Pearls AQI exploratory data analysis",
        flush=True,
    )

    hourly = load_hourly_data()

    print(
        f"\nTotal hourly rows: {len(hourly):,}",
        flush=True,
    )

    daily = build_daily_data(
        hourly
    )

    daily.to_csv(
        REPORT_DIR / "daily_eda_dataset.csv",
        index=False,
    )

    city_summary = city_analysis(
        daily
    )

    monthly = monthly_analysis(
        daily
    )

    seasonal = seasonal_analysis(
        daily
    )

    distribution_analysis(
        daily
    )

    change_rate_analysis(
        daily
    )

    missing = missingness_analysis(
        hourly
    )

    correlation = correlation_analysis(
        hourly
    )

    relationship_plot(
        hourly,
        "temperature_2m",
        "08_aqi_vs_temperature.png",
        "AQI vs Temperature",
    )

    relationship_plot(
        hourly,
        "relative_humidity_2m",
        "09_aqi_vs_humidity.png",
        "AQI vs Relative Humidity",
    )

    relationship_plot(
        hourly,
        "wind_speed_10m",
        "10_aqi_vs_wind_speed.png",
        "AQI vs Wind Speed",
    )

    relationship_plot(
        hourly,
        "pm2_5",
        "11_aqi_vs_pm25.png",
        "AQI vs PM2.5",
    )

    write_summary(
        hourly,
        daily,
        city_summary,
        monthly,
        seasonal,
        missing,
        correlation,
    )

    print(
        "\nEDA completed successfully.",
        flush=True,
    )

    print(
        f"Reports written to: {REPORT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()