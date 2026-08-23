from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .config import City

LOGGER = logging.getLogger(__name__)

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]
POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
]
TARGET_COLUMNS = ["target_aqi_mean", "target_aqi_max"]
SENSOR_TARGET_COLUMNS = ["target_observed_aqi_mean", "target_observed_aqi_max"]
CATEGORICAL_FEATURES = ["city", "province"]
IDENTIFIER_COLUMNS = ["issue_date", "target_date", "horizon_day", "city", "province"]


def _safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def _rolling_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Leakage-safe rolling linear slope using observations before issue time."""
    shifted = pd.to_numeric(series, errors="coerce").shift(1)

    def slope(values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        valid = np.isfinite(array)
        if int(valid.sum()) < min_periods:
            return np.nan
        x = np.arange(len(array), dtype=float)[valid]
        y = array[valid]
        x = x - x.mean()
        denominator = float(np.dot(x, x))
        return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else 0.0

    return shifted.rolling(window, min_periods=min_periods).apply(slope, raw=True)


def _daily_aggregate(hourly: pd.DataFrame, city: City) -> pd.DataFrame:
    if hourly.empty or "timestamp" not in hourly:
        return pd.DataFrame()
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    local = frame["timestamp"].dt.tz_convert(city.timezone)
    frame["local_date"] = local.dt.date
    aggregations: dict[str, list[str] | str] = {}
    if "us_aqi" in frame:
        aggregations["us_aqi"] = ["mean", "max", "min", "std", "median"]
    if "observed_us_aqi" in frame:
        aggregations["observed_us_aqi"] = ["mean", "max", "count"]
    for column in POLLUTANT_COLUMNS:
        if column in frame:
            aggregations[column] = ["mean", "max"]
    for column in WEATHER_COLUMNS:
        if column not in frame:
            continue
        if column in {"precipitation", "rain", "shortwave_radiation"}:
            aggregations[column] = ["mean", "max", "sum"]
        else:
            aggregations[column] = ["mean", "min", "max"]
    for lead in (1, 2, 3):
        for column in WEATHER_COLUMNS:
            name = f"lead_{lead}d_{column}"
            if name not in frame:
                continue
            if column in {"precipitation", "rain", "shortwave_radiation"}:
                aggregations[name] = ["mean", "max", "sum"]
            else:
                aggregations[name] = ["mean", "min", "max"]
    grouped = frame.groupby("local_date", observed=True).agg(aggregations)
    grouped.columns = ["__".join(item) for item in grouped.columns]
    daily = grouped.reset_index().rename(columns={"local_date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["hours"] = frame.groupby("local_date", observed=True).size().to_numpy()
    return daily.sort_values("date").reset_index(drop=True)


def _add_history_features(daily: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        column for column in daily.columns
        if column.startswith("us_aqi__")
        or any(column.startswith(f"{pollutant}__") for pollutant in POLLUTANT_COLUMNS)
    ]
    core_distribution_columns = {
        "us_aqi__mean",
        "us_aqi__max",
        "pm2_5__mean",
        "pm2_5__max",
        "pm10__mean",
        "pm10__max",
        "nitrogen_dioxide__mean",
        "ozone__mean",
    }
    engineered: dict[str, pd.Series] = {}
    for column in base_columns:
        series = pd.to_numeric(daily[column], errors="coerce")
        shifted = series.shift(1)
        for lag in (1, 2, 3, 7, 14, 21, 28):
            engineered[f"{column}__lag{lag}"] = series.shift(lag)
        for window in (3, 7, 14, 21, 28):
            minimum = max(2, window // 2)
            rolling = shifted.rolling(window, min_periods=minimum)
            engineered[f"{column}__roll{window}_mean"] = rolling.mean()
            engineered[f"{column}__roll{window}_std"] = rolling.std()
        engineered[f"{column}__trend3"] = series - series.shift(3)
        engineered[f"{column}__trend7"] = series - series.shift(7)
        engineered[f"{column}__trend14"] = series - series.shift(14)

        if column in core_distribution_columns:
            for window in (7, 14, 28):
                minimum = max(3, window // 2)
                rolling = shifted.rolling(window, min_periods=minimum)
                engineered[f"{column}__roll{window}_median"] = rolling.median()
                engineered[f"{column}__roll{window}_q25"] = rolling.quantile(0.25)
                engineered[f"{column}__roll{window}_q75"] = rolling.quantile(0.75)
            for span in (3, 7, 14, 28):
                engineered[f"{column}__ewm{span}"] = shifted.ewm(
                    span=span, adjust=False, min_periods=max(2, span // 3)
                ).mean()
            weekday_analogs = pd.concat(
                [series.shift(lag) for lag in (7, 14, 21, 28)], axis=1
            )
            engineered[f"{column}__same_weekday_4w_mean"] = weekday_analogs.mean(axis=1)
            engineered[f"{column}__same_weekday_4w_std"] = weekday_analogs.std(axis=1)
            engineered[f"{column}__recent_vs_28d"] = (
                shifted.rolling(3, min_periods=2).mean()
                - shifted.rolling(28, min_periods=14).mean()
            )

    # Issue-time weather history is required to compare the target-day forecast
    # with recent local conditions. All rolling values are shifted by one day,
    # so target-day weather and future observations cannot leak into training.
    weather_history_columns = [
        "temperature_2m__mean",
        "relative_humidity_2m__mean",
        "dew_point_2m__mean",
        "surface_pressure__mean",
        "wind_speed_10m__mean",
        "wind_direction_10m__mean",
        "rain__sum",
        "precipitation__sum",
    ]
    for column in weather_history_columns:
        if column not in daily:
            continue
        series = pd.to_numeric(daily[column], errors="coerce")
        shifted = series.shift(1)
        engineered[f"{column}__lag1"] = shifted
        engineered[f"{column}__lag3"] = series.shift(3)
        for window in (3, 7, 14):
            rolling = shifted.rolling(window, min_periods=max(2, window // 2))
            engineered[f"{column}__roll{window}_mean"] = rolling.mean()
            engineered[f"{column}__roll{window}_std"] = rolling.std()
        engineered[f"{column}__trend3"] = shifted - series.shift(4)
        engineered[f"{column}__trend7"] = shifted - series.shift(8)

    if "us_aqi__mean" in daily:
        mean_aqi = pd.to_numeric(daily["us_aqi__mean"], errors="coerce")
        engineered["aqi_mean_change_1d"] = mean_aqi - mean_aqi.shift(1)
        engineered["aqi_mean_change_7d"] = mean_aqi - mean_aqi.shift(7)
        engineered["aqi_high_days_14"] = (mean_aqi.shift(1) >= 151).rolling(14, min_periods=7).sum()
        engineered["aqi_unhealthy_days_28"] = (mean_aqi.shift(1) >= 101).rolling(28, min_periods=14).sum()
        engineered["aqi_slope_7"] = _rolling_slope(mean_aqi, 7, 4)
        engineered["aqi_slope_14"] = _rolling_slope(mean_aqi, 14, 7)
        engineered["aqi_slope_28"] = _rolling_slope(mean_aqi, 28, 14)
        engineered["aqi_slope_acceleration"] = (
            engineered["aqi_slope_7"] - engineered["aqi_slope_14"]
        )
        lag1 = mean_aqi.shift(1)
        weekday_mean = pd.concat(
            [mean_aqi.shift(lag) for lag in (7, 14, 21, 28)], axis=1
        ).mean(axis=1)
        engineered["aqi_persistence_blend"] = (
            0.55 * lag1
            + 0.25 * mean_aqi.shift(2)
            + 0.20 * weekday_mean
        )
        engineered["aqi_day3_trend_projection"] = (
            lag1 + 3.0 * engineered["aqi_slope_7"]
        )
        month_group = pd.to_datetime(daily["date"], errors="coerce").dt.month
        weekday_group = pd.to_datetime(daily["date"], errors="coerce").dt.weekday
        engineered["aqi_city_month_climatology"] = mean_aqi.groupby(month_group).transform(
            lambda values: values.shift(1).expanding(min_periods=2).mean()
        )
        engineered["aqi_city_weekday_climatology"] = mean_aqi.groupby(weekday_group).transform(
            lambda values: values.shift(1).expanding(min_periods=3).mean()
        )
        engineered["aqi_vs_month_climatology"] = (
            mean_aqi.shift(1) - engineered["aqi_city_month_climatology"]
        )
    if "us_aqi__max" in daily and "us_aqi__mean" in daily:
        max_aqi = pd.to_numeric(daily["us_aqi__max"], errors="coerce")
        mean_aqi = pd.to_numeric(daily["us_aqi__mean"], errors="coerce")
        engineered["aqi_peak_excess"] = max_aqi - mean_aqi
        engineered["aqi_peak_excess_roll14"] = (max_aqi - mean_aqi).shift(1).rolling(14, min_periods=7).mean()

    if not engineered:
        return daily.copy()
    return pd.concat([daily.reset_index(drop=True), pd.DataFrame(engineered)], axis=1)


def _season_features(target_date: pd.Series) -> dict[str, pd.Series]:
    day_of_year = target_date.dt.dayofyear.astype(float)
    weekday = target_date.dt.weekday.astype(float)
    month = target_date.dt.month.astype(float)
    return {
        "target_month": month,
        "target_weekday": weekday,
        "target_day_of_year": day_of_year,
        "target_month_sin": np.sin(2 * np.pi * month / 12.0),
        "target_month_cos": np.cos(2 * np.pi * month / 12.0),
        "target_doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "target_doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "target_weekday_sin": np.sin(2 * np.pi * weekday / 7.0),
        "target_weekday_cos": np.cos(2 * np.pi * weekday / 7.0),
        "is_winter_smog": month.isin([11, 12, 1, 2]).astype("int8"),
        "is_monsoon": month.isin([7, 8, 9]).astype("int8"),
        "is_pre_monsoon": month.isin([4, 5, 6]).astype("int8"),
    }




def _add_future_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free horizon-aware weather and AQI trajectory interactions."""
    result = frame.copy()

    def numeric(name: str) -> pd.Series:
        if name not in result:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[name], errors="coerce")

    temperature = numeric("future_temperature_2m__mean")
    humidity = numeric("future_relative_humidity_2m__mean")
    dew_point = numeric("future_dew_point_2m__mean")
    wind_speed = numeric("future_wind_speed_10m__mean")
    wind_direction = numeric("future_wind_direction_10m__mean")
    rain = numeric("future_rain__sum")
    precipitation = numeric("future_precipitation__sum")
    pressure = numeric("future_surface_pressure__mean")
    horizon = numeric("horizon_day").fillna(1.0).clip(lower=1.0, upper=7.0)

    result["future_dewpoint_depression"] = temperature - dew_point
    result["future_temperature_humidity"] = temperature * humidity / 100.0
    radians = np.deg2rad(wind_direction)
    result["future_wind_u"] = wind_speed * np.cos(radians)
    result["future_wind_v"] = wind_speed * np.sin(radians)
    result["future_wind_direction_sin"] = np.sin(radians)
    result["future_wind_direction_cos"] = np.cos(radians)
    result["future_ventilation_proxy"] = wind_speed * (100.0 - humidity).clip(lower=0.0)
    result["future_rain_washout"] = (
        rain.fillna(0.0) + precipitation.fillna(0.0)
    ) * wind_speed.fillna(0.0)
    result["future_stagnation_indicator"] = (
        (wind_speed < 2.0) & (humidity > 65.0)
    ).astype("int8")
    result["future_heat_dust_index"] = (
        temperature.clip(lower=0.0)
        * (100.0 - humidity).clip(lower=0.0)
        / 100.0
    )

    historical_pressure = numeric("surface_pressure__mean__roll7_mean")
    historical_pressure = historical_pressure.where(
        historical_pressure.notna(),
        numeric("surface_pressure__mean__lag1"),
    )
    result["future_pressure_anomaly"] = pressure - historical_pressure

    current_temperature = numeric("temperature_2m__mean__lag1")
    current_humidity = numeric("relative_humidity_2m__mean__lag1")
    current_wind = numeric("wind_speed_10m__mean__lag1")
    result["future_temperature_change"] = temperature - current_temperature
    result["future_humidity_change"] = humidity - current_humidity
    result["future_wind_change"] = wind_speed - current_wind
    result["future_weather_shift_magnitude"] = (
        result["future_temperature_change"].abs()
        + result["future_humidity_change"].abs() / 10.0
        + result["future_wind_change"].abs()
    )

    aqi_lag1 = numeric("us_aqi__mean__lag1")
    slope7 = numeric("aqi_slope_7")
    slope14 = numeric("aqi_slope_14")
    result["aqi_horizon_trend_projection"] = aqi_lag1 + horizon * slope7
    result["aqi_horizon_trend_consensus"] = (
        aqi_lag1 + horizon * (0.65 * slope7 + 0.35 * slope14)
    )
    result["aqi_trend_disagreement"] = (slope7 - slope14).abs()
    result["day3_pollution_persistence_risk"] = (
        numeric("aqi_unhealthy_days_28").fillna(0.0) / 28.0
        + numeric("future_stagnation_indicator").fillna(0.0)
        + (numeric("future_rain_washout").fillna(0.0) <= 0.1).astype(float)
    ) * (horizon >= 3).astype(float)
    return result


def add_provider_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create issue-time provider consensus features used by long horizons."""
    result = frame.copy()
    mean_columns = [
        column for column in (
            "provider_open_meteo_aqi_mean",
            "provider_openweather_aqi_mean",
            "provider_aqicn_aqi_mean",
        ) if column in result
    ]
    max_columns = [
        column for column in (
            "provider_open_meteo_aqi_max",
            "provider_openweather_aqi_max",
            "provider_aqicn_aqi_max",
        ) if column in result
    ]
    if mean_columns:
        values = result[mean_columns].apply(pd.to_numeric, errors="coerce")
        result["provider_mean_consensus"] = values.mean(axis=1)
        result["provider_mean_spread"] = values.std(axis=1)
        result["provider_mean_range"] = values.max(axis=1) - values.min(axis=1)
        result["provider_mean_available"] = values.notna().sum(axis=1).astype("int8")
    if max_columns:
        values = result[max_columns].apply(pd.to_numeric, errors="coerce")
        result["provider_max_consensus"] = values.mean(axis=1)
        result["provider_max_spread"] = values.std(axis=1)
        result["provider_max_range"] = values.max(axis=1) - values.min(axis=1)
    if "provider_open_meteo_current_aqi" in result and "provider_openaq_current_aqi" in result:
        result["provider_current_sensor_bias"] = (
            pd.to_numeric(result["provider_openaq_current_aqi"], errors="coerce")
            - pd.to_numeric(result["provider_open_meteo_current_aqi"], errors="coerce")
        )
    return result

def build_city_training_frame(
    hourly: pd.DataFrame,
    city: City,
    *,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Create one leakage-controlled row per issue date and forecast day.

    The primary target is daily mean AQI and the secondary target is daily peak
    AQI. This directly matches the project requirement to forecast the next
    three days and avoids judging a 72-hour system on noisy individual hours.
    """
    daily = _add_history_features(_daily_aggregate(hourly, city))
    if daily.empty or "us_aqi__mean" not in daily or "us_aqi__max" not in daily:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    for horizon in range(1, forecast_days + 1):
        chunk = daily.copy()
        chunk["issue_date"] = chunk["date"]
        chunk["target_date"] = chunk["date"] + pd.to_timedelta(horizon, unit="D")
        chunk["horizon_day"] = horizon
        chunk["target_aqi_mean"] = daily["us_aqi__mean"].shift(-horizon)
        chunk["target_aqi_max"] = daily["us_aqi__max"].shift(-horizon)
        if "observed_us_aqi__mean" in daily:
            observed_count = daily.get(
                "observed_us_aqi__count",
                pd.Series(0, index=daily.index),
            )
            reliable = observed_count >= 12
            chunk["target_observed_aqi_mean"] = daily["observed_us_aqi__mean"].where(reliable).shift(-horizon)
            chunk["target_observed_aqi_max"] = daily["observed_us_aqi__max"].where(reliable).shift(-horizon)

        # At target time, lead_hd_* contains the weather forecast that was
        # available h days before the target, which matches the issue date.
        for column in list(daily.columns):
            prefix = f"lead_{horizon}d_"
            if column.startswith(prefix):
                chunk[f"future_{column[len(prefix):]}"] = daily[column].shift(-horizon)

        for name, values in _season_features(chunk["target_date"]).items():
            chunk[name] = values

        chunk["city"] = city.key
        chunk["province"] = city.province
        chunk["latitude"] = city.latitude
        chunk["longitude"] = city.longitude
        chunk["elevation_m"] = city.elevation_m
        chunk = _add_future_interactions(chunk)
        chunks.append(chunk)

    training = pd.concat(chunks, ignore_index=True, copy=False, sort=False)
    training = training.drop(columns=["date"], errors="ignore")
    training = training.dropna(subset=TARGET_COLUMNS)
    training = training[
        (training["hours"].fillna(0) >= 18)
        & training["target_aqi_mean"].between(0, 500)
        & training["target_aqi_max"].between(0, 500)
    ]
    return training.sort_values(["issue_date", "horizon_day"]).reset_index(drop=True)


def build_national_training_frame(
    histories: Iterable[tuple[City, pd.DataFrame]],
    *,
    forecast_days: int = 3,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for city, frame in histories:
        LOGGER.info("Building daily training rows for %s", city.key)
        city_frame = build_city_training_frame(frame, city, forecast_days=forecast_days)
        if not city_frame.empty:
            chunks.append(city_frame)
    if not chunks:
        raise ValueError("No valid city training rows were generated")
    result = pd.concat(chunks, ignore_index=True, copy=False, sort=False)
    return result.sort_values(["issue_date", "city", "horizon_day"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS + SENSOR_TARGET_COLUMNS + ["timestamp"])
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    numeric: list[str] = []
    for column in frame.columns:
        if column in excluded or column in categorical:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            # Exclude direct target-day realized weather and future AQI.
            if column.startswith("lead_"):
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if not values.notna().any():
                continue
            numeric.append(column)
    return numeric, categorical


def latest_live_feature_rows(
    hourly_history: pd.DataFrame,
    future_weather: pd.DataFrame,
    city: City,
    *,
    forecast_days: int = 3,
) -> pd.DataFrame:
    daily_history = _add_history_features(_daily_aggregate(hourly_history, city))
    if daily_history.empty:
        raise ValueError(f"No historical daily data is available for {city.key}")
    # Match the training contract: issue-day aggregates require at least
    # 18 completed local hours. Before that point, forecast from the latest
    # sufficiently complete day instead of a partial-day distribution.
    eligible = daily_history[daily_history["hours"].fillna(0) >= 18]
    issue = (eligible if not eligible.empty else daily_history).iloc[-1:].copy()
    issue_date = pd.Timestamp(issue["date"].iloc[0])

    weather = future_weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    weather["target_date"] = weather["timestamp"].dt.tz_convert(city.timezone).dt.normalize().dt.tz_localize(None)
    weather_daily = weather.groupby("target_date", observed=True).agg({
        column: ["mean", "min", "max"] if column not in {"precipitation", "rain", "shortwave_radiation"}
        else ["mean", "max", "sum"]
        for column in WEATHER_COLUMNS if column in weather
    })
    weather_daily.columns = ["__".join(item) for item in weather_daily.columns]
    weather_daily = weather_daily.reset_index()

    rows: list[pd.DataFrame] = []
    for horizon in range(1, forecast_days + 1):
        row = issue.copy()
        target_date = issue_date + pd.Timedelta(days=horizon)
        row["issue_date"] = issue_date
        row["target_date"] = target_date
        row["horizon_day"] = horizon
        match = weather_daily[weather_daily["target_date"] == target_date]
        if not match.empty:
            for column in match.columns:
                if column != "target_date":
                    row[f"future_{column}"] = match[column].iloc[0]
        for name, values in _season_features(pd.Series([target_date])).items():
            row[name] = values.iloc[0]
        row["city"] = city.key
        row["province"] = city.province
        row["latitude"] = city.latitude
        row["longitude"] = city.longitude
        row["elevation_m"] = city.elevation_m
        rows.append(row)
    result = pd.concat(rows, ignore_index=True, copy=False, sort=False)
    result = _add_future_interactions(result)
    return result.drop(columns=["date"], errors="ignore")
