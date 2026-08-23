from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Breakpoint:
    c_low: float
    c_high: float
    i_low: int
    i_high: int


PM25_BREAKPOINTS = [
    Breakpoint(0.0, 9.0, 0, 50),
    Breakpoint(9.1, 35.4, 51, 100),
    Breakpoint(35.5, 55.4, 101, 150),
    Breakpoint(55.5, 125.4, 151, 200),
    Breakpoint(125.5, 225.4, 201, 300),
    Breakpoint(225.5, 325.4, 301, 500),
]
PM10_BREAKPOINTS = [
    Breakpoint(0, 54, 0, 50),
    Breakpoint(55, 154, 51, 100),
    Breakpoint(155, 254, 101, 150),
    Breakpoint(255, 354, 151, 200),
    Breakpoint(355, 424, 201, 300),
    Breakpoint(425, 604, 301, 500),
]


def _subindex(value: float, breakpoints: list[Breakpoint]) -> float:
    if not np.isfinite(value):
        return np.nan
    value = max(0.0, float(value))
    for bp in breakpoints:
        if value <= bp.c_high:
            result = (
                (bp.i_high - bp.i_low)
                / (bp.c_high - bp.c_low)
                * (value - bp.c_low)
                + bp.i_low
            )
            return float(np.clip(round(result), 0, 500))
    return 500.0


def pm25_to_us_aqi(value: float) -> float:
    truncated = np.floor(float(value) * 10) / 10 if np.isfinite(value) else np.nan
    return _subindex(truncated, PM25_BREAKPOINTS)


def pm10_to_us_aqi(value: float) -> float:
    truncated = np.floor(float(value)) if np.isfinite(value) else np.nan
    return _subindex(truncated, PM10_BREAKPOINTS)


def concentration_aqi(pm25: float | None, pm10: float | None) -> float:
    values = [
        pm25_to_us_aqi(float(pm25)) if pm25 is not None else np.nan,
        pm10_to_us_aqi(float(pm10)) if pm10 is not None else np.nan,
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else np.nan


def add_computed_aqi(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "pm2_5" in result:
        result["computed_us_aqi_pm2_5"] = result["pm2_5"].map(pm25_to_us_aqi)
    if "pm10" in result:
        result["computed_us_aqi_pm10"] = result["pm10"].map(pm10_to_us_aqi)
    candidate_columns = [
        column
        for column in (
            "computed_us_aqi_pm2_5",
            "computed_us_aqi_pm10",
            "us_aqi_pm2_5",
            "us_aqi_pm10",
        )
        if column in result
    ]
    if "us_aqi" not in result and candidate_columns:
        result["us_aqi"] = result[candidate_columns].max(axis=1, skipna=True)
    return result


def category(aqi: float | int | None) -> str:
    if aqi is None or not np.isfinite(float(aqi)):
        return "Unavailable"
    value = float(aqi)
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for Sensitive Groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def health_message(aqi: float | int | None) -> str:
    label = category(aqi)
    return {
        "Good": "Air quality is satisfactory for normal outdoor activity.",
        "Moderate": "Unusually sensitive people should consider reducing prolonged exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should limit prolonged outdoor exertion.",
        "Unhealthy": "Everyone should reduce prolonged or heavy outdoor exertion.",
        "Very Unhealthy": "Avoid strenuous outdoor activity; keep windows closed where possible.",
        "Hazardous": "Remain indoors, use filtration, and follow local health guidance.",
        "Unavailable": "A health recommendation is unavailable until data is refreshed.",
    }[label]
