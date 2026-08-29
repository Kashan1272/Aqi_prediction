from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AQI_PROJECT_ROOT", str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aqi_predictor.config import get_settings
from aqi_predictor.city_selection import load_selected_city_keys
from aqi_predictor.inference import forecast_city
from aqi_predictor.registry import LocalModelRegistry
from aqi_predictor.storage import LocalStore
from aqi_predictor.aqi import category, health_message

settings = get_settings()
store = LocalStore(settings)

def render_aqi_alert(daily_forecast: list[dict]) -> None:
    if not daily_forecast:
        st.info("AQI alert status is unavailable.")
        return

    valid_days = []

    for day in daily_forecast:
        values = []

        for key in ("aqi_mean", "aqi_max"):
            try:
                value = float(day.get(key))
            except (TypeError, ValueError):
                continue

            if pd.notna(value):
                values.append(value)

        if values:
            valid_days.append((day, max(values)))

    if not valid_days:
        st.info("No valid AQI values are available for alerts.")
        return

    worst_day, worst_aqi = max(
        valid_days,
        key=lambda item: item[1],
    )

    label = category(worst_aqi)
    advice = health_message(worst_aqi)

    horizon = worst_day.get("horizon_day", "—")
    date = worst_day.get("date", "upcoming forecast")

    message = (
        f"Day {horizon} · {date} — "
        f"forecast peak {worst_aqi:.0f} AQI "
        f"({label}). {advice}"
    )

    if worst_aqi >= 301:
        st.error("🚨 HAZARDOUS AIR QUALITY ALERT\n\n" + message)

    elif worst_aqi >= 201:
        st.error("🔴 VERY UNHEALTHY AIR QUALITY ALERT\n\n" + message)

    elif worst_aqi >= 151:
        st.error("⚠️ UNHEALTHY AIR QUALITY ALERT\n\n" + message)

    elif worst_aqi >= 101:
        st.warning(
            "⚠️ AIR QUALITY ALERT FOR SENSITIVE GROUPS\n\n"
            + message
        )

    else:
        st.success(
            "✅ No unhealthy AQI alert in the current "
            "three-day forecast.\n\n"
            + message
        )

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 1.2rem;}
      .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0b172a 0%, #123b54 52%, #0f766e 100%);
        color: white;
        box-shadow: 0 20px 50px rgba(15, 23, 42, .18);
        margin-bottom: 1rem;
      }
      .hero h1 {font-size: 2.1rem; margin: 0 0 .3rem 0;}
      .hero p {opacity: .86; margin: 0;}
      .day-card {
        padding: 1.1rem;
        border: 1px solid rgba(148, 163, 184, .24);
        border-radius: 20px;
        background: rgba(255,255,255,.75);
        box-shadow: 0 12px 35px rgba(15, 23, 42, .08);
        min-height: 190px;
      }
      .day-card .value {font-size: 2rem; font-weight: 750;}
      .day-card .peak {font-size: .92rem; opacity: .75;}
      .small-note {font-size: .82rem; opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)

city_options = load_selected_city_keys(settings)
selected = st.sidebar.selectbox(
    "Pakistan city",
    city_options,
    index=city_options.index(settings.default_city) if settings.default_city in city_options else 0,
    format_func=lambda key: f"{settings.city(key).name} — {settings.city(key).province}",
)
refresh = st.sidebar.button("Refresh live forecast", use_container_width=True)
show_sources = st.sidebar.toggle("Show provider diagnostics", value=False)

payload = store.load_prediction(selected)
if refresh or not payload:
    with st.spinner("Fetching providers and generating the three-day forecast…"):
        try:
            payload = forecast_city(settings, selected)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

city = payload["city"]
st.markdown(
    f"""
    <div class="hero">
      <h1>Pearls AQI Predictor</h1>
      <p>{city['name']}, {city['province']} · Daily mean and peak AQI for the next three days · Calibrated 72-hour curve</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = payload.get("model", {})
current = payload.get("current_observations", {})
top = st.columns(5)
top[0].metric("Open-Meteo current AQI", current.get("open_meteo_aqi") or "—")
top[1].metric("OpenAQ observed AQI", (current.get("openaq") or {}).get("computed_us_aqi") or "—")
top[2].metric("Sensor bias applied", current.get("sensor_bias_applied", 0))
mean_metrics = (model.get("test_metrics") or {}).get("daily_mean") or {}
top[3].metric("Untouched test R²", f"{mean_metrics.get('r2', float('nan')):.3f}" if mean_metrics else "—")
top[4].metric("Untouched test RMSE", f"{mean_metrics.get('rmse', float('nan')):.2f}" if mean_metrics else "—")

daily = payload.get("daily_forecast") or []
render_aqi_alert(daily)
st.write("")
columns = st.columns(3)
for column, day in zip(columns, daily, strict=False):
    column.markdown(
        f"""
        <div class="day-card">
          <div class="small-note">Day {day['horizon_day']} · {day['date']}</div>
          <div class="value">{day['aqi_mean']:.0f} AQI</div>
          <div>{day['category']}</div>
          <div class="peak">Expected peak: {day['aqi_max']:.0f} · Interval: {day['aqi_mean_lower']:.0f}–{day['aqi_mean_upper']:.0f}</div>
          <hr/>
          <div class="small-note">{day['health_message']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hourly = pd.DataFrame(payload.get("hourly_forecast") or [])
if not hourly.empty:
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["timestamp"],
        y=hourly["aqi"],
        mode="lines",
        name="Calibrated AQI",
        line={"width": 3},
        fill="tozeroy",
        fillcolor="rgba(14, 116, 144, .08)",
    ))
    fig.update_layout(
        title="Calibrated 72-hour AQI profile",
        xaxis_title=None,
        yaxis_title="US AQI",
        height=390,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    provider_rows = []
    for day in daily:
        for provider, values in day.get("contributions", {}).items():
            provider_rows.append({
                "date": day["date"],
                "provider": provider,
                "mean": values.get("mean"),
                "max": values.get("max"),
            })
    provider_df = pd.DataFrame(provider_rows)
    if not provider_df.empty:
        fig = px.bar(
            provider_df,
            x="date",
            y="mean",
            color="provider",
            barmode="group",
            title="Provider and ML contributions",
        )
        fig.update_layout(height=360, margin={"l": 20, "r": 20, "t": 55, "b": 20})
        st.plotly_chart(fig, use_container_width=True)

with right:
    selected_algorithms = model.get("selected_algorithms") or {}
    st.subheader("Production model")
    st.write("**Target:** daily mean and daily peak US AQI for Day 1, Day 2 and Day 3.")
    st.json(selected_algorithms, expanded=False)
    st.caption(
        "The hourly chart is not used to inflate the model score. It is a provider forecast "
        "calibrated to the independently evaluated daily ML predictions."
    )

if show_sources:
    st.subheader("Provider diagnostics")
    st.json(payload.get("provider_health") or {}, expanded=False)
