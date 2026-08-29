from __future__ import annotations

import os
import sys
from html import escape
from pathlib import Path

import streamlit as st

# IMPORTANT: this must be the first Streamlit command in the file.
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AQI_PROJECT_ROOT", str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

# Local development uses the private project .env file.
load_dotenv(ROOT / ".env", override=False)

# Streamlit Community Cloud uses st.secrets. We only read it when
# the local environment does not already contain the Hopsworks key.
SECRET_KEYS = (
    "HOPSWORKS_HOST",
    "HOPSWORKS_PORT",
    "HOPSWORKS_PROJECT",
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_FEATURE_GROUP_VERSION",
    "HOPSWORKS_ONLINE_ENABLED",
)

if not os.environ.get("HOPSWORKS_API_KEY"):
    try:
        for key in SECRET_KEYS:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # Expected locally when .streamlit/secrets.toml does not exist.
        pass

os.environ.setdefault("HOPSWORKS_FEATURE_GROUP_VERSION", "9")
os.environ.setdefault("HOPSWORKS_ONLINE_ENABLED", "false")
os.environ.setdefault("RUNNING_IN_HOPSWORKS", "false")
os.environ.setdefault("FEATURE_STORE_BACKEND", "hybrid")

from aqi_predictor.aqi import category, health_message
from aqi_predictor.config import get_settings
from aqi_predictor.hopsworks_integration import HopsworksAdapter


# -----------------------------------------------------------------------------
# Styling — intentionally close to frontend/index.html
# -----------------------------------------------------------------------------

st.html(
    """
<style>
:root {
  --bg:#F4F7F8;
  --surface:#FFFFFF;
  --surface-soft:#F8FBFC;
  --panel:#FFFFFF;
  --panel2:#EEF5F4;
  --line:#D9E3E6;
  --line-strong:#C8D7DA;
  --text:#172033;
  --muted:#66758A;
  --teal:#0F766E;
  --teal-bright:#0D9488;
  --cyan:#0891B2;
  --coral:#E76F51;
  --amber:#D97706;
  --green:#15803D;
  --orange:#EA580C;
  --red:#DC2626;
  --purple:#7C3AED;
  --shadow:0 14px 38px rgba(24,45,55,.10);
  --shadow-soft:0 8px 22px rgba(24,45,55,.07);
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 4% 2%, rgba(13,148,136,.10), transparent 24%),
    radial-gradient(circle at 96% 5%, rgba(231,111,81,.08), transparent 22%),
    linear-gradient(180deg,#F8FBFC 0%,var(--bg) 100%) !important;
  color:var(--text) !important;
}

[data-testid="stMainBlockContainer"] {
  max-width:1500px;
  padding-top:1.2rem;
  padding-left:1.35rem;
  padding-right:1.35rem;
  padding-bottom:2.5rem;
}

header[data-testid="stHeader"] { background:rgba(248,251,252,.88); backdrop-filter:blur(12px); }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

.aqi-nav {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:18px;
  gap:16px;
  padding:2px 4px;
}
.aqi-brand { font-weight:850; letter-spacing:.1px; font-size:1.08rem; color:var(--text); }
.aqi-brand span { color:var(--teal); }
.aqi-status { font-size:12px; color:var(--muted); display:flex; gap:8px; align-items:center; }
.aqi-dot { width:8px; height:8px; border-radius:50%; background:var(--teal-bright); box-shadow:0 0 0 4px rgba(13,148,136,.10); }

.aqi-hero {
  padding:32px;
  border:1px solid rgba(15,118,110,.20);
  border-radius:28px;
  background:
    radial-gradient(circle at 95% 0%, rgba(255,255,255,.16), transparent 26%),
    linear-gradient(128deg,#16324A 0%,#0F5961 54%,#0F766E 100%);
  box-shadow:0 20px 50px rgba(15,70,76,.18);
  margin-bottom:18px;
  position:relative;
  overflow:hidden;
}
.aqi-hero:after {
  content:"";
  position:absolute;
  width:310px;
  height:310px;
  right:-110px;
  top:-165px;
  border-radius:50%;
  border:1px solid rgba(255,255,255,.13);
  background:rgba(255,255,255,.05);
}
.aqi-kicker {
  color:#B9F2EA;
  text-transform:uppercase;
  letter-spacing:.13em;
  font-size:.72rem;
  font-weight:850;
  margin-bottom:.75rem;
}
.aqi-hero h1 {
  font-size:clamp(34px,5vw,68px);
  line-height:.98;
  margin:0 0 12px;
  max-width:900px;
  color:#FFFFFF;
  letter-spacing:-.04em;
}
.aqi-sub { color:#D9EDF0; max-width:850px; font-size:15px; line-height:1.65; }

.aqi-section-title { color:var(--text); font-size:1rem; font-weight:850; margin:0 0 .2rem; }
.aqi-section-sub { color:var(--muted); font-size:.78rem; margin:0 0 .8rem; }
.aqi-city-title { color:var(--text); font-size:1.55rem; font-weight:850; margin:.2rem 0 .85rem; }
.aqi-eyebrow { color:var(--teal); font-size:.7rem; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }

.aqi-metric {
  border:1px solid var(--line);
  background:linear-gradient(180deg,#FFFFFF 0%,#FBFDFD 100%);
  border-radius:18px;
  box-shadow:var(--shadow-soft);
  padding:16px;
  min-height:126px;
  height:100%;
  position:relative;
  overflow:hidden;
}
.aqi-metric:before {
  content:"";
  position:absolute;
  left:0;
  top:0;
  bottom:0;
  width:4px;
  background:linear-gradient(180deg,var(--teal-bright),var(--cyan));
}
.aqi-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; }
.aqi-value { font-size:27px; font-weight:850; margin-top:8px; line-height:1.05; }
.aqi-subvalue { color:var(--muted); font-size:11px; margin-top:9px; line-height:1.4; }

.aqi-alert {
  padding:14px 16px;
  border-radius:16px;
  border:1px solid var(--line);
  margin:16px 0;
  box-shadow:0 6px 18px rgba(24,45,55,.05);
}
.aqi-alert-title { font-size:.82rem; font-weight:850; margin-bottom:.3rem; color:var(--text); }
.aqi-alert-text { font-size:.78rem; color:#526175; line-height:1.55; }
.alert-good { background:#ECFDF3; border-color:#A7E3BF; }
.alert-sensitive { background:#FFF8E6; border-color:#F5D38A; }
.alert-unhealthy { background:#FFF0ED; border-color:#F1B1A3; }
.alert-hazardous { background:#F5EFFF; border-color:#D0B7F4; }

.aqi-card {
  padding:20px;
  position:relative;
  overflow:hidden;
  min-height:235px;
  height:100%;
  border:1px solid var(--line);
  background:linear-gradient(180deg,#FFFFFF 0%,#F9FCFC 100%);
  border-radius:20px;
  box-shadow:var(--shadow-soft);
}
.aqi-card:after {
  content:"";
  position:absolute;
  right:-45px;
  top:-55px;
  width:130px;
  height:130px;
  border-radius:50%;
  background:linear-gradient(135deg,rgba(13,148,136,.12),rgba(8,145,178,.05));
}
.aqi-day { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.aqi-number { font-size:44px; font-weight:850; margin:10px 0 2px; line-height:1; }
.aqi-unit { color:var(--muted); font-size:.85rem; font-weight:500; }
.aqi-category { color:var(--teal); font-weight:750; margin-top:.4rem; }
.aqi-divider { border-top:1px solid var(--line); margin:14px 0; }
.aqi-detail { color:#59697C; font-size:12px; line-height:1.55; }
.aqi-detail strong { color:var(--text) !important; }

.aqi-panel {
  border:1px solid var(--line);
  background:#FFFFFF;
  border-radius:20px;
  box-shadow:var(--shadow-soft);
  padding:18px;
  min-height:210px;
  height:100%;
}
.aqi-info-row {
  display:flex;
  justify-content:space-between;
  gap:14px;
  padding:10px 0;
  border-bottom:1px solid #E8EFF1;
  color:var(--muted);
  font-size:12px;
}
.aqi-info-row:last-child { border-bottom:0; }
.aqi-info-row strong { color:var(--text); text-align:right; }
.aqi-foot { color:var(--muted); font-size:11px; text-align:center; margin:24px 0 8px; }

/* Streamlit widgets */
[data-testid="stSelectbox"] label {
  color:var(--muted) !important;
  font-size:.72rem !important;
  text-transform:uppercase;
  letter-spacing:.08em;
  font-weight:750;
}
div[data-baseweb="select"] > div {
  background:#FFFFFF !important;
  color:var(--text) !important;
  border:1px solid var(--line-strong) !important;
  border-radius:14px !important;
  box-shadow:0 4px 12px rgba(24,45,55,.04);
}
div[data-baseweb="select"] span { color:var(--text) !important; }
.stButton > button {
  width:100%;
  min-height:2.75rem;
  border-radius:14px;
  border:1px solid #0F766E;
  color:#FFFFFF;
  font-weight:800;
  background:linear-gradient(135deg,#0F766E,#0891B2);
  box-shadow:0 8px 18px rgba(15,118,110,.15);
}
.stButton > button:hover {
  border-color:#0D9488;
  color:#FFFFFF;
  background:linear-gradient(135deg,#0D9488,#087C9B);
}
[data-testid="stCaptionContainer"] { color:var(--muted) !important; }
[data-testid="stSpinner"] { color:var(--teal) !important; }

@media(max-width:900px) {
  [data-testid="stMainBlockContainer"] { padding-left:.85rem; padding-right:.85rem; }
  .aqi-hero { padding:22px; }
  .aqi-nav { align-items:flex-start; flex-direction:column; }
  .aqi-card { min-height:auto; }
}
</style>
"""
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def accent_for_aqi(value: float | None) -> str:
    if value is None:
        return "#66758A"
    if value <= 50:
        return "#15803D"
    if value <= 100:
        return "#D97706"
    if value <= 150:
        return "#EA580C"
    if value <= 200:
        return "#DC2626"
    if value <= 300:
        return "#7C3AED"
    return "#B91C1C"


def render_metric(label: str, value: str, subtitle: str, accent: str = "#0F766E") -> None:
    st.html(
        f"""
<div class="aqi-metric">
  <div class="aqi-label">{escape(label)}</div>
  <div class="aqi-value" style="color:{escape(accent)}">{escape(value)}</div>
  <div class="aqi-subvalue">{escape(subtitle)}</div>
</div>
"""
    )


def render_forecast_card(
    day_number: int,
    date_text: str,
    mean_aqi: float,
    peak_aqi: float,
    label: str,
    advice: str,
) -> None:
    accent = accent_for_aqi(mean_aqi)
    st.html(
        f"""
<div class="aqi-card">
  <div class="aqi-day">DAY {day_number} · {escape(date_text)}</div>
  <div class="aqi-number" style="color:{accent}">{mean_aqi:.0f} <span class="aqi-unit">AQI</span></div>
  <div class="aqi-category">{escape(label)}</div>
  <div class="aqi-divider"></div>
  <div class="aqi-detail">Expected peak: <strong style="color:var(--text)">{peak_aqi:.0f} AQI</strong></div>
  <div class="aqi-detail" style="margin-top:8px">{escape(advice)}</div>
</div>
"""
    )


def render_alert(forecast: pd.DataFrame) -> None:
    if forecast.empty:
        st.info("No current forecast is available.")
        return

    valid = forecast.dropna(subset=["aqi_max"])
    if valid.empty:
        st.info("AQI alert status is unavailable.")
        return

    worst = valid.loc[valid["aqi_max"].idxmax()]
    worst_aqi = float(worst["aqi_max"])
    label = category(worst_aqi)
    advice = health_message(worst_aqi)
    date_text = pd.Timestamp(worst["date"]).strftime("%Y-%m-%d")

    if worst_aqi >= 301:
        css_class = "alert-hazardous"
        title = "HAZARDOUS AIR QUALITY ALERT"
    elif worst_aqi >= 151:
        css_class = "alert-unhealthy"
        title = "VERY UNHEALTHY AIR QUALITY ALERT" if worst_aqi >= 201 else "UNHEALTHY AIR QUALITY ALERT"
    elif worst_aqi >= 101:
        css_class = "alert-sensitive"
        title = "AIR QUALITY ALERT FOR SENSITIVE GROUPS"
    else:
        css_class = "alert-good"
        title = "NO UNHEALTHY AQI ALERT"

    st.html(
        f"""
<div class="aqi-alert {css_class}">
  <div class="aqi-alert-title">{escape(title)}</div>
  <div class="aqi-alert-text">
    {escape(date_text)} · forecast peak <strong style="color:var(--text)">{worst_aqi:.0f} AQI</strong>
    · {escape(label)}. {escape(advice)}
  </div>
</div>
"""
    )


# -----------------------------------------------------------------------------
# Hopsworks backend
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_backend():
    settings = get_settings()
    adapter = HopsworksAdapter(settings)
    return settings, adapter


@st.cache_data(ttl=600, show_spinner=False)
def load_production_data():
    settings, adapter = get_backend()

    predictions = adapter.read_group("aqi_daily_predictions_v69")
    hourly = adapter.read_group("aqi_hourly_v69")

    model_info: dict[str, object] = {
        "name": settings.model_name,
        "version": None,
        "metrics": {},
    }

    try:
        models = adapter.model_registry.get_models(settings.model_name)
        if models:
            champion = max(models, key=lambda item: int(getattr(item, "version", 0)))
            model_info["name"] = getattr(champion, "name", settings.model_name)
            model_info["version"] = getattr(champion, "version", None)
            model_info["metrics"] = getattr(champion, "metrics", {}) or {}
    except Exception:
        # Forecast serving remains usable if registry metadata is temporarily unavailable.
        pass

    return predictions, hourly, model_info


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["city"] = result["city"].astype(str).str.strip().str.lower()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")

    if "generated_at" in result.columns:
        result["generated_at"] = pd.to_datetime(result["generated_at"], utc=True, errors="coerce")
    else:
        result["generated_at"] = pd.NaT

    for column in ("aqi_mean", "aqi_max"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["city", "date", "aqi_mean", "aqi_max"])

    return (
        result.sort_values(["generated_at", "date"])
        .drop_duplicates(subset=["city", "date"], keep="last")
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )


def prepare_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["city"] = result["city"].astype(str).str.strip().str.lower()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")

    for column in ("us_aqi", "observed_us_aqi", "computed_us_aqi"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result.dropna(subset=["city", "timestamp"]).sort_values("timestamp").reset_index(drop=True)


def forecast_window(predictions: pd.DataFrame, city_key: str, days: int) -> pd.DataFrame:
    city_frame = predictions[predictions["city"] == city_key].copy().sort_values("date")
    if city_frame.empty:
        return city_frame

    today = pd.Timestamp.now(tz="UTC").normalize()
    current = city_frame[city_frame["date"] >= today].head(days).copy()

    # If the hourly job has not yet produced a full new UTC-day window,
    # show the newest available complete forecast instead of an empty page.
    if len(current) < days:
        current = city_frame.tail(days).copy()

    return current.sort_values("date").reset_index(drop=True)


def latest_current_aqi(hourly: pd.DataFrame, city_key: str) -> tuple[float | None, pd.Timestamp | None, str]:
    city_frame = hourly[hourly["city"] == city_key].copy().sort_values("timestamp")
    if city_frame.empty:
        return None, None, "Unavailable"

    # The production hourly feature group always contains us_aqi. Prefer it
    # for the public current-AQI card; observed_us_aqi can legitimately be absent.
    candidates = (
        ("us_aqi", "Feature Store US AQI"),
        ("observed_us_aqi", "Observed US AQI"),
        ("computed_us_aqi", "Computed US AQI"),
    )

    for column, source in candidates:
        if column not in city_frame.columns:
            continue
        available = city_frame.dropna(subset=[column])
        if available.empty:
            continue
        row = available.iloc[-1]
        return float(row[column]), pd.Timestamp(row["timestamp"]), source

    return None, None, "Unavailable"


def recent_hourly_series(hourly: pd.DataFrame, city_key: str, rows: int = 48) -> tuple[pd.DataFrame, str | None]:
    city_frame = hourly[hourly["city"] == city_key].copy().sort_values("timestamp")
    if city_frame.empty:
        return pd.DataFrame(), None

    for column in ("us_aqi", "observed_us_aqi", "computed_us_aqi"):
        if column in city_frame.columns and city_frame[column].notna().any():
            output = city_frame[["timestamp", column]].dropna().tail(rows).copy()
            output = output.rename(columns={column: "aqi"})
            return output, column

    return pd.DataFrame(), None


def city_display(settings, key: str) -> str:
    try:
        city = settings.city(key)
        return f"{city.name} — {city.province}"
    except Exception:
        return key.replace("_", " ").title()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.html(
    """
<div class="aqi-nav">
  <div class="aqi-brand">Pearls <span>AQI</span> Predictor</div>
  <div class="aqi-status"><span class="aqi-dot"></span> Hopsworks production backend</div>
</div>
"""
)

st.html(
    """
<section class="aqi-hero">
  <div class="aqi-kicker">Production air-quality intelligence</div>
  <h1>Three-day air quality intelligence for Pakistan.</h1>
  <div class="aqi-sub">
    Daily mean and peak AQI forecasts are served directly from the Hopsworks Feature Store.
    Forecast generation and model training run independently from this dashboard, keeping the
    application fast, scalable and production-ready.
  </div>
</section>
"""
)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

try:
    with st.spinner("Connecting to production Feature Store…"):
        prediction_raw, hourly_raw, model_info = load_production_data()
except Exception as exc:
    st.error("Could not connect to the Hopsworks production backend.")
    st.exception(exc)
    st.stop()

predictions = prepare_predictions(prediction_raw)
hourly = prepare_hourly(hourly_raw)

if predictions.empty:
    st.error("The Hopsworks prediction Feature Group contains no usable forecast rows.")
    st.stop()

settings, _ = get_backend()
available_cities = sorted(predictions["city"].dropna().unique().tolist())

if not available_cities:
    st.error("No production city forecasts are available.")
    st.stop()


# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------

left_control, right_control = st.columns([4, 1])
default_index = available_cities.index(settings.default_city) if settings.default_city in available_cities else 0

with left_control:
    selected = st.selectbox(
        "City",
        available_cities,
        index=default_index,
        format_func=lambda key: city_display(settings, key),
    )

with right_control:
    st.write("")
    st.write("")
    if st.button("Refresh data", use_container_width=True):
        load_production_data.clear()
        st.rerun()

forecast = forecast_window(predictions, selected, settings.forecast_days)
if forecast.empty:
    st.error(f"No forecast rows are available for {city_display(settings, selected)}.")
    st.stop()

try:
    city_obj = settings.city(selected)
    city_name = city_obj.name
    province = city_obj.province
except Exception:
    city_name = selected.replace("_", " ").title()
    province = "Pakistan"

st.html(
    f"""
<div style="margin-top:14px;margin-bottom:10px">
  <div class="aqi-eyebrow">Live production forecast</div>
  <div class="aqi-city-title">{escape(city_name)}, {escape(province)}</div>
</div>
"""
)


# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------

current_aqi, current_time, current_source = latest_current_aqi(hourly, selected)
day1_mean = float(forecast.iloc[0]["aqi_mean"])
highest_peak = float(forecast["aqi_max"].max())

generated_values = forecast["generated_at"].dropna() if "generated_at" in forecast.columns else pd.Series(dtype="datetime64[ns, UTC]")
generated_at = generated_values.max() if not generated_values.empty else None

metric_columns = st.columns(5)

with metric_columns[0]:
    render_metric(
        "Current AQI",
        f"{current_aqi:.0f}" if current_aqi is not None else "—",
        category(current_aqi) if current_aqi is not None else current_source,
        accent_for_aqi(current_aqi),
    )

with metric_columns[1]:
    render_metric("Day 1 mean", f"{day1_mean:.0f}", "Predicted daily US AQI", accent_for_aqi(day1_mean))

with metric_columns[2]:
    render_metric("3-day peak", f"{highest_peak:.0f}", "Highest expected AQI", accent_for_aqi(highest_peak))

with metric_columns[3]:
    registry_version = model_info.get("version")
    render_metric(
        "Production model",
        f"v{registry_version}" if registry_version is not None else "Online",
        str(model_info.get("name") or settings.model_name),
        "#0F766E",
    )

with metric_columns[4]:
    render_metric(
        "Forecast updated",
        generated_at.strftime("%H:%M") if generated_at is not None and pd.notna(generated_at) else "—",
        generated_at.strftime("%d %b %Y UTC") if generated_at is not None and pd.notna(generated_at) else "Feature Store",
        "#0891B2",
    )

if current_time is not None:
    st.caption(f"Latest hourly observation: {current_time.strftime('%Y-%m-%d %H:%M UTC')} · {current_source}")

render_alert(forecast)


# -----------------------------------------------------------------------------
# Forecast cards
# -----------------------------------------------------------------------------

forecast_columns = st.columns(len(forecast))
for index, row in forecast.iterrows():
    mean_aqi = float(row["aqi_mean"])
    peak_aqi = float(row["aqi_max"])

    raw_label = row.get("category")
    label = str(raw_label).strip() if raw_label is not None and not pd.isna(raw_label) else category(mean_aqi)
    if not label:
        label = category(mean_aqi)

    with forecast_columns[index]:
        render_forecast_card(
            day_number=index + 1,
            date_text=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            mean_aqi=mean_aqi,
            peak_aqi=peak_aqi,
            label=label,
            advice=health_message(mean_aqi),
        )


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------

st.write("")
chart_left, chart_right = st.columns([1.35, 1])

with chart_left:
    st.html(
        """
<div class="aqi-section-title">Production 3-day AQI forecast</div>
<div class="aqi-section-sub">Daily mean AQI and expected peak AQI</div>
"""
    )

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["aqi_mean"],
            mode="lines+markers",
            name="Daily mean AQI",
            line={"width": 3, "color": "#0891B2"},
            marker={"size": 9, "color": "#0891B2"},
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.07)",
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["aqi_max"],
            mode="lines+markers",
            name="Daily peak AQI",
            line={"width": 2, "dash": "dot", "color": "#E76F51"},
            marker={"size": 8, "color": "#E76F51"},
        )
    )
    forecast_fig.update_layout(
        height=360,
        margin={"l": 15, "r": 15, "t": 20, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FBFC",
        font={"color": "#66758A"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": None, "showgrid": False, "linecolor": "rgba(100,116,139,.20)"},
        yaxis={"title": "US AQI", "gridcolor": "rgba(100,116,139,.14)", "zeroline": False},
    )
    st.plotly_chart(forecast_fig, use_container_width=True, config={"displayModeBar": False})

with chart_right:
    st.html(
        """
<div class="aqi-section-title">Recent hourly AQI</div>
<div class="aqi-section-sub">Latest Feature Store observations</div>
"""
    )

    recent, recent_source = recent_hourly_series(hourly, selected, rows=48)
    if not recent.empty:
        recent_fig = go.Figure()
        recent_fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent["aqi"],
                mode="lines",
                name="Hourly AQI",
                line={"width": 2.5, "color": "#0F766E"},
                fill="tozeroy",
                fillcolor="rgba(45,212,191,.06)",
            )
        )
        recent_fig.update_layout(
            height=360,
            margin={"l": 15, "r": 15, "t": 20, "b": 15},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFC",
            font={"color": "#66758A"},
            hovermode="x unified",
            showlegend=False,
            xaxis={"title": None, "showgrid": False, "linecolor": "rgba(100,116,139,.20)"},
            yaxis={"title": "US AQI", "gridcolor": "rgba(100,116,139,.14)", "zeroline": False},
        )
        st.plotly_chart(recent_fig, use_container_width=True, config={"displayModeBar": False})
        if recent_source:
            st.caption(f"Hourly chart source: {recent_source}")
    else:
        st.info("Recent hourly AQI observations are unavailable for this city.")


# -----------------------------------------------------------------------------
# Production information
# -----------------------------------------------------------------------------

info_left, info_right = st.columns(2)

with info_left:
    st.html(
        f"""
<div class="aqi-panel">
  <div class="aqi-section-title">Production backend</div>
  <div class="aqi-section-sub">Serverless forecast serving architecture</div>
  <div class="aqi-info-row"><span>Forecast source</span><strong>aqi_daily_predictions_v69</strong></div>
  <div class="aqi-info-row"><span>Observation source</span><strong>aqi_hourly_v69</strong></div>
  <div class="aqi-info-row"><span>Feature Group version</span><strong>v{int(settings.hopsworks_feature_group_version)}</strong></div>
  <div class="aqi-info-row"><span>Serving mode</span><strong>Precomputed predictions</strong></div>
</div>
"""
    )

with info_right:
    version_text = str(model_info.get("version")) if model_info.get("version") is not None else "—"
    st.html(
        f"""
<div class="aqi-panel">
  <div class="aqi-section-title">Model Registry</div>
  <div class="aqi-section-sub">Production forecasting model</div>
  <div class="aqi-info-row"><span>Model</span><strong>{escape(str(model_info.get('name') or settings.model_name))}</strong></div>
  <div class="aqi-info-row"><span>Registry version</span><strong>{escape(version_text)}</strong></div>
  <div class="aqi-info-row"><span>Registry status</span><strong style="color:#0F766E">Connected</strong></div>
  <div class="aqi-info-row"><span>Prediction horizon</span><strong>{int(settings.forecast_days)} days</strong></div>
</div>
"""
    )

st.html(
    """
<div class="aqi-panel" style="margin-top:14px;min-height:auto">
  <div class="aqi-section-title">Production architecture</div>
  <div class="aqi-section-sub" style="margin-bottom:0;line-height:1.7">
    Weather and pollution providers → feature engineering → Hopsworks Feature Store →
    production ML ensemble → precomputed three-day forecasts → Streamlit dashboard.
    The web app does not retrain the model or call external forecast providers on every page load.
  </div>
</div>
"""
)

st.html(
    """
<div class="aqi-foot">
  Pearls AQI Predictor · Hopsworks Feature Store + Model Registry · Production three-day AQI forecasting
</div>
"""
)
