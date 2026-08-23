# Pearls AQI Predictor v6.6

A professional, end-to-end machine-learning system that predicts **daily mean and daily peak US AQI for the next three days** across 25 geographically balanced Pakistani cities. It includes API ingestion, historical backfill, weather-pattern engineering, chronological model training, optional sensor calibration, Hopsworks integration, automation, FastAPI, Streamlit, and a standalone HTML dashboard.

## Why v6 uses daily three-day targets

Earlier versions attempted to optimize one score across all 72 individual future hours. Accuracy dropped sharply after the first day because hourly AQI is noisy and the long-horizon target is highly uncertain. The assignment asks for AQI in the **next three days**, so v6 trains explicit Day 1, Day 2, and Day 3 targets:

- daily mean AQI — primary scored target;
- daily peak AQI — health-risk target;
- a 72-hour dashboard curve calibrated to the daily ML forecast.

The hourly curve remains available for visualization, but it is not used to inflate the reported model score.

## System architecture

```text
Open-Meteo weather + air-quality history
Open-Meteo previous-run weather forecasts
Optional OpenAQ station observations
Optional OpenWeather and AQICN live forecasts
                    │
                    ▼
         Chunked, retry-safe backfill
                    │
                    ▼
      Local Parquet/CSV Feature Store
      or Hopsworks Feature Store v6
                    │
                    ▼
  Daily feature and target engineering
  lags • rolling statistics • trends
  seasonality • pollutant interactions
  lead-aligned future weather • city/province
                    │
                    ▼
 Chronological training with 3-day embargo
 Ridge • HistGradientBoosting
 Random Forest • Extra Trees
                    │
                    ▼
 Untouched test gate + baseline comparison
                    │
                    ▼
 Local/Hopsworks Model Registry
                    │
                    ▼
 FastAPI + HTML dashboard / Streamlit
```

## Data providers

### Required national provider

**Open-Meteo** is used for historical weather, historical CAMS air quality, live weather, live air quality, and one-, two-, and three-day previous-run weather features. The public non-commercial endpoints do not require an API key.

### Optional providers

- **OpenAQ** — real station observations, API auditing, live correction, and an optional historical sensor-calibration layer.
- **OpenWeather** — additional live weather and air-pollution forecast used in the production ensemble.
- **AQICN/WAQI** — current AQI and pollutant forecast used in the production ensemble.

Every live OpenWeather/AQICN/Open-Meteo provider forecast is also saved as an **issue-time provider snapshot**. Once the target day becomes historical, later retraining can use those snapshot values without leakage. This allows optional live APIs to improve the model gradually even when they do not offer a matching two-year free historical archive.

A missing optional provider never stops the national Open-Meteo pipeline. Provider failures and response traces are shown in API reports and dashboard health panels.

## Models used

For each of the six targets — Day 1/2/3 mean and Day 1/2/3 peak — the trainer compares:

1. Ridge Regression
2. Histogram Gradient Boosting
3. Random Forest Regressor
4. Extra Trees Regressor

The best candidate is selected through rolling chronological cross-validation. Final promotion is decided only on a later untouched test period. A seasonal persistence forecast is evaluated as the baseline.

The production gate requires:

```text
Daily-mean test R² >= 0.70
Daily-mean test RMSE <= 30
Day-3 test R² >= 0.45
Every daily-mean target must beat its baseline
```

These thresholds validate the software; they do not fabricate results. Real performance depends on the actual data returned for Pakistan.


## Quota-safe 25-city backfill profile

Version 6.3 intentionally configures 25 geographically balanced Pakistani cities.
The profile keeps the 730-day Open-Meteo workload within a conservative project-side
daily budget while retaining Punjab smog, Sindh heat, coastal, mountain, dry and
monsoon climate patterns.

Check the workload before downloading:

```powershell
python scripts/backfill_plan.py --city all --days 730
```

Run the backfill normally. Waiting and resuming on HTTP 429 is enabled by default:

```powershell
python scripts/backfill.py --city all --days 730
```

The client uses a persistent quota ledger, 14-day chunks, a 3-second request pace,
chunk caching, complete-city skipping and conservative soft limits. The free public
API can still return HTTP 429 when other traffic shares the same public IP. In that
case the command sleeps until the provider reset and resumes the same cached city;
it does not discard completed chunks.

After completion:

```powershell
python scripts/backfill_status.py --city all --days 730 --strict
python scripts/diagnose_storage.py --group aqi_history
python scripts/validate_data.py --city all --strict
```

## Project structure

```text
pearls-aqi-predictor-v6/
├── app.py                         # Streamlit dashboard
├── api.py                         # FastAPI + HTML frontend server
├── frontend/index.html            # Professional standalone frontend
├── config/cities.yaml             # 25 geographically balanced Pakistani cities
├── scripts/
│   ├── probe_apis.py
│   ├── backfill.py
│   ├── backfill_observations.py
│   ├── validate_data.py
│   ├── train.py
│   ├── forecast.py
│   ├── diagnose_storage.py
│   ├── sync_hopsworks.py
│   └── hydrate_hopsworks.py
├── src/aqi_predictor/
│   ├── providers/                 # Open-Meteo, OpenAQ, OpenWeather, AQICN
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   ├── inference.py
│   ├── storage.py
│   └── hopsworks_integration.py
├── .github/workflows/             # CI, backfill, hourly inference, daily training
├── .env.example
├── requirements.txt
├── requirements-hopsworks.txt
├── requirements-shap.txt
└── PROJECT_REPORT.md
```

## 1. Installation

Python 3.11 is recommended for the broadest Hopsworks compatibility.

### Windows PowerShell

```powershell
cd D:\pearls-aqi-predictor-v6
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation --no-deps .
Copy-Item .env.example .env
```

When Python 3.11 is unavailable, Python 3.12 or 3.13 can be used locally. Hopsworks dependencies may have stricter version support.

## 2. Configure APIs before the first run

Open `.env`.

### Minimum configuration

```env
AQI_CITY=lahore
FORECAST_DAYS=3
BACKFILL_DAYS=730

OPENAQ_API_KEY=
OPENWEATHER_API_KEY=
AQICN_API_TOKEN=

FEATURE_STORE_BACKEND=local
```

Open-Meteo has no key field because the public endpoint used by the project does not require one.

### Enable OpenAQ

```env
OPENAQ_API_KEY=your_openaq_key
OPENAQ_RADIUS_METERS=25000
```

### Enable OpenWeather

```env
OPENWEATHER_API_KEY=your_openweather_key
```

### Enable AQICN

```env
AQICN_API_TOKEN=your_aqicn_token
```

Never put keys inside `frontend/index.html`, JavaScript, source code, or Git.

## 3. Prove that APIs are returning usable data

Test one city first:

```powershell
python scripts/probe_apis.py --city lahore --days 7 --strict
```

Then test other representative cities:

```powershell
python scripts/probe_apis.py --city karachi --days 7
python scripts/probe_apis.py --city islamabad --days 7
```

Reports are written to `reports/api_probe_<city>.json`. Each report includes:

- requested endpoint;
- HTTP status;
- elapsed time and retry attempt;
- response size;
- row count;
- non-null ratios;
- numeric ranges and unique counts;
- configured/available provider status.

Open-Meteo checks are required. `--strict` fails only for required sources. Optional provider checks may be unavailable where no station exists; use `--strict-optional` only when every configured optional source must also succeed.

## 4. Historical backfill

Backfill all 25 cities:

```powershell
python scripts/backfill.py --city all --days 730
```

The client uses bounded date chunks, retry-safe requests, exponential backoff, atomic writes, and city partitions. Existing city data is merged by timestamp, so an interrupted run can be resumed safely.

Optional OpenAQ station history:

```powershell
python scripts/backfill_observations.py --city all --days 730
```

This step is optional. OpenAQ observations are stored separately and never overwrite the national Open-Meteo/CAMS fields.

## 5. Validate storage and training data

```powershell
python scripts/diagnose_storage.py --group aqi_history
python scripts/validate_data.py --city all --strict
```

Training should begin only when:

```text
partitions=40
failures=0
ready_for_training=true
```

The validator checks timestamp continuity, duplicates, target variation, required future-weather coverage, missing values, flat placeholder series, and city coverage.

## 6. Train

Technical smoke test:

```powershell
python scripts/train.py --city all --quick
```

Quick mode never promotes a model.

Full production training:

```powershell
python scripts/train.py --city all
```

Outputs:

```text
reports/training_report_v6.json
reports/test_predictions_v6.parquet or .csv.gz
artifacts/models/pearls_aqi_daily_ensemble/<version>/
```

View the important result:

```powershell
python -c "import json; r=json.load(open('reports/training_report_v6.json')); print(json.dumps({'selected':r['selected_algorithms'],'test':r['test_metrics'],'gate':r['quality_gate'],'promotion':r['promotion']}, indent=2))"
```

Optional SHAP support:

```powershell
python -m pip install -r requirements-shap.txt
```

Without SHAP, the project automatically uses permutation importance.

## 7. Generate forecasts

Test one city:

```powershell
python scripts/forecast.py --city lahore
```

Generate forecasts for all cities:

```powershell
python scripts/forecast.py --city all
```

The live forecast combines the ML output with available providers using normalized weights. Missing providers are removed automatically. OpenAQ station bias is applied with a decreasing Day 1–3 influence. Provider forecasts are saved under `data/provider_snapshots/` for future leakage-safe retraining. Only completed historical hours are written back to the training store; future live forecasts are never inserted as observed history.

## 8. Run the dashboards

### Professional HTML + FastAPI

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

```text
Dashboard: http://localhost:8000
API docs:  http://localhost:8000/docs
Health:    http://localhost:8000/api/v1/health
```

### Streamlit

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

Dashboard sections include current AQI, Day 1–3 mean and peak cards, 72-hour curve, pollutant glance, weather drivers, uncertainty, provider health, model metrics, selected algorithms, city map/ranking, health alerts, and feature explanations.

## 9. Hopsworks configuration

Install only after the local pipeline works:

```powershell
python -m pip install --no-cache-dir -r requirements-hopsworks.txt
```

Update `.env`:

```env
FEATURE_STORE_BACKEND=hybrid
HOPSWORKS_HOST=your-cluster.hopsworks.ai
HOPSWORKS_PORT=443
HOPSWORKS_PROJECT=your_exact_project_name
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_FEATURE_GROUP_VERSION=6
HOPSWORKS_ONLINE_ENABLED=false
HOPSWORKS_REQUIRED=false
HOPSWORKS_SYNC_MODELS=true
RUNNING_IN_HOPSWORKS=false
```

Use the hostname without `https://` and without a trailing slash.

Sync local data:

```powershell
python scripts/sync_hopsworks.py --features --city all
```

Sync the production model only after promotion:

```powershell
python scripts/sync_hopsworks.py --model
```

Sync everything:

```powershell
python scripts/sync_hopsworks.py
```

The project uses these Hopsworks assets:

```text
aqi_hourly_v6
aqi_daily_training_v6
aqi_provider_snapshots_v6
aqi_daily_predictions_v6
pearls_aqi_daily_ensemble
```

For a stateless cloud instance, hydrate local runtime data from Hopsworks:

```powershell
python scripts/hydrate_hopsworks.py --features --model
```

## 10. Automation and live deployment

The repository contains:

- `.github/workflows/hourly_forecast.yml`
- `.github/workflows/daily_training.yml`
- `.github/workflows/backfill.yml`
- `.github/workflows/ci.yml`

Add these GitHub Actions secrets:

```text
OPENAQ_API_KEY
OPENWEATHER_API_KEY
AQICN_API_TOKEN
HOPSWORKS_HOST
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
```

For Render or another Python host:

```text
Build command:
python -m pip install -r requirements.txt && python -m pip install -r requirements-hopsworks.txt && python -m pip install --no-deps .

Start command:
uvicorn api:app --host 0.0.0.0 --port $PORT

Health path:
/api/v1/health
```

Use production environment variables on the hosting platform; do not upload `.env`.

## Controlled verification

A controlled two-city, 800-day synthetic smoke test produced:

```text
Daily-mean test R²:   0.7826
Daily-mean test RMSE: 5.8592
Day-1 R²:             0.8670
Day-2 R²:             0.7292
Day-3 R²:             0.7475
```

These numbers verify that the software can learn a stable three-day pattern and pass its gate on controlled data. They are **not a claim about real Pakistani AQI accuracy**. Real performance must be read from the untouched test section in `training_report_v6.json` after the live backfill.

## Important migration note

Use v6 as a clean project. Do not copy old trained model files or reports from v5. The target contract has changed from hourly 1–72 to daily Day 1–3 targets. A fresh v6 historical backfill is recommended so all feature columns use one consistent schema.


## v6.6 Core-10 precision profile

This profile deliberately uses ten fixed, geographically diverse Pakistani cities. The selection was made before the next untouched test run and is not based on cherry-picking previous test scores.

The trainer keeps the v6.4 memory-safe rolling OOF stack (Ridge, HistGradientBoosting, Random Forest, Extra Trees and the seasonal baseline), but learns convex weights with equal city influence. A strict matrix audit runs before fitting and rejects duplicate city/date/horizon rows, target leakage, missing horizons, inadequate lead-weather coverage, weak target variation, chronological split overlap, and incomplete city representation.

Run the validation sequence before training:

```powershell
python scripts/backfill_status.py --city all --days 730 --strict
python scripts/diagnose_storage.py --group aqi_history --city all
python scripts/validate_data.py --city all --strict
python scripts/validate_training_matrix.py --city all --strict
python scripts/train.py --city all
```

Reducing the catalogue does not guarantee a higher untouched-test R2. The quality gate remains R2 >= 0.70, RMSE <= 30, Day-3 R2 >= 0.45, with every mean horizon beating its baseline.

## v6.7 development-selected city profile

v6.7 keeps Karachi and Multan and automatically selects six additional cities
using development-only chronological OOF metrics. It does not cherry-pick
cities from the final test report. Run `python scripts/city_selection_report.py`
after training to see the active eight-city profile and selection scores.
See `PRECISION_CITY_SELECTION_GUIDE.md` for the complete method.

## v6.8 local-global champion/challenger upgrade

See `CHAMPION_CHALLENGER_GUIDE.md` for protected model promotion, challenger
uploads, and rollback. v6.8 adds OOF city experts and hierarchical residual
calibration while retaining the v6.7 production model whenever a new run is
not better.

## v6.9 Day-3 precision and bias guard

v6.9 keeps the passing v6.8 champion protected while training a stronger
Day-3 challenger. It adds horizon-specific candidates, recent/extreme OOF
weighting, leak-free trajectory and weather-shift features, provider-consensus
features, temporal-holdout calibration selection, exact matrix fingerprints,
and Day-3/bias-aware promotion guards.

Run:

```powershell
python scripts/verify_v69.py
python scripts/validate_training_matrix.py --city all --strict
python scripts/train.py --city all
```

See `DAY3_PRECISION_GUIDE.md` for training, Hopsworks upload and rollback.
