# Pearls AQI Predictor

Production-ready **3-day Air Quality Index (AQI) forecasting system for major cities in Pakistan**, built with Hopsworks, Scikit-learn, PyTorch, SHAP, and Streamlit.

> **Live Dashboard:** `https://aqiprediction-ai.streamlit.app/`  
> **Full Technical Report:** [PROJECT_REPORT.md](PROJECT_REPORT.md)

![Pearls AQI Dashboard](reports/final/evidence/01_streamlit_dashboard_top.png)

---

## Overview

Pearls AQI Predictor is an end-to-end machine-learning and MLOps project that forecasts **daily mean AQI and expected peak AQI for the next three days**.

The system combines:

- Air-quality and pollutant observations
- Weather and future-weather forecasts
- Historical AQI behaviour
- Lag and rolling features
- Seasonal and climatological signals
- Hopsworks Feature Store
- Hopsworks Model Registry
- Automated hourly forecasting
- Automated daily retraining
- SHAP explainability
- AQI health alerts
- Streamlit Community Cloud deployment

The public dashboard serves **precomputed production forecasts from Hopsworks**, so it does not retrain the model or call all providers on every page load.

---

## Architecture

```text
External Weather / Pollution Providers
                |
                v
        Feature Engineering
                |
                v
       Hopsworks Feature Store
                |
        +-------+------------------+
        |                          |
        v                          v
Hourly Production Job       Daily Training Job
        |                          |
        v                          v
Production Forecasts        Candidate Models
        |                          |
        v                          v
Prediction Feature Group    Evaluation / Promotion Gate
        |                          |
        |                          v
        |                    Model Registry
        |                          |
        +-------------+------------+
                      |
                      v
           Streamlit Cloud Dashboard
```

---

## Key Results

### Production Ensemble

| Metric | Result |
|---|---:|
| R² | **0.7473** |
| RMSE | **17.3178** |
| MAE | **12.2618** |
| Bias | **-0.0245** |
| Day-3 R² | **0.6311** |
| Day-3 RMSE | **20.9695** |
| Macro-city R² | **0.4767** |

The pooled production R² is higher than the macro-city R² because city-level forecasting difficulty and historical data coverage are uneven.

**Multan repeatedly produced weaker city-level R² because its usable historical AQI target history was more limited than that of the stronger training cities.** Since macro-city R² gives each city equal weight, Multan's weaker city-level performance lowers the macro aggregate more strongly than it lowers the pooled metric.

---

## Model Comparison

| Model | RMSE | MAE | R² | Bias |
|---|---:|---:|---:|---:|
| **Production ensemble** | **17.32** | **12.26** | **0.747** | **-0.02** |
| Persistence baseline | 20.64 | 13.25 | 0.622 | -0.07 |
| PyTorch MLP | 22.68 | 18.16 | 0.543 | +13.15 |

The PyTorch neural network was retained as an experiment rather than promoted because it underperformed both the persistence baseline and the production ensemble.

---

## Core Features

- 3-day AQI forecasting
- Current/latest AQI display
- Daily mean and daily peak AQI predictions
- AQI health-warning alerts
- Historical backfill
- Hopsworks Feature Store integration
- Hopsworks Model Registry integration
- Hourly production pipeline
- Daily model retraining
- Scikit-learn ensemble modelling
- PyTorch deep-learning experiment
- SHAP explainability
- Chronological train/validation/test evaluation
- Streamlit Community Cloud deployment
- GitHub source control

---

## Feature Store

The project uses four main Hopsworks Feature Groups:

```text
aqi_hourly_v69
aqi_daily_training_v69
aqi_provider_snapshots_v69
aqi_daily_predictions_v69
```

**Feature Group version:** `9`

The Hopsworks project contains:

- **4 Feature Groups**
- **843 registered features**

The implementation accesses Feature Groups directly through the Hopsworks SDK.

![Hopsworks Feature Groups](reports/final/evidence/03_hopsworks_feature_groups_v9.png)

---

## Production Model

The production champion is registered in Hopsworks Model Registry as:

```text
pearls_aqi_daily_ensemble
```

**Hopsworks version:** `2`  
**Framework:** Scikit-learn

![Hopsworks Model Registry](reports/final/evidence/05_hopsworks_model_registry_v2_metrics.png)

---

## Automation

### Hourly Production Pipeline

Hopsworks job:

```text
aqi_hourly_v69
```

Cron:

```text
17 * * * *
```

Runs at **17 minutes past every hour**.

Main flow:

```text
hydrate_hopsworks.py --features --model
        ↓
forecast.py --city all
        ↓
sync_hopsworks.py --features --city all
```

### Daily Training Pipeline

Hopsworks job:

```text
aqi_daily_training_v69
```

Cron:

```text
25 3 * * *
```

Runs every day at **03:25 UTC**.

The daily job:

1. Hydrates training features
2. Validates data
3. Rebuilds the chronological training matrix
4. Trains candidate models
5. Evaluates the candidate
6. Applies the quality/promotion gate
7. Uploads the candidate
8. Updates the production champion only when promotion criteria are satisfied

---

## Exploratory Data Analysis

EDA covered:

- **175,200 hourly rows**
- **7,310 daily city observations**
- **10 cities**
- Historical range: **2024-07-25 to 2026-07-24**

Key EDA results:

| Finding | Result |
|---|---:|
| Mean AQI | 119.81 |
| Median AQI | 109.67 |
| Maximum daily AQI | 486.08 |
| AQI ≥ 151 city-days | 23.56% |
| Strongest positive AQI correlation | PM2.5 ≈ +0.770 |
| Strongest negative AQI correlation | Wind speed ≈ -0.267 |
| Highest average AQI season | Winter Smog |
| Highest average AQI city | Faisalabad |

### EDA Artifacts

```text
reports/eda/
```

Examples:

![Average AQI by City](reports/eda/01_city_mean_aqi.png)

![Seasonal AQI](reports/eda/03_seasonal_aqi.png)

![Correlation Heatmap](reports/eda/07_correlation_heatmap.png)

---

## SHAP Explainability

SHAP was used to explain the production forecasting components.

### Day-1 Drivers

The strongest Day-1 features include:

1. `pm2_5__mean`
2. `pm2_5__max`
3. `aqi_peak_excess`
4. `future_wind_speed_10m__mean`
5. `pm10__max`
6. `us_aqi__median`
7. `us_aqi__max`
8. `future_ventilation_proxy`
9. `aqi_mean_change_1d`
10. `ozone__max`

### Day-3 Drivers

Longer-horizon predictions rely more strongly on:

- Smoothed PM2.5 history
- Smoothed AQI history
- Future wind conditions
- City-month climatology
- Seasonal/time signals
- Future temperature

![Day-1 SHAP](reports/explainability/shap_day1_importance.png)

![Day-3 SHAP](reports/explainability/shap_day3_importance.png)

---

## AQI Alerts

The dashboard uses the **highest predicted AQI peak** in the three-day period when deciding the alert level.

| AQI Peak | Alert |
|---:|---|
| Below 101 | No unhealthy alert |
| 101–150 | Sensitive-groups warning |
| 151–200 | Unhealthy |
| 201–300 | Very unhealthy |
| 301+ | Hazardous |

This prevents a short but dangerous AQI peak from being hidden by a lower daily average.

---

## Dashboard

The deployed Streamlit application displays:

- City selector
- Latest/current AQI
- Current AQI category
- Day-1 mean AQI
- Highest 3-day AQI peak
- Production model version
- Forecast update time
- Health alert banner
- Day 1, Day 2, and Day 3 forecast cards
- Daily mean and peak forecast chart
- Recent hourly AQI chart
- Feature Store serving details
- Model Registry information
- Production architecture

![Dashboard Top](reports/final/evidence/01_streamlit_dashboard_top.png)

![Dashboard Bottom](reports/final/evidence/02_streamlit_dashboard_bottom.png)

---

## Deep-Learning Experiment

A PyTorch multilayer perceptron was tested using:

- 32 input features
- Dense layers: 128 → 64 → 32 → 1
- ReLU activations
- Dropout
- AdamW optimizer
- MSE loss
- Early stopping
- Chronological 70/15/15 split

Training stopped at epoch **33**.

The neural model did not outperform the production ensemble, which demonstrates that model promotion was based on objective held-out performance rather than model complexity.

Artifacts:

```text
reports/deep_learning/
```

---

## Repository Structure

```text
Aqi_prediction/
│
├── README.md
├── PROJECT_REPORT.md
├── app.py
├── requirements.txt
│
├── cloud/
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── config/
├── data/
├── frontend/
├── reports/
│   ├── eda/
│   ├── explainability/
│   ├── deep_learning/
│   └── final/
│       └── evidence/
│
├── scripts/
│   ├── backfill.py
│   ├── eda.py
│   ├── explain.py
│   ├── forecast.py
│   ├── train_deep.py
│   ├── verify_serving.py
│   ├── hopsworks_hourly_pipeline.py
│   └── hopsworks_daily_training_pipeline.py
│
├── src/
│   └── aqi_predictor/
│
├── tests/
└── .github/
```

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kashan1272/Aqi_prediction.git
cd Aqi_prediction
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a private `.env` file locally.

Example variable names:

```text
HOPSWORKS_HOST
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
HOPSWORKS_FEATURE_GROUP_VERSION
HOPSWORKS_ONLINE_ENABLED
```

**Never commit the real `.env` file or Hopsworks API key.**

### 5. Run the Local Dashboard

```bash
streamlit run app.py
```

### 6. Run the Cloud-Serving Dashboard Locally

```bash
streamlit run cloud/streamlit_app.py
```

---

## Useful Project Commands

EDA:

```bash
python scripts/eda.py
```

Deep-learning experiment:

```bash
python scripts/train_deep.py
```

SHAP explainability:

```bash
python scripts/explain.py
```

Verify Hopsworks serving:

```bash
python scripts/verify_serving.py
```

Generate forecasts:

```bash
python scripts/forecast.py --city all
```

---

## Security

Secrets are intentionally separated from source code.

Local development:

```text
.env
```

Cloud deployment:

```text
Streamlit Secrets
```

The real `HOPSWORKS_API_KEY` must never be stored in GitHub, README files, screenshots, or source code.

---

## Known Limitations

- Historical AQI target coverage is uneven between cities.
- Multan has repeatedly shown weaker city-level R² because of its more limited usable historical AQI target history.
- Day-3 forecasting is harder than Day-1 forecasting.
- External provider/network issues can occasionally cause an hourly execution to fail.
- `future_pressure_anomaly` can be all-missing for some model matrices and may trigger a non-blocking Scikit-learn imputation warning.
- The project uses Feature Groups directly, so Hopsworks may show `0 Feature Views`.
- The project serves precomputed predictions through Streamlit Cloud, so Hopsworks may show `0 Model Deployments`.

These values reflect architectural choices rather than missing production functionality.

---

## Future Improvements

- Increase clean historical AQI coverage for Multan
- Add provider retries with exponential backoff
- Add stronger pipeline failure alerting
- Remove/conditionally generate all-missing pressure features
- Evaluate LSTM and GRU models
- Evaluate temporal transformers
- Add probabilistic forecast intervals
- Add data/model drift monitoring
- Add satellite aerosol data
- Add traffic and industrial-emission features
- Add automatic email/message alerts for hazardous AQI
- Optionally add Hopsworks Feature Views for richer lineage

---

## Documentation

For the complete technical implementation, requirement mapping, detailed EDA, chronological evaluation, deep-learning experiment, SHAP results, Hopsworks evidence, limitations, and production architecture:

### [Read the Full Project Report](PROJECT_REPORT.md)

---

## Evidence

Production evidence is stored under:

```text
reports/final/evidence/
```

It includes:

- Streamlit production dashboard
- Hopsworks Feature Groups
- Prediction Feature Group
- Model Registry
- Hourly job schedule
- Hourly successful executions
- Daily training schedule
- Daily successful training execution
- Job configurations

---

## Final Status

**End-to-end AQI system:** Complete  
**Historical backfill:** Complete  
**Feature Store:** Complete  
**Model Registry:** Complete  
**Hourly automation:** Complete  
**Daily retraining:** Complete  
**EDA:** Complete  
**Deep learning experiment:** Complete  
**SHAP explainability:** Complete  
**AQI alerts:** Complete  
**Cloud dashboard:** Deployed  
**Final technical report:** Complete  

---

## Links

**GitHub Repository:**  
https://github.com/Kashan1272/Aqi_prediction

**Live Streamlit Dashboard:**  
`REPLACE_WITH_YOUR_STREAMLIT_APP_URL`

**Full Technical Report:**  
[PROJECT_REPORT.md](PROJECT_REPORT.md)
