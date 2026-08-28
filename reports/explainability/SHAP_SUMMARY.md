# Pearls AQI Predictor — SHAP Explainability

- Production version: `20260729T212359325910Z`
- Model name: `pearls_aqi_daily_ensemble`
- Explainability method: SHAP
- Targets explained: Day 1 mean AQI and Day 3 mean AQI

## day1_mean

- Dominant ensemble component: `hist_gradient`

### Top features

- pm2_5__mean: 25.782260
- pm2_5__max: 3.617967
- aqi_peak_excess: 1.777292
- future_wind_speed_10m__mean: 1.016773
- pm10__max: 0.817725
- us_aqi__median: 0.797989
- us_aqi__max: 0.733260
- future_ventilation_proxy: 0.704744
- aqi_mean_change_1d: 0.645176
- ozone__max: 0.616502

## day3_mean

- Dominant ensemble component: `hist_gradient`

### Top features

- pm2_5__mean__ewm7: 5.401881
- us_aqi__mean__ewm7: 4.156347
- pm2_5__mean: 3.994099
- future_wind_speed_10m__mean: 2.676857
- aqi_city_month_climatology: 2.391775
- future_wind_direction_10m__min: 1.852202
- target_doy_cos: 1.704630
- pm2_5__mean__roll14_mean: 1.672257
- future_wind_v: 1.423958
- future_temperature_2m__max: 1.419454
