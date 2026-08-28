# Pearls AQI Predictor — EDA Summary

## Dataset

- Hourly rows analyzed: 175,200
- Daily city observations: 7,310
- Cities analyzed: 10
- Data range: 2024-07-25 00:00:00+00:00 to 2026-07-24 23:00:00+00:00

## Main AQI Findings

- Overall mean AQI: 119.81
- Overall median AQI: 109.67
- Maximum daily AQI: 486.08
- AQI >= 151: 23.56% of city-days
- Highest average AQI city: faisalabad
- Highest average AQI month: 2026-01
- Highest average AQI season: Winter Smog

## Relationships

- Strongest positive AQI correlation: pm2_5 (0.770)
- Strongest negative AQI correlation: wind_speed_10m (-0.267)

## Data Quality

- Highest missing feature: apparent_temperature (0.00%)

Generated automatically by `scripts/eda.py`.
