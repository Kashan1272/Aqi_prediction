from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .bootstrap import PROJECT_ROOT


@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str
    province: str
    latitude: float
    longitude: float
    timezone: str
    elevation_m: float
    priority: int


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app_env: str
    log_level: str
    default_city: str
    forecast_days: int
    backfill_days: int
    api_timeout_seconds: int
    request_pause_seconds: float
    api_retry_attempts: int
    chunk_days: int
    backfill_cache_enabled: bool
    backfill_keep_cache: bool
    backfill_skip_complete: bool
    api_quota_max_per_minute: int
    api_quota_max_per_hour: int
    api_quota_max_per_day: int
    random_seed: int
    model_n_jobs: int
    stacking_enabled: bool
    stacking_cv_folds: int
    stacking_l2_regularization: float
    stacking_max_iterations: int
    stacking_min_improvement: float
    stacking_city_shrinkage: float
    stacking_province_shrinkage: float
    stacking_max_base_models: int
    stacking_min_component_weight: float
    stacking_memory_recovery: bool
    stacking_city_balanced: bool
    stacking_recent_fold_weight: float
    day3_recent_fold_weight: float
    day3_extreme_sample_weight: float
    day3_extreme_threshold: float
    day3_stacking_l2_regularization: float
    local_experts_enabled: bool
    local_expert_candidates: tuple[str, ...]
    local_expert_min_rows: int
    local_expert_min_oof_rows: int
    local_expert_max_weight: float
    local_expert_weight_shrinkage: float
    local_expert_min_gain: float
    local_expert_max_cities_per_target: int
    day3_local_expert_max_weight: float
    day3_local_expert_max_cities: int
    calibration_month_shrinkage: float
    calibration_city_month_shrinkage: float
    calibration_recent_city_enabled: bool
    calibration_recent_city_shrinkage: float
    calibration_bias_shrinkage: float
    day3_calibration_enabled: bool
    day3_calibration_min_rmse_gain: float
    day3_calibration_max_abs_bias_regression: float
    champion_min_r2_gain: float
    champion_min_rmse_gain: float
    champion_max_rmse_regression: float
    champion_min_day3_r2_gain: float
    champion_max_abs_bias_regression: float
    hopsworks_candidate_model_name: str
    city_selection_enabled: bool
    city_selection_target_count: int
    city_selection_cv_folds: int
    city_selection_min_oof_rows: int
    city_selection_min_baseline_gain: float
    city_selection_mandatory: tuple[str, ...]
    matrix_strict: bool
    matrix_min_rows_per_city: int
    matrix_min_issue_dates: int
    matrix_min_numeric_features: int
    matrix_min_future_weather_coverage: float
    matrix_max_feature_missing_ratio: float
    matrix_max_duplicate_ratio: float
    minimum_test_r2: float
    maximum_test_rmse: float
    minimum_day3_r2: float
    generate_explainability: bool
    explainability_sample_rows: int
    minimum_provider_rows: int
    openaq_api_key: str
    openaq_radius_meters: int
    openweather_api_key: str
    aqicn_api_token: str
    provider_weight_model: float
    provider_weight_open_meteo: float
    provider_weight_openweather: float
    provider_weight_aqicn: float
    feature_store_backend: str
    hopsworks_host: str
    hopsworks_port: int
    hopsworks_project: str
    hopsworks_api_key: str
    hopsworks_feature_group_version: int
    hopsworks_online_enabled: bool
    hopsworks_required: bool
    hopsworks_sync_models: bool
    running_in_hopsworks: bool
    model_name: str
    model_version: int | None
    cities: dict[str, City]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def hourly_dir(self) -> Path:
        return self.data_dir / "hourly"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def provider_snapshots_dir(self) -> Path:
        return self.data_dir / "provider_snapshots"

    @property
    def api_cache_dir(self) -> Path:
        return self.data_dir / "api_cache"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def api_quota_state_path(self) -> Path:
        return self.reports_dir / "open_meteo_quota_state.json"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "artifacts" / "models" / self.model_name

    @property
    def selected_city_profile_path(self) -> Path:
        return self.project_root / "config" / "selected_training_cities.json"

    def city(self, key: str) -> City:
        normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in self.cities:
            raise KeyError(f"Unknown city '{key}'. Available: {', '.join(self.cities)}")
        return self.cities[normalized]


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _load_cities(path: Path) -> dict[str, City]:
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    output: dict[str, City] = {}
    for key, item in payload.get("cities", {}).items():
        output[key] = City(
            key=key,
            name=str(item["name"]),
            country=str(item.get("country", "Pakistan")),
            province=str(item["province"]),
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            timezone=str(item.get("timezone", "Asia/Karachi")),
            elevation_m=float(item.get("elevation_m", 0)),
            priority=int(item.get("priority", 999)),
        )
    if not output:
        raise ValueError(f"No cities were loaded from {path}")
    return dict(sorted(output.items(), key=lambda pair: pair[1].priority))


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model_version_value = os.getenv("MODEL_VERSION", "").strip()
    settings = Settings(
        project_root=PROJECT_ROOT,
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        default_city=os.getenv("AQI_CITY", "multan"),
        forecast_days=_int("FORECAST_DAYS", 3),
        backfill_days=_int("BACKFILL_DAYS", 730),
        api_timeout_seconds=_int("API_TIMEOUT_SECONDS", 120),
        request_pause_seconds=_float("REQUEST_PAUSE_SECONDS", 1.0),
        api_retry_attempts=_int("API_RETRY_ATTEMPTS", 6),
        chunk_days=max(1, _int("BACKFILL_CHUNK_DAYS", 14)),
        backfill_cache_enabled=_bool("BACKFILL_CACHE_ENABLED", True),
        backfill_keep_cache=_bool("BACKFILL_KEEP_CACHE", False),
        backfill_skip_complete=_bool("BACKFILL_SKIP_COMPLETE", True),
        api_quota_max_per_minute=max(1, _int("API_QUOTA_MAX_PER_MINUTE", 240)),
        api_quota_max_per_hour=max(1, _int("API_QUOTA_MAX_PER_HOUR", 3200)),
        api_quota_max_per_day=max(1, _int("API_QUOTA_MAX_PER_DAY", 8000)),
        random_seed=_int("RANDOM_SEED", 42),
        model_n_jobs=max(1, _int("MODEL_N_JOBS", 1)),
        stacking_enabled=_bool("STACKING_ENABLED", True),
        stacking_cv_folds=max(2, _int("STACKING_CV_FOLDS", 3)),
        stacking_l2_regularization=max(0.0, _float("STACKING_L2_REGULARIZATION", 2.0)),
        stacking_max_iterations=max(100, _int("STACKING_MAX_ITERATIONS", 2500)),
        stacking_min_improvement=max(0.0, _float("STACKING_MIN_IMPROVEMENT", 0.001)),
        stacking_city_shrinkage=max(0.0, _float("STACKING_CITY_SHRINKAGE", 60.0)),
        stacking_province_shrinkage=max(0.0, _float("STACKING_PROVINCE_SHRINKAGE", 120.0)),
        stacking_max_base_models=max(1, _int("STACKING_MAX_BASE_MODELS", 2)),
        stacking_min_component_weight=max(0.0, _float("STACKING_MIN_COMPONENT_WEIGHT", 0.02)),
        stacking_memory_recovery=_bool("STACKING_MEMORY_RECOVERY", True),
        stacking_city_balanced=_bool("STACKING_CITY_BALANCED", True),
        stacking_recent_fold_weight=max(1.0, _float("STACKING_RECENT_FOLD_WEIGHT", 1.5)),
        day3_recent_fold_weight=max(1.0, _float("DAY3_RECENT_FOLD_WEIGHT", 2.0)),
        day3_extreme_sample_weight=max(1.0, _float("DAY3_EXTREME_SAMPLE_WEIGHT", 1.25)),
        day3_extreme_threshold=min(500.0, max(0.0, _float("DAY3_EXTREME_THRESHOLD", 151.0))),
        day3_stacking_l2_regularization=max(0.0, _float("DAY3_STACKING_L2_REGULARIZATION", 1.25)),
        local_experts_enabled=_bool("LOCAL_EXPERTS_ENABLED", True),
        local_expert_candidates=tuple(
            item.strip() for item in os.getenv(
                "LOCAL_EXPERT_CANDIDATES", "ridge,hist_gradient,hist_gradient_long"
            ).split(",") if item.strip()
        ),
        local_expert_min_rows=max(240, _int("LOCAL_EXPERT_MIN_ROWS", 500)),
        local_expert_min_oof_rows=max(90, _int("LOCAL_EXPERT_MIN_OOF_ROWS", 180)),
        local_expert_max_weight=min(0.65, max(0.0, _float("LOCAL_EXPERT_MAX_WEIGHT", 0.35))),
        local_expert_weight_shrinkage=max(0.0, _float("LOCAL_EXPERT_WEIGHT_SHRINKAGE", 180.0)),
        local_expert_min_gain=max(0.0, _float("LOCAL_EXPERT_MIN_GAIN", 0.005)),
        local_expert_max_cities_per_target=max(1, _int("LOCAL_EXPERT_MAX_CITIES_PER_TARGET", 4)),
        day3_local_expert_max_weight=min(0.75, max(0.0, _float("DAY3_LOCAL_EXPERT_MAX_WEIGHT", 0.45))),
        day3_local_expert_max_cities=max(1, _int("DAY3_LOCAL_EXPERT_MAX_CITIES", 6)),
        calibration_month_shrinkage=max(0.0, _float("CALIBRATION_MONTH_SHRINKAGE", 180.0)),
        calibration_city_month_shrinkage=max(0.0, _float("CALIBRATION_CITY_MONTH_SHRINKAGE", 120.0)),
        calibration_recent_city_enabled=_bool("CALIBRATION_RECENT_CITY_ENABLED", True),
        calibration_recent_city_shrinkage=max(0.0, _float("CALIBRATION_RECENT_CITY_SHRINKAGE", 180.0)),
        calibration_bias_shrinkage=max(0.0, _float("CALIBRATION_BIAS_SHRINKAGE", 240.0)),
        day3_calibration_enabled=_bool("DAY3_CALIBRATION_ENABLED", True),
        day3_calibration_min_rmse_gain=max(0.0, _float("DAY3_CALIBRATION_MIN_RMSE_GAIN", 0.001)),
        day3_calibration_max_abs_bias_regression=max(0.0, _float("DAY3_CALIBRATION_MAX_ABS_BIAS_REGRESSION", 0.10)),
        champion_min_r2_gain=max(0.0, _float("CHAMPION_MIN_R2_GAIN", 0.0005)),
        champion_min_rmse_gain=max(0.0, _float("CHAMPION_MIN_RMSE_GAIN", 0.05)),
        champion_max_rmse_regression=max(0.0, _float("CHAMPION_MAX_RMSE_REGRESSION", 0.05)),
        champion_min_day3_r2_gain=max(0.0, _float("CHAMPION_MIN_DAY3_R2_GAIN", 0.0005)),
        champion_max_abs_bias_regression=max(0.0, _float("CHAMPION_MAX_ABS_BIAS_REGRESSION", 0.15)),
        hopsworks_candidate_model_name=os.getenv(
            "HOPSWORKS_CANDIDATE_MODEL_NAME", "pearls_aqi_daily_ensemble_challengers"
        ).strip(),
        city_selection_enabled=_bool("CITY_SELECTION_ENABLED", True),
        city_selection_target_count=max(2, _int("CITY_SELECTION_TARGET_COUNT", 8)),
        city_selection_cv_folds=max(2, _int("CITY_SELECTION_CV_FOLDS", 3)),
        city_selection_min_oof_rows=max(90, _int("CITY_SELECTION_MIN_OOF_ROWS", 450)),
        city_selection_min_baseline_gain=_float("CITY_SELECTION_MIN_BASELINE_GAIN", 0.02),
        city_selection_mandatory=tuple(
            item.strip().lower().replace("-", "_").replace(" ", "_")
            for item in os.getenv("CITY_SELECTION_MANDATORY", "karachi,multan").split(",")
            if item.strip()
        ),
        matrix_strict=_bool("MATRIX_STRICT", True),
        matrix_min_rows_per_city=max(100, _int("MATRIX_MIN_ROWS_PER_CITY", 1200)),
        matrix_min_issue_dates=max(90, _int("MATRIX_MIN_ISSUE_DATES", 500)),
        matrix_min_numeric_features=max(10, _int("MATRIX_MIN_NUMERIC_FEATURES", 60)),
        matrix_min_future_weather_coverage=min(1.0, max(0.0, _float("MATRIX_MIN_FUTURE_WEATHER_COVERAGE", 0.65))),
        matrix_max_feature_missing_ratio=min(1.0, max(0.0, _float("MATRIX_MAX_FEATURE_MISSING_RATIO", 0.95))),
        matrix_max_duplicate_ratio=min(1.0, max(0.0, _float("MATRIX_MAX_DUPLICATE_RATIO", 0.0))),
        minimum_test_r2=_float("MINIMUM_TEST_R2", 0.70),
        maximum_test_rmse=_float("MAXIMUM_TEST_RMSE", 30.0),
        minimum_day3_r2=_float("MINIMUM_DAY3_R2", 0.45),
        generate_explainability=_bool("GENERATE_EXPLAINABILITY", True),
        explainability_sample_rows=_int("EXPLAINABILITY_SAMPLE_ROWS", 180),
        minimum_provider_rows=_int("MINIMUM_PROVIDER_ROWS", 24),
        openaq_api_key=os.getenv("OPENAQ_API_KEY", "").strip(),
        openaq_radius_meters=_int("OPENAQ_RADIUS_METERS", 25000),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
        aqicn_api_token=os.getenv("AQICN_API_TOKEN", "").strip(),
        provider_weight_model=_float("PROVIDER_WEIGHT_MODEL", 0.55),
        provider_weight_open_meteo=_float("PROVIDER_WEIGHT_OPEN_METEO", 0.30),
        provider_weight_openweather=_float("PROVIDER_WEIGHT_OPENWEATHER", 0.10),
        provider_weight_aqicn=_float("PROVIDER_WEIGHT_AQICN", 0.05),
        feature_store_backend=os.getenv("FEATURE_STORE_BACKEND", "local").strip().lower(),
        hopsworks_host=os.getenv("HOPSWORKS_HOST", "").strip(),
        hopsworks_port=_int("HOPSWORKS_PORT", 443),
        hopsworks_project=os.getenv("HOPSWORKS_PROJECT", "").strip(),
        hopsworks_api_key=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        hopsworks_feature_group_version=_int("HOPSWORKS_FEATURE_GROUP_VERSION", 9),
        hopsworks_online_enabled=_bool("HOPSWORKS_ONLINE_ENABLED", False),
        hopsworks_required=_bool("HOPSWORKS_REQUIRED", False),
        hopsworks_sync_models=_bool("HOPSWORKS_SYNC_MODELS", True),
        running_in_hopsworks=_bool("RUNNING_IN_HOPSWORKS", False),
        model_name=os.getenv("MODEL_NAME", "pearls_aqi_daily_ensemble").strip(),
        model_version=int(model_version_value) if model_version_value else None,
        cities=_load_cities(PROJECT_ROOT / "config" / "cities.yaml"),
    )
    for directory in (
        settings.hourly_dir,
        settings.predictions_dir,
        settings.provider_snapshots_dir,
        settings.api_cache_dir,
        settings.reports_dir,
        settings.model_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
