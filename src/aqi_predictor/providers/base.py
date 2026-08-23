from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class APITrace:
    provider: str
    endpoint: str
    status_code: int | None
    elapsed_ms: float
    attempt: int
    response_bytes: int
    error: str | None = None
    rate_limited: bool = False
    retry_after_seconds: int | None = None
    from_cache: bool = False
    cost_units: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "attempt": self.attempt,
            "response_bytes": self.response_bytes,
            "error": self.error,
            "rate_limited": self.rate_limited,
            "retry_after_seconds": self.retry_after_seconds,
            "from_cache": self.from_cache,
            "cost_units": self.cost_units,
        }


class ProviderError(RuntimeError):
    pass


class RateLimitError(ProviderError):
    """Raised when a provider asks the client to stop until a later time."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int,
        trace: APITrace,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.trace = trace


class QuotaGuard:
    """Conservative, persistent request-budget guard.

    Open-Meteo documents free-service ceilings per minute, hour and day. This
    guard deliberately uses lower soft limits and records estimated weighted
    request units in a JSON ledger. It prevents this project from consuming the
    published ceiling by itself. A provider-side HTTP 429 can still occur when
    the same public IP is shared with unrelated traffic; RateLimitError handles
    that case and the backfill resumes from cache.
    """

    def __init__(
        self,
        state_path: Path | None,
        *,
        minute_limit: int,
        hour_limit: int,
        day_limit: int,
    ) -> None:
        self.state_path = state_path
        self.minute_limit = max(1, int(minute_limit))
        self.hour_limit = max(1, int(hour_limit))
        self.day_limit = max(1, int(day_limit))

    @staticmethod
    def _keys(now: datetime) -> tuple[str, str, str]:
        return (
            now.strftime("%Y-%m-%dT%H:%MZ"),
            now.strftime("%Y-%m-%dT%HZ"),
            now.strftime("%Y-%m-%d"),
        )

    def _empty(self, now: datetime) -> dict[str, Any]:
        minute_key, hour_key, day_key = self._keys(now)
        return {
            "minute": {"key": minute_key, "count": 0},
            "hour": {"key": hour_key, "count": 0},
            "day": {"key": day_key, "count": 0},
            "updated_at": now.isoformat(),
        }

    def _load(self, now: datetime) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return self._empty(now)
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._empty(now)
            return payload
        except (OSError, ValueError, TypeError):
            return self._empty(now)

    def _save(self, payload: dict[str, Any]) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=self.state_path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            temp_path = Path(handle.name)
        try:
            os.replace(temp_path, self.state_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _seconds_to_next_minute(now: datetime) -> int:
        target = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return max(1, int((target - now).total_seconds()) + 3)

    @staticmethod
    def _seconds_to_next_hour(now: datetime) -> int:
        target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return max(1, int((target - now).total_seconds()) + 15)

    @staticmethod
    def _seconds_to_next_day(now: datetime) -> int:
        target = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return max(1, int((target - now).total_seconds()) + 60)

    def acquire(self, units: int) -> None:
        units = max(1, int(units))
        if units > self.day_limit:
            raise ValueError(
                f"Single request cost {units} exceeds daily soft limit {self.day_limit}."
            )

        while True:
            now = datetime.now(UTC)
            minute_key, hour_key, day_key = self._keys(now)
            state = self._load(now)
            for period, key in (("minute", minute_key), ("hour", hour_key), ("day", day_key)):
                bucket = state.get(period)
                if not isinstance(bucket, dict) or bucket.get("key") != key:
                    state[period] = {"key": key, "count": 0}

            waits: list[tuple[str, int]] = []
            if int(state["minute"].get("count", 0)) + units > self.minute_limit:
                waits.append(("minute", self._seconds_to_next_minute(now)))
            if int(state["hour"].get("count", 0)) + units > self.hour_limit:
                waits.append(("hour", self._seconds_to_next_hour(now)))
            if int(state["day"].get("count", 0)) + units > self.day_limit:
                waits.append(("day", self._seconds_to_next_day(now)))

            if waits:
                period, wait_seconds = max(waits, key=lambda item: item[1])
                jitter = random.randint(1, 8)
                total_wait = wait_seconds + jitter
                LOGGER.warning(
                    "Local Open-Meteo %s budget reached; sleeping %s seconds before the next request.",
                    period,
                    total_wait,
                )
                time.sleep(total_wait)
                continue

            for period in ("minute", "hour", "day"):
                state[period]["count"] = int(state[period].get("count", 0)) + units
            state["updated_at"] = now.isoformat()
            self._save(state)
            return


class HTTPClient:
    def __init__(
        self,
        *,
        timeout_seconds: int = 120,
        attempts: int = 6,
        pause_seconds: float = 3.0,
        user_agent: str = "Pearls-AQI-Predictor/6.3",
        quota_state_path: Path | None = None,
        quota_minute_limit: int = 240,
        quota_hour_limit: int = 3200,
        quota_day_limit: int = 8000,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self.pause_seconds = max(0.0, float(pause_seconds))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_started: float | None = None
        self.quota_guard = QuotaGuard(
            quota_state_path,
            minute_limit=quota_minute_limit,
            hour_limit=quota_hour_limit,
            day_limit=quota_day_limit,
        )

    def _pace(self) -> None:
        """Apply a minimum interval between every physical HTTP request."""
        if self._last_request_started is not None and self.pause_seconds > 0:
            elapsed = time.monotonic() - self._last_request_started
            remaining = self.pause_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> int:
        value = response.headers.get("Retry-After", "").strip()
        if value:
            try:
                return max(1, int(float(value)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    return max(1, int((retry_at - datetime.now(UTC)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass

        reason = response.text.lower()
        now = datetime.now(UTC)
        if "next hour" in reason or "hourly" in reason:
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return max(60, int((next_hour - now).total_seconds()) + 75)
        if "tomorrow" in reason or "daily" in reason or "next day" in reason:
            next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return max(3600, int((next_day - now).total_seconds()) + 300)
        return 3600

    def get_json(
        self,
        provider: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cost_units: int = 1,
    ) -> tuple[dict[str, Any], APITrace]:
        last_error: Exception | None = None
        cost_units = max(1, int(cost_units))
        for attempt in range(1, self.attempts + 1):
            self.quota_guard.acquire(cost_units)
            self._pace()
            started = time.perf_counter()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000

                if response.status_code == 429:
                    wait_seconds = self._retry_after_seconds(response)
                    message = f"HTTP 429 rate limit: {response.text[:300]}"
                    trace = APITrace(
                        provider=provider,
                        endpoint=response.url,
                        status_code=429,
                        elapsed_ms=elapsed_ms,
                        attempt=attempt,
                        response_bytes=len(response.content),
                        error=message,
                        rate_limited=True,
                        retry_after_seconds=wait_seconds,
                        cost_units=cost_units,
                    )
                    raise RateLimitError(
                        message,
                        retry_after_seconds=wait_seconds,
                        trace=trace,
                    )

                if response.status_code in {500, 502, 503, 504}:
                    raise ProviderError(
                        f"Temporary HTTP {response.status_code}: {response.text[:200]}"
                    )
                response.raise_for_status()
                payload = response.json()
                return payload, APITrace(
                    provider=provider,
                    endpoint=response.url,
                    status_code=response.status_code,
                    elapsed_ms=elapsed_ms,
                    attempt=attempt,
                    response_bytes=len(response.content),
                    cost_units=cost_units,
                )
            except RateLimitError:
                raise
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                elapsed_ms = (time.perf_counter() - started) * 1000
                LOGGER.warning(
                    "%s request failed on attempt %s/%s: %s",
                    provider,
                    attempt,
                    self.attempts,
                    exc,
                )
                if attempt < self.attempts:
                    time.sleep(min(60.0, self.pause_seconds * (2 ** (attempt - 1))))
                else:
                    return {}, APITrace(
                        provider=provider,
                        endpoint=url,
                        status_code=getattr(getattr(exc, "response", None), "status_code", None),
                        elapsed_ms=elapsed_ms,
                        attempt=attempt,
                        response_bytes=0,
                        error=str(exc),
                        cost_units=cost_units,
                    )
        raise ProviderError(str(last_error))
