from datetime import UTC, datetime
from functools import lru_cache
from time import monotonic

import httpx
from pydantic import Field

from engine.schemas import Model


class CurrentConditions(Model):
    temperature_2m: float = Field(ge=-20, le=50)
    relative_humidity_2m: float = Field(ge=5, le=100)
    time: datetime


def current_weather(latitude: float, longitude: float) -> dict:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude or longitude is outside its valid range")
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m",
            "timezone": "UTC",
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("current"), dict):
        raise ValueError("Weather response does not contain current conditions")
    current = CurrentConditions.model_validate(
        {
            key: payload["current"].get(key)
            for key in ["temperature_2m", "relative_humidity_2m", "time"]
        }
    )
    return {
        "ambient_dry_bulb_c": current.temperature_2m,
        "ambient_relative_humidity": current.relative_humidity_2m,
        "observed_at": current.time.isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": "Open-Meteo",
        "latitude": latitude,
        "longitude": longitude,
    }


@lru_cache(maxsize=128)
def _cached_weather(latitude: float, longitude: float, minute: int) -> dict:
    return current_weather(latitude, longitude)


def cached_current_weather(latitude: float, longitude: float) -> dict:
    """Limit live loops to one weather fetch per location per minute."""
    return dict(_cached_weather(latitude, longitude, int(monotonic() // 60)))
