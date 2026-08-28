from datetime import UTC, datetime

import httpx
from pydantic import Field

from app.telemetry import TelemetrySample, TelemetryStore, apply_sample
from engine.optimizer import optimize
from engine.schemas import Model, Scenario
from engine.weather import cached_current_weather


class LiveRequest(Model):
    scenario: Scenario
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    use_weather: bool = True
    max_sample_age_seconds: float = Field(default=300, ge=1, le=86400)


class TelemetryBatch(Model):
    samples: list[TelemetrySample] = Field(min_length=1, max_length=1000)


def control_cycle(
    request: LiveRequest, store: TelemetryStore, weather_provider=None, now: datetime | None = None
) -> dict:
    sample = store.latest(request.scenario.datacenter_id)
    if sample is None:
        raise ValueError("No telemetry for this datacenter")
    sample = TelemetrySample.model_validate(sample)
    now = now or datetime.now(UTC)
    age = (now - sample.observed_at).total_seconds()
    if age > request.max_sample_age_seconds or age < -30:
        raise ValueError("Telemetry is stale or more than 30 seconds in the future")
    scenario = apply_sample(request.scenario, sample)
    weather = None
    if request.use_weather:
        try:
            weather = (weather_provider or cached_current_weather)(
                request.latitude, request.longitude
            )
            payload = scenario.model_dump()
            payload.update(
                ambient_dry_bulb_c=weather["ambient_dry_bulb_c"],
                ambient_relative_humidity=weather["ambient_relative_humidity"],
            )
            # A current observation is held across the horizon; it is not a weather forecast.
            for step in payload["forecast"]:
                step.update(
                    ambient_dry_bulb_c=weather["ambient_dry_bulb_c"],
                    ambient_relative_humidity=weather["ambient_relative_humidity"],
                )
            scenario = Scenario.model_validate(payload)
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise ValueError(
                "Weather unavailable; cycle skipped rather than using stale weather"
            ) from error
    result = optimize(scenario)
    return {
        "observed_at": sample.observed_at.isoformat(),
        "computed_at": now.isoformat(),
        "sample_age_seconds": age,
        "weather": weather,
        "scenario": scenario.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
