import pytest
from pydantic import ValidationError

from engine.schemas import NetworkConfig, RackCluster, Scenario
from engine.weather import _cached_weather, cached_current_weather


@pytest.mark.parametrize(
    "fields",
    [
        {"idle_power_kw": 50},
        {"idle_power_kw": 50, "rated_power_kw": 40},
        {"x_m": 1},
    ],
)
def test_rejects_incomplete_physical_configuration(fields):
    with pytest.raises(ValidationError):
        RackCluster(cluster_id="A", it_load_kw=100, **fields)


def test_fan_curve_and_probe_rack_identifiers(scenario):
    with pytest.raises(ValidationError):
        NetworkConfig(shutoff_pressure_pa=100)
    payload = scenario.model_dump()
    payload["initial_state"] = {"UNKNOWN": {"inlet_temp_c": 20, "outlet_temp_c": 30}}
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_live_weather_cache(monkeypatch):
    calls = []

    def fetch(latitude, longitude):
        calls.append((latitude, longitude))
        return {"ambient_dry_bulb_c": 20, "ambient_relative_humidity": 50}

    monkeypatch.setattr("engine.weather.current_weather", fetch)
    _cached_weather.cache_clear()
    try:
        assert cached_current_weather(0, 0) == cached_current_weather(0, 0)
        assert len(calls) == 1
    finally:
        _cached_weather.cache_clear()
