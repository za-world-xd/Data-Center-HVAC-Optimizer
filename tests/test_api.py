import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.telemetry import TelemetrySample
from engine.weather import current_weather


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path / "telemetry.db"))) as value:
        yield value


def test_health_simulation_and_optimization(client, scenario):
    assert client.get("/health").json()["mode"] == "simulation"
    for path in ["simulate", "optimize"]:
        response = client.post(f"/api/v1/{path}", json=scenario.model_dump(mode="json"))
        assert response.status_code == 200, response.text
    assert client.get("/docs").status_code == 200


def test_validation_and_infeasibility(client, scenario):
    assert client.post("/api/v1/optimize", json={}).status_code == 422
    payload = scenario.model_dump(mode="json")
    payload["rack_clusters"][0]["design_airflow_m3_s"] = 0.1
    assert client.post("/api/v1/optimize", json=payload).status_code == 422


def test_telemetry_orders_by_observation_and_applies_loads(client, scenario):
    sample = {
        "datacenter_id": scenario.datacenter_id,
        "observed_at": "2026-11-06T10:00:00Z",
        "rack_loads_kw": {"ROW-A": 100, "ROW-B": 80},
    }
    assert client.get(f"/api/v1/telemetry/{scenario.datacenter_id}").status_code == 404
    assert (
        client.post(
            "/api/v1/optimize?use_latest_telemetry=true", json=scenario.model_dump(mode="json")
        ).status_code
        == 404
    )
    assert client.post("/api/v1/telemetry", json=sample).status_code == 201
    older = dict(sample, observed_at="2026-11-05T10:00:00Z", rack_loads_kw={"ROW-A": 200})
    client.post("/api/v1/telemetry", json=older)
    latest = client.get(f"/api/v1/telemetry/{scenario.datacenter_id}").json()
    assert latest["rack_loads_kw"] == sample["rack_loads_kw"]
    response = client.post(
        "/api/v1/optimize?use_latest_telemetry=true", json=scenario.model_dump(mode="json")
    )
    assert response.status_code == 200
    assert response.json()["optimized"]["it_kw"] == 180
    with client.websocket_connect(f"/ws/telemetry/{scenario.datacenter_id}") as websocket:
        assert websocket.receive_json()["latest"]["rack_loads_kw"] == sample["rack_loads_kw"]


def test_telemetry_mismatch_and_timezone(client, scenario):
    sample = {
        "datacenter_id": scenario.datacenter_id,
        "observed_at": "2026-11-06T10:00:00Z",
        "rack_loads_kw": {"UNKNOWN": 100},
    }
    client.post("/api/v1/telemetry", json=sample)
    assert (
        client.post(
            "/api/v1/optimize?use_latest_telemetry=true", json=scenario.model_dump(mode="json")
        ).status_code
        == 422
    )
    with pytest.raises(ValueError):
        TelemetrySample.model_validate(dict(sample, observed_at="2026-11-06T10:00:00"))


def test_weather_fetch_and_failure(client, monkeypatch):
    def get_weather(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 25,
                    "relative_humidity_2m": 55,
                    "time": "2026-11-06T10:00",
                }
            },
            request=httpx.Request("GET", "https://api.open-meteo.com"),
        )

    monkeypatch.setattr(httpx, "get", get_weather)
    assert current_weather(19, 72)["ambient_dry_bulb_c"] == 25
    assert client.get("/api/v1/weather?latitude=19&longitude=72").status_code == 200
    assert client.get("/api/v1/weather?latitude=190&longitude=72").status_code == 422

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", fail)
    assert client.get("/api/v1/weather?latitude=19&longitude=72").status_code == 502


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"current": None},
        {
            "current": {
                "temperature_2m": None,
                "relative_humidity_2m": 50,
                "time": "2026-11-06T10:00",
            }
        },
        {
            "current": {
                "temperature_2m": "nan",
                "relative_humidity_2m": 50,
                "time": "2026-11-06T10:00",
            }
        },
        {
            "current": {
                "temperature_2m": 20,
                "relative_humidity_2m": 150,
                "time": "2026-11-06T10:00",
            }
        },
        {"current": {"temperature_2m": 20, "relative_humidity_2m": 50, "time": "invalid"}},
    ],
)
def test_invalid_weather_is_an_actionable_upstream_error(client, monkeypatch, payload):
    def invalid_weather(*args, **kwargs):
        return httpx.Response(
            200, json=payload, request=httpx.Request("GET", "https://api.open-meteo.com")
        )

    monkeypatch.setattr(httpx, "get", invalid_weather)
    response = client.get("/api/v1/weather?latitude=19&longitude=72")
    assert response.status_code == 502
    assert "Weather service unavailable" in response.json()["detail"]
