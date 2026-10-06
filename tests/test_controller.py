from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.controller import LiveRequest, control_cycle
from app.telemetry import TelemetrySample, TelemetryStore, apply_sample
from engine.schemas import Scenario


@pytest.fixture
def advanced():
    return Scenario.model_validate_json(Path("examples/network-scenario.json").read_text())


def sample(scenario, stamp=None, **fields):
    return TelemetrySample(
        datacenter_id=scenario.datacenter_id, observed_at=stamp or datetime.now(UTC), **fields
    )


def test_utilization_fan_and_probe_mapping(advanced):
    reading = sample(
        advanced,
        rack_cpu_utilization_pct={"ROW-A": 50, "ROW-B": 75},
        rack_fan_rpm={"ROW-A": 9600},
        rack_inlet_temp_c={"ROW-A": 23},
        rack_outlet_temp_c={"ROW-A": 34},
    )
    updated = apply_sample(advanced, reading)
    assert updated.rack_clusters[0].it_load_kw == pytest.approx(185)
    assert updated.rack_clusters[1].it_load_kw == pytest.approx(145)
    assert updated.rack_clusters[0].server_fan_speed_pct == 80
    assert updated.initial_state["ROW-A"].inlet_temp_c == 23
    assert updated.initial_state["ROW-A"].outlet_temp_c == 34
    assert set(updated.initial_state) == {"ROW-A", "ROW-B"}


def test_cpu_requires_calibration_and_rejects_ambiguous_loads(scenario):
    reading = sample(scenario, rack_cpu_utilization_pct={"ROW-A": 50, "ROW-B": 75})
    with pytest.raises(ValueError, match="idle and rated"):
        apply_sample(scenario, reading)
    with pytest.raises(ValueError, match="one load source"):
        sample(scenario, rack_loads_kw={"ROW-A": 100}, rack_cpu_utilization_pct={"ROW-A": 40})


def test_live_cycle_staleness_weather_and_economizer(advanced, tmp_path):
    store = TelemetryStore(str(tmp_path / "trace.db"))
    now = datetime.now(UTC)
    request = LiveRequest(scenario=advanced, latitude=19, longitude=72)
    with pytest.raises(ValueError, match="No telemetry"):
        control_cycle(request, store)
    reading = sample(advanced, now, rack_loads_kw={"ROW-A": 120, "ROW-B": 100})
    store.append(reading)

    def cold(*_):
        return {"ambient_dry_bulb_c": 0, "ambient_relative_humidity": 60}

    cycle = control_cycle(request, store, weather_provider=cold, now=now)
    assert cycle["result"]["optimized"]["economizer_active"]
    assert cycle["result"]["optimized"]["it_kw"] == 220
    assert cycle["sample_age_seconds"] == 0
    with pytest.raises(ValueError, match="stale"):
        control_cycle(request, store, weather_provider=cold, now=now + timedelta(minutes=6))
    with pytest.raises(ValueError, match="future"):
        control_cycle(request, store, now=now - timedelta(minutes=2))

    def fail(*_):
        raise httpx.ConnectError("offline")

    with pytest.raises(ValueError, match="Weather unavailable"):
        control_cycle(request, store, weather_provider=fail, now=now)


def test_trace_api_and_control_websocket(advanced, tmp_path):
    with TestClient(create_app(str(tmp_path / "telemetry.db"))) as client:
        readings = [
            sample(
                advanced,
                datetime.now(UTC) - timedelta(seconds=offset),
                rack_loads_kw={"ROW-A": 150, "ROW-B": 100},
            ).model_dump(mode="json")
            for offset in [60, 30, 0]
        ]
        response = client.post("/api/v1/telemetry/batch", json={"samples": readings})
        assert response.status_code == 201
        assert response.json()["stored"] == 3
        history = client.get(f"/api/v1/telemetry/{advanced.datacenter_id}/history?limit=2")
        assert len(history.json()["samples"]) == 2
        assert history.json()["samples"][-1]["observed_at"] == readings[-1]["observed_at"]
        request = LiveRequest(scenario=advanced, latitude=19, longitude=72, use_weather=False)
        response = client.post("/api/v1/control/cycle", json=request.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json()["result"]["optimized"]["it_kw"] == 250
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(request.model_dump(mode="json"))
            assert websocket.receive_json()["status"] == "planned"
            websocket.send_json({"action": "stop"})
        request.scenario.datacenter_id = "EMPTY"
        assert (
            client.post("/api/v1/control/cycle", json=request.model_dump(mode="json")).status_code
            == 422
        )
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(request.model_dump(mode="json"))
            assert websocket.receive_json()["status"] == "skipped"
            websocket.send_json({"action": "stop"})
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"invalid": True})
            assert websocket.receive_json()["status"] == "error"
