from datetime import UTC, datetime
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from app.telemetry import TelemetrySample, TelemetryStore


def test_dashboard_study_and_controls(monkeypatch):
    app = AppTest.from_file(str(Path("app/main.py").resolve()), default_timeout=45).run()
    assert not app.exception
    assert "No study results" in app.caption[-1].value
    app.number_input(key="dry_bulb").set_value(25)
    app.number_input(key="humidity").set_value(55)
    app.slider[0].set_value(3)
    app.slider[1].set_value(15)
    app.button(key="optimize").click().run()
    assert not app.exception
    assert len(app.metric) == 3
    assert len(app.session_state["result"].plan) == 3
    app.number_input(key="dry_bulb").set_value(28).run()
    assert any("Inputs have changed" in warning.value for warning in app.warning)
    assert app.session_state["request"].ambient_dry_bulb_c == 25
    app.button(key="optimize").click().run()
    assert not any("Inputs have changed" in warning.value for warning in app.warning)
    assert app.session_state["request"].ambient_dry_bulb_c == 28
    app.radio(key="thermal_view").set_value("3D rack profile").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 3

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", fail)
    app.button(key="fetch_weather").click().run()
    assert "Weather unavailable" in app.error[0].value
    assert not app.exception


def test_detailed_models_room_views_and_live_cycle(monkeypatch, tmp_path):
    database = str(tmp_path / "dashboard.db")
    monkeypatch.setenv("COOLTWIN_DB", database)
    store = TelemetryStore(database)
    store.append(
        TelemetrySample(
            datacenter_id="DC-WEST-01",
            observed_at=datetime.now(UTC),
            rack_loads_kw={"ROW-A": 150, "ROW-B": 100},
        )
    )
    app = AppTest.from_file(str(Path("app/main.py").resolve()), default_timeout=45).run()
    app.checkbox[0].check()
    app.checkbox[1].check()
    app.button(key="optimize").click().run()
    assert not app.exception
    assert app.session_state["result"].optimized.tower_ntu > 0
    assert app.session_state["result"].optimized.coil_capacity_margin_kw >= -1e-6
    for view in ["Room contours", "3D room"]:
        app.radio(key="thermal_view").set_value(view).run()
        assert not app.exception
    app.radio(key="study_mode").set_value("Live telemetry").run()
    assert not app.exception
    app.checkbox[-1].uncheck()
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert app.session_state["live_cycle"]["result"]["optimized"]["it_kw"] == 250
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert "live_cycle" in app.session_state
    app.number_input(key="dry_bulb").set_value(28).run()
    assert not app.exception
    assert "live_cycle" not in app.session_state
    assert "live_history" not in app.session_state
    assert any("No plan for these inputs" in caption.value for caption in app.caption)
