import json
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from app.charts import hotspot_rows, room_figure, time_series_figure
from app.live_dashboard import render_live
from engine.optimizer import InfeasibleScenario, optimize
from engine.schemas import ForecastStep, NetworkConfig, Scenario, TowerConfig
from engine.weather import current_weather

st.set_page_config(page_title="DC-CoolTwin | Cooling study", layout="wide")
st.html("""<style>
[data-testid="stCaptionContainer"] {color: #425b52; opacity: 1;}
button, a[href], summary, select, label {cursor: pointer;}
button {min-height: 44px;}
button:disabled {cursor: not-allowed;}
:focus-visible {outline: 3px solid #17634c; outline-offset: 3px;}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {animation: none !important; transition: none !important;}
}
</style>""")
st.title("DC-CoolTwin")
st.caption("Cooling study · Lumped thermal simulation · Advisory setpoints")
st.info(
    "Simulation only. Temperature checks cover 18 to 27 °C inlet and a 15 K rack rise. "
    "They do not establish ASHRAE compliance or authorize equipment changes."
)

with st.sidebar:
    st.header("Ambient conditions")
    latitude = st.number_input("Latitude (°)", -90.0, 90.0, 19.08)
    longitude = st.number_input("Longitude (°)", -180.0, 180.0, 72.88)
    if st.button("Fetch Open-Meteo weather", key="fetch_weather"):
        try:
            with st.spinner("Fetching ambient conditions…"):
                weather = current_weather(latitude, longitude)
            st.session_state["dry_bulb"] = float(weather["ambient_dry_bulb_c"])
            st.session_state["humidity"] = float(weather["ambient_relative_humidity"])
            st.session_state["weather_source"] = weather
        except (httpx.HTTPError, ValueError, KeyError):
            st.error("Weather unavailable. Enter conditions manually.")
    st.session_state.setdefault("dry_bulb", 32.5)
    st.session_state.setdefault("humidity", 45.0)
    dry = st.number_input("Dry-bulb temperature (°C)", -20.0, 50.0, key="dry_bulb")
    rh = st.number_input("Relative humidity (%)", 5.0, 100.0, key="humidity")
    if "weather_source" in st.session_state:
        st.caption(
            f"Open-Meteo observation: {st.session_state['weather_source']['observed_at']} UTC. "
            "Manual changes override fetched values."
        )
    tariff = st.number_input("Electricity tariff (USD/kWh)", 0.0, 10.0, 0.12, step=0.01)
    st.header("Planning horizon")
    horizon = st.slider("Steps of 5 minutes", 1, 6, 1)
    growth = st.slider("Load change by final step (%)", -50, 50, 10)

st.header("Rack clusters")
st.caption(
    "The initial rows are synthetic examples. Airflow is the full-speed design value. "
    "Uploaded scenarios replace rack rows and plant assumptions; ambient and forecast "
    "controls remain editable here."
)
uploaded = st.file_uploader("Load scenario JSON", type=["json"])
try:
    source = (
        uploaded.getvalue()
        if uploaded
        else (Path(__file__).resolve().parents[1] / "examples/scenario.json").read_bytes()
    )
    scenario = Scenario.model_validate_json(source)
except (ValidationError, ValueError) as error:
    st.error(f"Invalid scenario: {error}")
    st.stop()

with st.expander("Physics models"):
    network_enabled = st.checkbox(
        "Resolve airflow and CRAH coil", value=scenario.network is not None
    )
    tower_enabled = st.checkbox("Use Merkel tower model", value=scenario.tower is not None)
    if network_enabled:
        settings = scenario.network or NetworkConfig()
        ua = st.number_input("CRAH coil UA (kW/K)", 0.1, 10000.0, float(settings.coil_ua_kw_k))
        scenario.network = settings.model_copy(update={"coil_ua_kw_k": ua})
    else:
        scenario.network = None
    scenario.tower = (scenario.tower or TowerConfig()) if tower_enabled else None
    st.caption(
        "Detailed models use configured or illustrative fan curves, resistances, thermal "
        "capacities, water flow, and tower UA. Calibrate them before site studies. "
        "Upload examples/network-scenario.json for a complete example configuration."
    )
use_scenario_forecast = st.checkbox(
    "Use scenario forecast", value=bool(scenario.forecast), disabled=not scenario.forecast
)

rows = st.data_editor(
    pd.DataFrame([rack.model_dump(mode="json") for rack in scenario.rack_clusters]),
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_order=[
        "cluster_id",
        "it_load_kw",
        "server_count",
        "containment_type",
        "design_airflow_m3_s",
        "server_fan_speed_pct",
        "x_m",
        "y_m",
    ],
    column_config={
        "cluster_id": st.column_config.TextColumn("Cluster", required=True),
        "it_load_kw": st.column_config.NumberColumn("IT load (kW)", min_value=0.1),
        "server_count": st.column_config.NumberColumn("Servers", min_value=1, step=1),
        "containment_type": st.column_config.SelectboxColumn(
            "Containment", options=["COLD_AISLE", "HOT_AISLE", "NONE"], required=True
        ),
        "design_airflow_m3_s": st.column_config.NumberColumn(
            "Design airflow (m³/s)", min_value=0.1
        ),
        "server_fan_speed_pct": st.column_config.NumberColumn(
            "Server fan (%)", min_value=20, max_value=120
        ),
        "x_m": st.column_config.NumberColumn("Room x (m)", min_value=0),
        "y_m": st.column_config.NumberColumn("Room y (m)", min_value=0),
    },
)
workflow = st.radio(
    "Study mode", ["Manual study", "Live telemetry"], horizontal=True, key="study_mode"
)
selected_forecast = (
    scenario.forecast
    if use_scenario_forecast
    else [
        ForecastStep(
            load_multiplier=1 + growth / 100 * i / max(1, horizon - 1),
            ambient_dry_bulb_c=dry,
            ambient_relative_humidity=rh,
        )
        for i in range(horizon)
    ]
)
planning_seconds = scenario.timestep_seconds if use_scenario_forecast else 300
payload = scenario.model_dump()
payload.update(
    rack_clusters=rows.astype(object).where(pd.notna(rows), None).to_dict("records"),
    ambient_dry_bulb_c=dry,
    ambient_relative_humidity=rh,
    electricity_usd_per_kwh=tariff,
    timestep_seconds=planning_seconds,
    forecast=selected_forecast,
)
try:
    current_request = Scenario.model_validate(payload)
except ValidationError as error:
    current_request = None
    if not rows.empty:
        st.error(f"Correct the study inputs: {error}")
if workflow == "Live telemetry":
    if current_request is not None:
        render_live(current_request, latitude, longitude)
    elif rows.empty:
        st.warning("Add at least one rack cluster to run a cooling study.")
    st.stop()
if rows.empty:
    st.warning("Add at least one rack cluster to run a cooling study.")

if st.button(
    "Optimize cooling setpoints", key="optimize", type="primary", disabled=current_request is None
):
    st.session_state.pop("result", None)
    try:
        with st.spinner("Solving thermal constraints across the horizon…"):
            result = optimize(current_request)
        st.session_state["result"] = result
        st.session_state["request"] = current_request
    except (ValidationError, InfeasibleScenario, ValueError) as error:
        st.error(str(error))

if "result" not in st.session_state:
    st.caption("No study results yet. Review rack loads and optimize cooling setpoints.")
    st.stop()

result = st.session_state["result"]
if current_request != st.session_state["request"]:
    st.warning(
        "Inputs have changed. Results below belong to the last successful study; "
        "optimize again to update them."
    )
else:
    st.caption("Results match the current study inputs.")
left, middle, right = st.columns(3)
left.metric(
    "Modeled PUE",
    f"{result.optimized_pue:.3f}",
    f"{result.optimized_pue - result.baseline_pue:.3f} vs baseline",
    delta_color="inverse",
)
middle.metric("Peak inlet", f"{result.max_rack_inlet_temp_c:.1f} °C")
right.metric("First-step savings", f"${result.estimated_hourly_savings_usd:.2f}/h")

st.header("Rack thermal profile")
view = st.radio(
    "Thermal view",
    ["2D temperatures", "3D rack profile", "Room contours", "3D room"],
    horizontal=True,
    key="thermal_view",
)
racks = result.optimized.racks
names = [r.cluster_id for r in racks]
temperatures = [[r.inlet_temp_c for r in racks], [r.outlet_temp_c for r in racks]]
if view == "2D temperatures":
    figure = go.Figure(
        go.Heatmap(
            z=temperatures,
            x=names,
            y=["Inlet", "Outlet"],
            colorscale="Cividis",
            colorbar_title="°C",
        )
    )
    for row, position in enumerate(["Inlet", "Outlet"]):
        for column, name in enumerate(names):
            figure.add_annotation(
                x=name,
                y=position,
                text=f"{temperatures[row][column]:.1f} °C",
                showarrow=False,
                font={"color": "#172d27"},
                bgcolor="#ffffff",
                borderpad=3,
            )
elif view == "3D rack profile":
    figure = go.Figure()
    for index, rack in enumerate(racks):
        figure.add_trace(
            go.Scatter3d(
                x=[index, index],
                y=[0, 1],
                z=[rack.inlet_temp_c, rack.outlet_temp_c],
                mode="lines+markers",
                name=rack.cluster_id,
                marker={
                    "size": 6,
                    "color": [rack.inlet_temp_c, rack.outlet_temp_c],
                    "colorscale": "Cividis",
                    "cmin": 18,
                    "cmax": 45,
                },
            )
        )
    figure.update_layout(
        scene={
            "xaxis": {"tickvals": list(range(len(racks))), "ticktext": names},
            "yaxis": {"tickvals": [0, 1], "ticktext": ["Inlet", "Outlet"]},
            "zaxis_title": "Temperature (°C)",
        }
    )
else:
    figure = room_figure(st.session_state["request"], result, view == "3D room")
figure.update_layout(
    height=380,
    margin={"l": 0, "r": 0, "t": 15, "b": 0},
    font={"color": "#172d27"},
)
st.plotly_chart(figure, width="stretch")
st.caption(
    "Each column represents one lumped cluster, not a spatial CFD contour. "
    "3D lines connect inlet and outlet temperatures, not physical room geometry."
)
if view in ["Room contours", "3D room"]:
    positioned = all(r.x_m is not None for r in st.session_state["request"].rack_clusters)
    st.caption(
        "Room positions: "
        + (
            "configured coordinates"
            if positioned
            else "illustrative row layout for racks without coordinates"
        )
        + ". Contours interpolate modeled inlet values, not CFD. 3D rack envelopes "
        "are illustrative 0.6 × 1.2 × 2 m boxes."
    )
warnings = hotspot_rows(result, st.session_state["request"].timestep_seconds)
if warnings:
    st.warning(
        "Forecast approaches a thermal limit: inlet at least 26 °C or rack rise at least 14 K."
    )
    st.dataframe(warnings, hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([r.model_dump() for r in racks]), hide_index=True, width="stretch")

st.header("Plant power and setpoints")
power_fields = [
    "it_kw",
    "chiller_kw",
    "crah_fans_kw",
    "condenser_pumps_kw",
    "towers_kw",
    "auxiliary_kw",
]
st.bar_chart(
    pd.DataFrame(
        {
            "Baseline (kW)": [getattr(result.baseline, k) for k in power_fields],
            "Optimized (kW)": [getattr(result.optimized, k) for k in power_fields],
        },
        index=[k.removesuffix("_kw").replace("_", " ") for k in power_fields],
    ),
    stack=False,
)
st.json(result.control_setpoints.model_dump())
st.write(
    "Water-side economizer:",
    "Available in model"
    if result.optimized.economizer_active
    else "Mechanical cooling required in model",
)
st.caption(
    f"Wet-bulb estimate: {result.optimized.wet_bulb_c:.1f} °C. "
    f"Horizon energy reduction: {result.horizon_energy_savings_kwh:.2f} kWh."
)
if result.optimized.coil_capacity_kw is not None:
    st.write(
        {
            "Plenum pressure (Pa)": result.optimized.plenum_pressure_pa,
            "Supply airflow (m³/s)": result.optimized.supply_airflow_m3_s,
            "Bypass airflow (m³/s)": result.optimized.bypass_airflow_m3_s,
            "Coil capacity (kW)": result.optimized.coil_capacity_kw,
            "Coil load (kW)": result.optimized.coil_load_kw,
        }
    )
if result.optimized.tower_ntu is not None:
    st.write(
        {
            "Tower leaving water (°C)": result.optimized.tower_leaving_water_temp_c,
            "Tower NTU": result.optimized.tower_ntu,
            "Tower rejected heat (kW)": result.optimized.tower_heat_rejection_kw,
            "Chiller PLR": result.optimized.chiller_plr,
            "Chiller kW/ton": result.optimized.chiller_kw_per_ton,
        }
    )
plan_table = pd.DataFrame(
    [
        {
            "Minute": (step.step + 1) * st.session_state["request"].timestep_seconds / 60,
            "PUE": step.plant.pue,
            "Total power (kW)": step.plant.total_kw,
            "Peak inlet (°C)": max(r.inlet_temp_c for r in step.plant.racks),
        }
        for step in result.plan
    ]
).set_index("Minute")
st.plotly_chart(time_series_figure(plan_table[["PUE"]], "PUE"), width="stretch")
st.plotly_chart(time_series_figure(plan_table[["Total power (kW)"]], "Power (kW)"), width="stretch")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Minute": (step.step + 1) * st.session_state["request"].timestep_seconds / 60,
                "PUE": step.plant.pue,
                "Air supply (°C)": step.control_setpoints.supply_air_temp_c,
                "Fan (%)": step.control_setpoints.crah_fan_speed_pct,
                "Water supply (°C)": step.control_setpoints.chilled_water_supply_temp_c,
            }
            for step in result.plan
        ]
    ),
    hide_index=True,
    width="stretch",
)
st.download_button(
    "Download study JSON",
    json.dumps(
        {
            "scenario": st.session_state["request"].model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        },
        indent=2,
    ),
    file_name="cooling-study.json",
    mime="application/json",
)
with st.expander("Model assumptions"):
    st.write(
        "98% sensible IT heat; constant air density and heat capacity; cubic CRAH fan power; "
        "empirical chiller COP. The basic mode uses prescribed rack airflow and a one-node "
        "inlet lag. The optional network solves pressure-driven rack and bypass flow, "
        "coupled chassis/plenum energy balances, and CRAH coil capacity. The optional "
        "Merkel-equivalent tower uses configured UA and flow rates with negligible "
        "evaporation mass loss. Measured temperature probes can initialize the thermal "
        "states. Forecast checks cover transient and steady temperatures. Humidity limits, "
        "redundancy, equipment sequencing, and CFD remain outside this model."
    )
