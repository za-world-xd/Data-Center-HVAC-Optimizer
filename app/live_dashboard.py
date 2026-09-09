import os
from datetime import UTC, datetime

import httpx
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.charts import hotspot_rows, room_figure, time_series_figure
from app.controller import LiveRequest, TelemetryBatch, control_cycle
from app.telemetry import TelemetryStore
from engine.schemas import OptimizationResult, Scenario


def render_live(scenario: Scenario, latitude: float, longitude: float):
    st.header("Live advisory planning")
    st.caption(
        "Uses the shared local telemetry database. Incoming samples trigger new studies; "
        "setpoints remain advisory and do not operate equipment."
    )
    store = TelemetryStore(os.getenv("COOLTWIN_DB", "data/telemetry.db"))
    trace = st.file_uploader("Import telemetry trace JSON", type=["json"], key="trace_upload")
    if st.button("Import telemetry trace", disabled=trace is None):
        try:
            batch = TelemetryBatch.model_validate_json(trace.getvalue())
            if any(sample.datacenter_id != scenario.datacenter_id for sample in batch.samples):
                raise ValueError("Trace datacenter must match the configured scenario")
            store.append_many(batch.samples)
            st.success(f"Imported {len(batch.samples)} observations.")
        except (ValidationError, ValueError) as error:
            st.error(str(error))
    use_weather = st.checkbox("Refresh weather during live planning", value=True)
    enabled = st.toggle("Enable automatic advisory planning", value=False)
    request = LiveRequest(
        scenario=scenario, latitude=latitude, longitude=longitude, use_weather=use_weather
    )
    context = request.model_dump_json()
    if st.session_state.get("live_context") != context:
        st.session_state.pop("live_cycle", None)
        st.session_state.pop("live_history", None)
        st.session_state["live_context"] = context

    @st.fragment(run_every=5 if enabled else None)
    def monitor():
        sample = store.latest(scenario.datacenter_id)
        if sample is None:
            st.info("No telemetry yet. Import a trace or POST observations to /api/v1/telemetry.")
            return
        stamp = pd.Timestamp(sample["observed_at"])
        age = (datetime.now(UTC) - stamp.to_pydatetime()).total_seconds()
        st.caption(f"Latest observation: {sample['observed_at']} · age {age:.0f} s")
        if age > 300 or age < -30:
            st.warning("Latest telemetry is stale or future-dated. Live planning is paused.")
        history = store.history(scenario.datacenter_id, 200)
        load_rows = [
            {"Observed": row["observed_at"], **row["rack_loads_kw"]}
            for row in history
            if row["rack_loads_kw"]
        ]
        if load_rows:
            st.plotly_chart(
                time_series_figure(
                    pd.DataFrame(load_rows).set_index("Observed"), "Measured load (kW)"
                ),
                width="stretch",
            )
        cpu_rows = [
            {"Observed": row["observed_at"], **row.get("rack_cpu_utilization_pct", {})}
            for row in history
            if row.get("rack_cpu_utilization_pct")
        ]
        if cpu_rows:
            st.plotly_chart(
                time_series_figure(
                    pd.DataFrame(cpu_rows).set_index("Observed"), "CPU utilization (%)"
                ),
                width="stretch",
            )
        if enabled:
            try:
                with st.spinner("Replanning from telemetry and ambient conditions…"):
                    cycle = control_cycle(request, store)
                st.session_state["live_cycle"] = cycle
                points = st.session_state.setdefault("live_history", [])
                points.append(
                    {
                        "Computed": cycle["computed_at"],
                        "PUE": cycle["result"]["optimized_pue"],
                        "Total power (kW)": cycle["result"]["optimized"]["total_kw"],
                        "Peak inlet (°C)": cycle["result"]["max_rack_inlet_temp_c"],
                        "Chiller (kW)": cycle["result"]["optimized"]["chiller_kw"],
                        "CRAH fans (kW)": cycle["result"]["optimized"]["crah_fans_kw"],
                        "Tower (kW)": cycle["result"]["optimized"]["towers_kw"],
                        "Pumps (kW)": cycle["result"]["optimized"]["condenser_pumps_kw"],
                    }
                )
                del points[:-200]
            except (ValueError, httpx.HTTPError) as error:
                st.error(f"Cycle skipped: {error}")
        cycle = st.session_state.get("live_cycle")
        if cycle and cycle["scenario"]["datacenter_id"] == scenario.datacenter_id:
            result = OptimizationResult.model_validate(cycle["result"])
            current = Scenario.model_validate(cycle["scenario"])
            st.caption(
                f"Last successful plan: {cycle['computed_at']}. "
                "A skipped cycle leaves this saved plan unchanged."
            )
            st.write(
                "Economizer advisory:",
                "Free cooling" if result.optimized.economizer_active else "Mechanical cooling",
            )
            st.json(result.control_setpoints.model_dump())
            points = pd.DataFrame(st.session_state["live_history"])
            for columns, units in [
                (["PUE"], "PUE"),
                (["Total power (kW)"], "Power (kW)"),
                (["Chiller (kW)", "CRAH fans (kW)", "Tower (kW)", "Pumps (kW)"], "Power (kW)"),
            ]:
                st.plotly_chart(
                    time_series_figure(points.set_index("Computed")[columns], units),
                    width="stretch",
                )
            st.plotly_chart(room_figure(current, result), width="stretch", key="live_room")
            warnings = hotspot_rows(result, current.timestep_seconds)
            if warnings:
                st.warning("The current plan approaches an inlet or rack-rise limit.")
                st.dataframe(warnings, hide_index=True)
        elif not enabled:
            st.caption(
                "No plan for these inputs yet. Enable automatic advisory planning to calculate one."
            )

    monitor()
