import asyncio
import os

import httpx
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.controller import LiveRequest, TelemetryBatch, control_cycle
from app.telemetry import TelemetrySample, TelemetryStore, apply_sample
from engine.optimizer import InfeasibleScenario, optimize
from engine.physics import evaluate
from engine.schemas import OptimizationResult, PlantResult, Scenario
from engine.weather import current_weather


def create_app(database_path=None):
    api = FastAPI(
        title="DC-CoolTwin",
        version="0.1.0",
        description="Advisory thermal simulations. No equipment control.",
    )
    store = TelemetryStore(database_path or os.getenv("COOLTWIN_DB", "data/telemetry.db"))
    api.state.telemetry = store

    @api.get("/health")
    def health():
        return {"status": "ok", "mode": "simulation"}

    @api.post("/api/v1/simulate", response_model=PlantResult)
    def simulate(scenario: Scenario):
        try:
            previous = (
                [scenario.initial_state[r.cluster_id] for r in scenario.rack_clusters]
                if scenario.initial_state
                else None
            )
            return evaluate(scenario, scenario.baseline_controls, previous)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @api.post("/api/v1/optimize", response_model=OptimizationResult)
    def optimization(scenario: Scenario, use_latest_telemetry: bool = False):
        if use_latest_telemetry:
            sample = store.latest(scenario.datacenter_id)
            if sample is None:
                raise HTTPException(404, "No telemetry for this datacenter")
            try:
                scenario = apply_sample(scenario, TelemetrySample.model_validate(sample))
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        try:
            return optimize(scenario)
        except (InfeasibleScenario, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @api.post("/api/v1/control/cycle")
    def cycle(request: LiveRequest):
        try:
            return control_cycle(request, store)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @api.websocket("/ws/control")
    async def control_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            # One configuration per connection. Closing the connection stops replanning.
            request = LiveRequest.model_validate(await websocket.receive_json())
            while True:
                try:
                    result = await run_in_threadpool(control_cycle, request, store)
                    await websocket.send_json({"status": "planned", **result})
                except ValueError as error:
                    await websocket.send_json({"status": "skipped", "error": str(error)})
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=5)
                    if not isinstance(message, dict):
                        raise ValueError("Control commands must be JSON objects")
                    if message.get("action") == "stop":
                        await websocket.close()
                        return
                except TimeoutError:
                    pass
        except ValueError as error:
            await websocket.send_json({"status": "error", "error": str(error)})
            await websocket.close(code=1008)
        except (WebSocketDisconnect, RuntimeError):
            return

    @api.post("/api/v1/telemetry", status_code=201)
    def ingest(sample: TelemetrySample):
        store.append(sample)
        return {"stored": True}

    @api.post("/api/v1/telemetry/batch", status_code=201)
    def ingest_trace(batch: TelemetryBatch):
        store.append_many(batch.samples)
        return {"stored": len(batch.samples)}

    @api.get("/api/v1/telemetry/{datacenter_id}/history")
    def history(datacenter_id: str, limit: int = Query(default=200, ge=1, le=1000)):
        return {"samples": store.history(datacenter_id, limit)}

    @api.get("/api/v1/telemetry/{datacenter_id}")
    def latest(datacenter_id: str):
        value = store.latest(datacenter_id)
        if value is None:
            raise HTTPException(404, "No telemetry for this datacenter")
        return value

    @api.get("/api/v1/weather")
    def weather(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180)):
        try:
            return current_weather(latitude, longitude)
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise HTTPException(
                502, "Weather service unavailable or returned invalid data"
            ) from error

    @api.websocket("/ws/telemetry/{datacenter_id}")
    async def telemetry_stream(websocket: WebSocket, datacenter_id: str):
        await websocket.accept()
        try:
            while True:
                sample = await run_in_threadpool(store.latest, datacenter_id)
                await websocket.send_json({"latest": sample})
                await asyncio.sleep(2)
        except (WebSocketDisconnect, RuntimeError):
            return

    return api


app = create_app()
