import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from engine.schemas import FinitePositive, Model, Scenario

Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Temperature = Annotated[float, Field(ge=-20, le=120, allow_inf_nan=False)]
FanRPM = Annotated[float, Field(gt=0, le=50000, allow_inf_nan=False)]


class TelemetrySample(Model):
    datacenter_id: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    rack_loads_kw: dict[str, FinitePositive] = Field(default_factory=dict, max_length=64)
    rack_cpu_utilization_pct: dict[str, Percentage] = Field(default_factory=dict, max_length=64)
    rack_fan_rpm: dict[str, FanRPM] = Field(default_factory=dict, max_length=64)
    rack_inlet_temp_c: dict[str, Temperature] = Field(default_factory=dict, max_length=64)
    rack_outlet_temp_c: dict[str, Temperature] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def load_source(self):
        if not self.rack_loads_kw and not self.rack_cpu_utilization_pct:
            raise ValueError("Supply rack loads or CPU utilization")
        if self.rack_loads_kw.keys() & self.rack_cpu_utilization_pct.keys():
            raise ValueError("Use one load source per rack: measured power or CPU utilization")
        return self

    @field_validator("observed_at")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("observed_at requires a timezone")
        return value


class TelemetryStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS samples "
                "(id INTEGER PRIMARY KEY, dc TEXT, stamp REAL, payload TEXT)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS dc_stamp ON samples(dc, stamp)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append(self, sample: TelemetrySample):
        self.append_many([sample])

    def append_many(self, samples: list[TelemetrySample]):
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO samples(dc, stamp, payload) VALUES (?, ?, ?)",
                [
                    (sample.datacenter_id, sample.observed_at.timestamp(), sample.model_dump_json())
                    for sample in samples
                ],
            )
            for datacenter_id in {sample.datacenter_id for sample in samples}:
                connection.execute(
                    "DELETE FROM samples WHERE dc = ? AND id NOT IN "
                    "(SELECT id FROM samples WHERE dc = ? "
                    "ORDER BY stamp DESC, id DESC LIMIT 1000)",
                    (datacenter_id, datacenter_id),
                )

    def history(self, datacenter_id: str, limit: int = 200):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM samples WHERE dc = ? ORDER BY stamp DESC, id DESC LIMIT ?",
                (datacenter_id, min(1000, max(1, limit))),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def latest(self, datacenter_id: str):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM samples WHERE dc = ? ORDER BY stamp DESC, id DESC LIMIT 1",
                (datacenter_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None


def apply_sample(scenario: Scenario, sample: TelemetrySample) -> Scenario:
    ids = {rack.cluster_id for rack in scenario.rack_clusters}
    load_ids = sample.rack_loads_kw.keys() | sample.rack_cpu_utilization_pct.keys()
    if scenario.datacenter_id != sample.datacenter_id or ids != load_ids:
        raise ValueError("Telemetry must match the datacenter and every configured rack cluster")
    for values in [sample.rack_fan_rpm, sample.rack_inlet_temp_c, sample.rack_outlet_temp_c]:
        if not set(values) <= ids:
            raise ValueError("Measurements reference an unknown rack cluster")
    racks = []
    for rack in scenario.rack_clusters:
        values = rack.model_dump()
        if rack.cluster_id in sample.rack_loads_kw:
            values["it_load_kw"] = sample.rack_loads_kw[rack.cluster_id]
        else:
            if rack.rated_power_kw is None:
                raise ValueError(f"{rack.cluster_id}: CPU traces require idle and rated power")
            utilization = sample.rack_cpu_utilization_pct[rack.cluster_id] / 100
            values["it_load_kw"] = rack.idle_power_kw + utilization * (
                rack.rated_power_kw - rack.idle_power_kw
            )
        if rack.cluster_id in sample.rack_fan_rpm:
            values["server_fan_speed_pct"] = (
                100 * sample.rack_fan_rpm[rack.cluster_id] / rack.nominal_fan_rpm
            )
        racks.append(type(rack).model_validate(values))
    payload = scenario.model_dump()
    payload["rack_clusters"] = racks
    updated = Scenario.model_validate(payload)
    if sample.rack_inlet_temp_c or sample.rack_outlet_temp_c:
        from engine.physics import evaluate

        plant = evaluate(updated, updated.baseline_controls)
        payload["initial_state"] = {
            rack.cluster_id: {
                "inlet_temp_c": sample.rack_inlet_temp_c.get(rack.cluster_id, rack.inlet_temp_c),
                "outlet_temp_c": sample.rack_outlet_temp_c.get(
                    rack.cluster_id,
                    sample.rack_inlet_temp_c.get(rack.cluster_id, rack.inlet_temp_c)
                    + rack.delta_t_c,
                ),
            }
            for rack in plant.racks
        }
        updated = Scenario.model_validate(payload)
    return updated
