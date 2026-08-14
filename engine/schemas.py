from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Containment(StrEnum):
    COLD_AISLE = "COLD_AISLE"
    HOT_AISLE = "HOT_AISLE"
    NONE = "NONE"


class RackCluster(Model):
    cluster_id: str = Field(min_length=1, max_length=80)
    it_load_kw: float = Field(gt=0, le=2000)
    server_count: int = Field(default=80, ge=1, le=10000)
    containment_type: Containment = Containment.COLD_AISLE
    design_airflow_m3_s: float = Field(default=20, gt=0, le=500)
    server_fan_speed_pct: float = Field(default=100, ge=20, le=120)
    nominal_fan_rpm: float = Field(default=12000, gt=0, le=50000)
    idle_power_kw: float | None = Field(default=None, ge=0, le=2000)
    rated_power_kw: float | None = Field(default=None, gt=0, le=2000)
    airflow_resistance_pa_s2_m6: float | None = Field(default=None, gt=0, le=1e8)
    x_m: float | None = Field(default=None, ge=0, le=1000)
    y_m: float | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def power_curve(self):
        if (self.idle_power_kw is None) != (self.rated_power_kw is None):
            raise ValueError("Supply both idle_power_kw and rated_power_kw for utilization traces")
        if self.rated_power_kw is not None and self.idle_power_kw >= self.rated_power_kw:
            raise ValueError("idle_power_kw must be below rated_power_kw")
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError("Rack coordinates require both x_m and y_m")
        return self


class NetworkConfig(Model):
    design_static_pressure_pa: float = Field(default=300, gt=0, le=3000)
    shutoff_pressure_pa: float = Field(default=600, gt=0, le=6000)
    bypass_design_fraction: float = Field(default=0.08, ge=0, le=0.5)
    coil_ua_kw_k: float = Field(default=250, gt=0, le=10000)
    chilled_water_mass_flow_kg_s: float = Field(default=30, gt=0, le=1000)
    chassis_heat_capacity_kj_k: float = Field(default=5000, gt=0, le=1e7)
    plenum_heat_capacity_kj_k: float = Field(default=10000, gt=0, le=1e7)

    @model_validator(mode="after")
    def fan_curve(self):
        if self.shutoff_pressure_pa <= self.design_static_pressure_pa:
            raise ValueError("Fan shutoff pressure must exceed design static pressure")
        return self


class TowerConfig(Model):
    ua_kw_k: float = Field(default=150, gt=0, le=10000)
    water_mass_flow_kg_s: float = Field(default=30, gt=0, le=1000)
    air_mass_flow_kg_s: float = Field(default=50, gt=0, le=1000)
    design_fan_kw: float = Field(default=12, gt=0, le=1000)
    fan_speed_pct: float = Field(default=100, ge=20, le=100)
    heat_exchanger_approach_k: float = Field(default=2, ge=0, le=10)


class ThermalState(Model):
    inlet_temp_c: float = Field(ge=-20, le=100)
    outlet_temp_c: float = Field(ge=-20, le=120)


class Controls(Model):
    supply_air_temp_c: float = Field(default=20, ge=18, le=26)
    crah_fan_speed_pct: float = Field(default=85, ge=35, le=100)
    chilled_water_supply_temp_c: float = Field(default=8, ge=5, le=18)


class ForecastStep(Model):
    load_multiplier: float = Field(default=1, ge=0.1, le=2)
    ambient_dry_bulb_c: float = Field(ge=-20, le=50)
    ambient_relative_humidity: float = Field(ge=5, le=100)


class Scenario(Model):
    datacenter_id: str = Field(default="DC-DEMO", min_length=1, max_length=80)
    ambient_dry_bulb_c: float = Field(default=32.5, ge=-20, le=50)
    ambient_relative_humidity: float = Field(default=45, ge=5, le=100)
    rack_clusters: list[RackCluster] = Field(min_length=1, max_length=64)
    electricity_usd_per_kwh: float = Field(default=0.12, ge=0, le=10)
    auxiliary_kw: float = Field(default=8, ge=0, le=10000)
    baseline_controls: Controls = Field(default_factory=Controls)
    forecast: list[ForecastStep] = Field(default_factory=list, max_length=12)
    timestep_seconds: float = Field(default=300, ge=30, le=3600)
    thermal_time_constant_seconds: float = Field(default=180, ge=10, le=3600)
    network: NetworkConfig | None = None
    tower: TowerConfig | None = None
    initial_state: dict[str, ThermalState] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_clusters(self):
        ids = [rack.cluster_id for rack in self.rack_clusters]
        if len(set(ids)) != len(ids):
            raise ValueError("cluster_id values must be unique")
        if self.initial_state and set(self.initial_state) != set(ids):
            raise ValueError("initial_state must contain every configured cluster exactly once")
        return self


class RackResult(Model):
    cluster_id: str
    inlet_temp_c: float
    outlet_temp_c: float
    delta_t_c: float
    recirculation_ratio: float
    bypass_ratio: float
    sensible_heat_kw: float
    extracted_heat_kw: float
    airflow_m3_s: float = 0
    fresh_airflow_m3_s: float = 0
    pressure_drop_pa: float = 0
    stored_heat_kw: float = 0


class PlantResult(Model):
    pue: float
    it_kw: float
    chiller_kw: float
    crah_fans_kw: float
    condenser_pumps_kw: float
    towers_kw: float
    auxiliary_kw: float
    total_kw: float
    wet_bulb_c: float
    economizer_active: bool
    within_temperature_limits: bool
    racks: list[RackResult]
    supply_airflow_m3_s: float = 0
    bypass_airflow_m3_s: float = 0
    plenum_pressure_pa: float = 0
    coil_capacity_kw: float | None = None
    coil_load_kw: float = 0
    coil_return_air_temp_c: float | None = None
    chilled_water_return_temp_c: float | None = None
    coil_capacity_margin_kw: float | None = None
    chiller_plr: float = 0
    chiller_kw_per_ton: float = 0
    tower_leaving_water_temp_c: float | None = None
    tower_heat_rejection_kw: float = 0
    tower_ntu: float | None = None


class PlanStep(Model):
    step: int
    control_setpoints: Controls
    plant: PlantResult


class OptimizationResult(Model):
    status: str
    message: str
    optimized_pue: float
    baseline_pue: float
    estimated_hourly_savings_usd: float
    horizon_energy_savings_kwh: float
    control_setpoints: Controls
    max_rack_inlet_temp_c: float
    within_temperature_limits: bool
    baseline: PlantResult
    optimized: PlantResult
    plan: list[PlanStep]
    iterations: int


FinitePositive = Annotated[float, Field(gt=0, le=2000, allow_inf_nan=False)]
