from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from engine.schemas import Controls, RackResult, Scenario

AIR_DENSITY = 1.2
AIR_CP = 1.005
WATER_CP = 4.186
CONTAINMENT_LEAKAGE = {"COLD_AISLE": 0.08, "HOT_AISLE": 0.05, "NONE": 0.22}


@dataclass
class AirNetwork:
    pressure_pa: float
    supply_m3_s: float
    bypass_m3_s: float
    branch_flows_m3_s: list[float]
    server_flows_m3_s: list[float]
    fresh_flows_m3_s: list[float]


def counterflow_effectiveness(ntu: float, capacity_ratio: float) -> float:
    if abs(capacity_ratio - 1) < 1e-8:
        return ntu / (1 + ntu)
    exponential = exp(-ntu * (1 - capacity_ratio))
    return (1 - exponential) / (1 - capacity_ratio * exponential)


def solve_air_network(scenario: Scenario, controls: Controls) -> AirNetwork:
    """Intersect a quadratic CRAH curve with parallel rack and bypass resistances."""
    config = scenario.network
    design = sum(r.design_airflow_m3_s for r in scenario.rack_clusters)
    resistances = [
        r.airflow_resistance_pa_s2_m6 or config.design_static_pressure_pa / r.design_airflow_m3_s**2
        for r in scenario.rack_clusters
    ]
    leak_conductance = (
        design * config.bypass_design_fraction / sqrt(config.design_static_pressure_pa)
    )
    conductance = sum(1 / sqrt(k) for k in resistances) + leak_conductance
    curve = (config.shutoff_pressure_pa - config.design_static_pressure_pa) / design**2
    shutoff = config.shutoff_pressure_pa * (controls.crah_fan_speed_pct / 100) ** 2
    pressure = brentq(lambda p: p + curve * (conductance * sqrt(p)) ** 2 - shutoff, 0, shutoff)
    branches = [sqrt(pressure / k) for k in resistances]
    servers = [r.design_airflow_m3_s * r.server_fan_speed_pct / 100 for r in scenario.rack_clusters]
    fresh = [
        (1 - CONTAINMENT_LEAKAGE[r.containment_type]) * min(branch, server)
        for r, branch, server in zip(scenario.rack_clusters, branches, servers)
    ]
    supply = conductance * sqrt(pressure)
    return AirNetwork(pressure, supply, supply - sum(fresh), branches, servers, fresh)


def resolve_rack_temperatures(scenario: Scenario, controls: Controls, previous=None):
    network = solve_air_network(scenario, controls)
    config = scenario.network
    racks = []
    for index, (rack, flow, fresh) in enumerate(
        zip(scenario.rack_clusters, network.server_flows_m3_s, network.fresh_flows_m3_s)
    ):
        capacity = AIR_DENSITY * AIR_CP * flow
        alpha = 1 - fresh / flow
        heat = 0.98 * rack.it_load_kw
        rise = heat / capacity
        inlet = controls.supply_air_temp_c + alpha * rise / (1 - alpha)
        outlet = inlet + rise
        if previous is not None:
            old = previous[index]
            old_inlet, old_outlet = (
                (old.inlet_temp_c, old.outlet_temp_c)
                if hasattr(old, "inlet_temp_c")
                else (old, old + rise)
            )
            # C dT/dt balances supply/recirculated air and chassis heat as two coupled nodes.
            matrix = np.array(
                [
                    [
                        -capacity / config.plenum_heat_capacity_kj_k,
                        alpha * capacity / config.plenum_heat_capacity_kj_k,
                    ],
                    [
                        capacity / config.chassis_heat_capacity_kj_k,
                        -capacity / config.chassis_heat_capacity_kj_k,
                    ],
                ]
            )
            state = np.array([inlet, outlet])
            state += expm(matrix * scenario.timestep_seconds) @ (
                np.array([old_inlet, old_outlet]) - state
            )
            inlet, outlet = state
        extracted = fresh * AIR_DENSITY * AIR_CP * (outlet - controls.supply_air_temp_c)
        racks.append(
            RackResult(
                cluster_id=rack.cluster_id,
                inlet_temp_c=inlet,
                outlet_temp_c=outlet,
                delta_t_c=outlet - inlet,
                recirculation_ratio=alpha,
                bypass_ratio=network.bypass_m3_s / network.supply_m3_s,
                sensible_heat_kw=heat,
                extracted_heat_kw=extracted,
                stored_heat_kw=heat - extracted,
                airflow_m3_s=flow,
                fresh_airflow_m3_s=fresh,
                pressure_drop_pa=network.pressure_pa,
            )
        )
    return network, racks


def coil_capacity(
    scenario: Scenario, supply_flow: float, return_air_c: float, water_supply_c: float
) -> float:
    config = scenario.network
    air_capacity = AIR_DENSITY * AIR_CP * supply_flow
    water_capacity = config.chilled_water_mass_flow_kg_s * WATER_CP
    smallest, largest = sorted([air_capacity, water_capacity])
    effectiveness = counterflow_effectiveness(config.coil_ua_kw_k / smallest, smallest / largest)
    return effectiveness * smallest * max(0, return_air_c - water_supply_c)
