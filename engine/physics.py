from math import atan, exp, sqrt

from scipy.optimize import brentq

from engine.network import WATER_CP, coil_capacity, resolve_rack_temperatures
from engine.schemas import Controls, PlantResult, RackResult, Scenario
from engine.tower import merkel_tower

AIR_DENSITY_KG_M3 = 1.2
AIR_CP_KJ_KG_K = 1.005
IT_SENSIBLE_FRACTION = 0.98
RECIRCULATION = {"COLD_AISLE": 0.08, "HOT_AISLE": 0.05, "NONE": 0.22}


def wet_bulb_c(dry_bulb_c: float, relative_humidity: float) -> float:
    """Stull approximation; screening estimate, not a psychrometric design calculation."""
    t, rh = dry_bulb_c, relative_humidity
    value = (
        t * atan(0.151977 * sqrt(rh + 8.313659))
        + atan(t + rh)
        - atan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * atan(0.023101 * rh)
        - 4.686035
    )
    return min(t, value)


def fan_power_kw(design_kw: float, speed_pct: float) -> float:
    return design_kw * (speed_pct / 100) ** 3


def _basic_racks(scenario: Scenario, controls: Controls, previous_inlets=None):
    speed = controls.crah_fan_speed_pct / 100
    racks = []
    for index, rack in enumerate(scenario.rack_clusters):
        flow = rack.design_airflow_m3_s * speed
        heat = rack.it_load_kw * IT_SENSIBLE_FRACTION
        delta = heat / (AIR_DENSITY_KG_M3 * flow * AIR_CP_KJ_KG_K)
        alpha = min(0.65, RECIRCULATION[rack.containment_type] + 0.12 * (1 - speed))
        inlet = controls.supply_air_temp_c + alpha * delta / (1 - alpha)
        if previous_inlets is not None:
            decay = exp(-scenario.timestep_seconds / scenario.thermal_time_constant_seconds)
            old_inlet = getattr(previous_inlets[index], "inlet_temp_c", previous_inlets[index])
            inlet += (old_inlet - inlet) * decay
        racks.append(
            RackResult(
                cluster_id=rack.cluster_id,
                inlet_temp_c=inlet,
                outlet_temp_c=inlet + delta,
                delta_t_c=delta,
                recirculation_ratio=alpha,
                bypass_ratio=max(0, 0.12 * speed),
                sensible_heat_kw=heat,
                extracted_heat_kw=flow * AIR_DENSITY_KG_M3 * AIR_CP_KJ_KG_K * delta,
                airflow_m3_s=flow,
                fresh_airflow_m3_s=flow * (1 - alpha),
            )
        )
    return racks


def evaluate(scenario: Scenario, controls: Controls, previous_inlets=None) -> PlantResult:
    speed = controls.crah_fan_speed_pct / 100
    network = None
    if scenario.network:
        network, racks = resolve_rack_temperatures(scenario, controls, previous_inlets)
    else:
        racks = _basic_racks(scenario, controls, previous_inlets)
    it = sum(r.it_load_kw for r in scenario.rack_clusters)
    fans = fan_power_kw(
        sum(r.design_airflow_m3_s * 0.8 for r in scenario.rack_clusters),
        controls.crah_fan_speed_pct,
    )
    cooling = max(0, sum(r.extracted_heat_kw for r in racks) + fans)
    wb = wet_bulb_c(scenario.ambient_dry_bulb_c, scenario.ambient_relative_humidity)
    # Water-side economizer needs a 5 K approach across tower and heat exchanger.
    economizer = wb + 5 <= controls.chilled_water_supply_temp_c
    plr = min(1, cooling / max(500, it * 1.3))

    def chiller_power(condenser_water_c):
        lift = max(8, condenser_water_c + 5 - controls.chilled_water_supply_temp_c)
        cop = max(2, min(10, 7.5 - 0.10 * lift - 0.8 * (1 - plr) ** 2))
        return cooling / cop

    tower = None
    if scenario.tower:
        free_cooling = merkel_tower(cooling, wb, scenario.tower)
        economizer = (
            free_cooling.leaving_water_c + scenario.tower.heat_exchanger_approach_k
            <= controls.chilled_water_supply_temp_c
        )
        chiller = (
            0
            if economizer
            else brentq(
                lambda power: (
                    power
                    - chiller_power(
                        merkel_tower(cooling + power, wb, scenario.tower).leaving_water_c
                    )
                ),
                0,
                max(1, cooling / 2),
            )
        )
        tower = merkel_tower(cooling + chiller, wb, scenario.tower)
    else:
        chiller = 0 if economizer else chiller_power(wb + 2)
    pumps = 0.018 * cooling
    towers = tower.fan_kw if tower else (cooling + chiller) * (0.008 + 0.0003 * max(0, wb - 10))
    supply_flow = (
        network.supply_m3_s
        if network
        else sum(r.design_airflow_m3_s * speed for r in scenario.rack_clusters)
    )
    return_air = controls.supply_air_temp_c + cooling / (
        AIR_DENSITY_KG_M3 * AIR_CP_KJ_KG_K * supply_flow
    )
    capacity = (
        coil_capacity(scenario, supply_flow, return_air, controls.chilled_water_supply_temp_c)
        if scenario.network
        else None
    )
    total = it + fans + chiller + pumps + towers + scenario.auxiliary_kw
    safe = all(18 - 1e-6 <= r.inlet_temp_c <= 27 + 1e-6 and r.delta_t_c <= 15 + 1e-6 for r in racks)
    return PlantResult(
        pue=total / it,
        it_kw=it,
        chiller_kw=chiller,
        crah_fans_kw=fans,
        condenser_pumps_kw=pumps,
        towers_kw=towers,
        auxiliary_kw=scenario.auxiliary_kw,
        total_kw=total,
        wet_bulb_c=wb,
        economizer_active=economizer,
        within_temperature_limits=safe,
        racks=racks,
        supply_airflow_m3_s=supply_flow,
        bypass_airflow_m3_s=network.bypass_m3_s if network else 0,
        plenum_pressure_pa=network.pressure_pa if network else 0,
        coil_capacity_kw=capacity,
        coil_load_kw=cooling,
        coil_return_air_temp_c=return_air,
        chilled_water_return_temp_c=(
            controls.chilled_water_supply_temp_c
            + cooling / (scenario.network.chilled_water_mass_flow_kg_s * WATER_CP)
            if scenario.network
            else None
        ),
        coil_capacity_margin_kw=capacity - cooling if capacity is not None else None,
        chiller_plr=plr,
        chiller_kw_per_ton=chiller / cooling * 3.51685284 if cooling else 0,
        tower_leaving_water_temp_c=tower.leaving_water_c if tower else None,
        tower_heat_rejection_kw=cooling + chiller,
        tower_ntu=tower.ntu if tower else None,
    )


def constraint_margins(plant: PlantResult, controls: Controls) -> list[float]:
    margins = [controls.supply_air_temp_c - controls.chilled_water_supply_temp_c - 3]
    if plant.coil_capacity_margin_kw is not None:
        margins.append(plant.coil_capacity_margin_kw)
    for rack in plant.racks:
        margins.extend([rack.inlet_temp_c - 18, 27 - rack.inlet_temp_c, 15 - rack.delta_t_c])
    return margins
