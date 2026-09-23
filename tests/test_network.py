from pathlib import Path

import pytest

from engine.network import coil_capacity, counterflow_effectiveness, solve_air_network
from engine.optimizer import InfeasibleScenario, optimize
from engine.physics import evaluate
from engine.schemas import Controls, Scenario, ThermalState, TowerConfig
from engine.tower import merkel_tower


@pytest.fixture
def advanced():
    return Scenario.model_validate_json(Path("examples/network-scenario.json").read_text())


def test_parallel_network_mass_and_pressure_balance(advanced):
    controls = Controls()
    air = solve_air_network(advanced, controls)
    design = sum(r.design_airflow_m3_s for r in advanced.rack_clusters)
    config = advanced.network
    fan_pressure = (
        config.shutoff_pressure_pa * 0.85**2
        - (config.shutoff_pressure_pa - config.design_static_pressure_pa)
        * (air.supply_m3_s / design) ** 2
    )
    assert air.pressure_pa == pytest.approx(fan_pressure)
    assert air.supply_m3_s == pytest.approx(sum(air.fresh_flows_m3_s) + air.bypass_m3_s)
    for flow, rack in zip(air.branch_flows_m3_s, advanced.rack_clusters):
        resistance = config.design_static_pressure_pa / rack.design_airflow_m3_s**2
        assert resistance * flow**2 == pytest.approx(air.pressure_pa)
    slow = solve_air_network(advanced, Controls(crah_fan_speed_pct=50))
    assert slow.supply_m3_s < air.supply_m3_s


def test_coil_capacity_and_energy_balance(advanced):
    controls = Controls()
    plant = evaluate(advanced, controls)
    assert plant.coil_capacity_kw > 0
    assert plant.coil_capacity_margin_kw == pytest.approx(
        plant.coil_capacity_kw - plant.coil_load_kw
    )
    assert plant.coil_capacity_kw == pytest.approx(
        coil_capacity(
            advanced,
            plant.supply_airflow_m3_s,
            plant.coil_return_air_temp_c,
            controls.chilled_water_supply_temp_c,
        )
    )
    for rack in plant.racks:
        assert rack.sensible_heat_kw == pytest.approx(rack.extracted_heat_kw)
        assert rack.stored_heat_kw == pytest.approx(0, abs=1e-9)
    assert counterflow_effectiveness(1, 1) == pytest.approx(0.5)


def test_coupled_transient_conserves_energy(advanced):
    controls = Controls()
    previous = [ThermalState(inlet_temp_c=20, outlet_temp_c=25)] * 2
    plant = evaluate(advanced, controls, previous)
    for rack in plant.racks:
        assert rack.sensible_heat_kw == pytest.approx(rack.extracted_heat_kw + rack.stored_heat_kw)
        assert rack.stored_heat_kw > 0
    assert plant.coil_load_kw + sum(r.stored_heat_kw for r in plant.racks) == pytest.approx(
        0.98 * plant.it_kw + plant.crah_fans_kw
    )
    steady = evaluate(advanced, controls)
    advanced.timestep_seconds = 3600
    late = evaluate(advanced, controls, previous)
    assert abs(late.racks[0].inlet_temp_c - steady.racks[0].inlet_temp_c) < abs(
        plant.racks[0].inlet_temp_c - steady.racks[0].inlet_temp_c
    )


def test_merkel_tower_water_and_air_energy_balance():
    config = TowerConfig()
    tower = merkel_tower(500, 20, config)
    assert (tower.entering_water_c - tower.leaving_water_c) * 30 * 4.186 == pytest.approx(500)
    assert tower.entering_water_c > tower.leaving_water_c > 20
    assert tower.ntu > 0
    bigger = merkel_tower(500, 20, TowerConfig(ua_kw_k=300))
    assert bigger.leaving_water_c < tower.leaving_water_c
    assert merkel_tower(0, 20, config).leaving_water_c == pytest.approx(20)


def test_advanced_optimization_and_undersized_coil(advanced):
    result = optimize(advanced)
    assert result.optimized.coil_capacity_margin_kw >= -1e-6
    assert result.optimized.total_kw <= result.baseline.total_kw
    assert result.optimized.tower_heat_rejection_kw == pytest.approx(
        result.optimized.coil_load_kw + result.optimized.chiller_kw
    )
    advanced.network.coil_ua_kw_k = 0.01
    with pytest.raises(InfeasibleScenario):
        optimize(advanced)
