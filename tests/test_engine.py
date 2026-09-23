import math

import pytest
from pydantic import ValidationError

from engine.optimizer import InfeasibleScenario, optimize
from engine.physics import constraint_margins, evaluate, fan_power_kw, wet_bulb_c
from engine.schemas import Controls, ForecastStep, Scenario


def test_heat_balance_closure(scenario):
    plant = evaluate(scenario, Controls())
    for rack in plant.racks:
        assert rack.extracted_heat_kw == pytest.approx(rack.sensible_heat_kw, rel=0.001)
    assert sum(r.sensible_heat_kw for r in plant.racks) == pytest.approx(0.98 * plant.it_kw)
    assert plant.total_kw == pytest.approx(
        sum(
            getattr(plant, key)
            for key in [
                "it_kw",
                "chiller_kw",
                "crah_fans_kw",
                "condenser_pumps_kw",
                "towers_kw",
                "auxiliary_kw",
            ]
        )
    )


def test_fan_affinity_laws():
    assert fan_power_kw(20, 50) / fan_power_kw(20, 100) == pytest.approx(0.125)


def test_mixing_analytic_solution(scenario):
    controls = Controls(crah_fan_speed_pct=100)
    rack = evaluate(scenario, controls).racks[0]
    assert rack.inlet_temp_c == pytest.approx(20 + 0.08 * rack.delta_t_c / 0.92)


def test_optimizer_feasibility_and_savings(scenario):
    result = optimize(scenario)
    assert result.optimized.total_kw <= result.baseline.total_kw
    assert result.optimized_pue >= 1
    assert result.within_temperature_limits
    assert min(constraint_margins(result.optimized, result.control_setpoints)) >= -1e-6
    assert result.estimated_hourly_savings_usd == pytest.approx(
        (result.baseline.total_kw - result.optimized.total_kw) * 0.12
    )


def test_ashrae_constraint_violation(scenario):
    payload = scenario.model_dump()
    payload["rack_clusters"][0]["design_airflow_m3_s"] = 0.1
    with pytest.raises(InfeasibleScenario):
        optimize(Scenario.model_validate(payload))


def test_forecast_checks_each_step_and_ramps(scenario):
    scenario.forecast = [
        ForecastStep(ambient_dry_bulb_c=32, ambient_relative_humidity=45, load_multiplier=load)
        for load in [1, 1.2, 0.9]
    ]
    result = optimize(scenario)
    assert len(result.plan) == 3
    for step in result.plan:
        assert step.plant.within_temperature_limits
        assert min(constraint_margins(step.plant, step.control_setpoints)) >= -1e-6
    assert result.plan[1].plant.it_kw == pytest.approx(370 * 1.2)
    for old, new in zip(result.plan, result.plan[1:]):
        assert (
            abs(old.control_setpoints.crah_fan_speed_pct - new.control_setpoints.crah_fan_speed_pct)
            <= 20 + 1e-6
        )


def test_transient_relaxes_toward_steady_state(scenario):
    control = Controls()
    steady = evaluate(scenario, control).racks[0].inlet_temp_c
    transient = evaluate(scenario, control, [30, 30]).racks[0].inlet_temp_c
    assert transient == pytest.approx(steady + (30 - steady) * math.exp(-300 / 180))


def test_economizer_and_weather(scenario):
    scenario.ambient_dry_bulb_c = 0
    cold = evaluate(scenario, Controls())
    assert cold.economizer_active
    assert cold.chiller_kw == 0
    assert wet_bulb_c(30, 100) == pytest.approx(30, abs=0.1)


@pytest.mark.parametrize(
    "field,value",
    [("ambient_relative_humidity", 0), ("ambient_dry_bulb_c", float("nan")), ("rack_clusters", [])],
)
def test_rejects_invalid_inputs(scenario, field, value):
    payload = scenario.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_duplicate_ids_rejected(scenario):
    payload = scenario.model_dump()
    payload["rack_clusters"][1]["cluster_id"] = payload["rack_clusters"][0]["cluster_id"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)
