import numpy as np
from scipy.optimize import minimize

from engine.physics import constraint_margins, evaluate
from engine.schemas import Controls, ForecastStep, OptimizationResult, PlanStep, Scenario


class InfeasibleScenario(ValueError):
    pass


def optimize(scenario: Scenario) -> OptimizationResult:
    weather = scenario.forecast or [
        ForecastStep(
            ambient_dry_bulb_c=scenario.ambient_dry_bulb_c,
            ambient_relative_humidity=scenario.ambient_relative_humidity,
        )
    ]
    scenarios = [
        scenario.model_copy(
            update={
                "ambient_dry_bulb_c": step.ambient_dry_bulb_c,
                "ambient_relative_humidity": step.ambient_relative_humidity,
                "rack_clusters": [
                    rack.model_copy(update={"it_load_kw": rack.it_load_kw * step.load_multiplier})
                    for rack in scenario.rack_clusters
                ],
            }
        )
        for step in weather
    ]
    initial = evaluate(scenario, scenario.baseline_controls)
    initial_inlets = (
        [scenario.initial_state[rack.cluster_id] for rack in initial.racks]
        if scenario.initial_state
        else initial.racks
        if scenario.network
        else [rack.inlet_temp_c for rack in initial.racks]
    )

    def rollout(vector):
        previous = initial_inlets
        plan, margins = [], []
        for i, step_scenario in enumerate(scenarios):
            air, fan, water = vector[3 * i : 3 * i + 3]
            controls = Controls(
                supply_air_temp_c=air, crah_fan_speed_pct=fan, chilled_water_supply_temp_c=water
            )
            plant = evaluate(step_scenario, controls, previous)
            steady = evaluate(step_scenario, controls)
            margins.extend(constraint_margins(plant, controls))
            margins.extend(constraint_margins(steady, controls))
            if i:
                # Limit changes between future planned steps; first action has no actuator model.
                difference = vector[3 * i : 3 * i + 3] - vector[3 * i - 3 : 3 * i]
                margins.extend(np.array([2, 20, 2]) - np.abs(difference))
            plan.append(PlanStep(step=i, control_setpoints=controls, plant=plant))
            previous = (
                plant.racks if scenario.network else [rack.inlet_temp_c for rack in plant.racks]
            )
        return plan, np.asarray(margins)

    def objective(vector):
        plan, _ = rollout(vector)
        return sum(step.plant.total_kw for step in plan) / len(plan)

    bounds = [(18, 26), (35, 100), (5, 18)] * len(scenarios)
    candidates = []
    for seed in ([20, 100, 8], [24, 100, 18], [22, 70, 14]):
        result = minimize(
            objective,
            np.tile(seed, len(scenarios)),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x: rollout(x)[1],
                }
            ],
            options={"maxiter": 150, "ftol": 1e-7},
        )
        plan, margins = rollout(result.x)
        if result.success and np.min(margins) >= -1e-6:
            candidates.append((result, plan))
    if not candidates:
        raise InfeasibleScenario(
            "No converged feasible plan. Reduce rack loads or increase "
            "design airflow; inspect the 27 C inlet and 15 K rise limits."
        )
    result, plan = min(candidates, key=lambda item: item[0].fun)
    previous = initial_inlets
    baseline_plan = []
    for step_scenario in scenarios:
        plant = evaluate(step_scenario, scenario.baseline_controls, previous)
        baseline_plan.append(plant)
        previous = plant.racks if scenario.network else [rack.inlet_temp_c for rack in plant.racks]
    best, baseline = plan[0].plant, baseline_plan[0]
    energy = (
        sum(old.total_kw - new.plant.total_kw for old, new in zip(baseline_plan, plan))
        * scenario.timestep_seconds
        / 3600
    )
    return OptimizationResult(
        status="converged",
        message="Locally optimized simulation; advisory setpoints only.",
        optimized_pue=best.pue,
        baseline_pue=baseline.pue,
        estimated_hourly_savings_usd=(baseline.total_kw - best.total_kw)
        * scenario.electricity_usd_per_kwh,
        horizon_energy_savings_kwh=energy,
        control_setpoints=plan[0].control_setpoints,
        max_rack_inlet_temp_c=max(r.inlet_temp_c for r in best.racks),
        within_temperature_limits=all(step.plant.within_temperature_limits for step in plan),
        baseline=baseline,
        optimized=best,
        plan=plan,
        iterations=int(result.nit),
    )
