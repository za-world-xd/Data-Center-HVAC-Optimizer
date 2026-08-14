from dataclasses import dataclass
from functools import lru_cache
from math import exp

from scipy.optimize import brentq

from engine.network import WATER_CP, counterflow_effectiveness
from engine.schemas import TowerConfig


def saturated_enthalpy_kj_kg(temperature_c: float) -> float:
    """Sea-level moist air enthalpy; saturation pressure uses the Buck water correlation."""
    vapor_kpa = 0.61121 * exp(
        (18.678 - temperature_c / 234.5) * temperature_c / (257.14 + temperature_c)
    )
    ratio = 0.621945 * vapor_kpa / (101.325 - vapor_kpa)
    return 1.006 * temperature_c + ratio * (2501 + 1.86 * temperature_c)


@dataclass(frozen=True)
class TowerResult:
    leaving_water_c: float
    entering_water_c: float
    rejected_kw: float
    fan_kw: float
    ntu: float


@lru_cache(maxsize=4096)
def _tower(load_kw, wet_bulb_c, ua, water_flow, air_flow, design_fan_kw, speed_pct):
    """Merkel equivalent counterflow NTU model, neglecting evaporation mass loss."""
    speed = speed_pct / 100
    water_capacity = water_flow * WATER_CP
    air_flow *= speed
    incoming_h = saturated_enthalpy_kj_kg(wet_bulb_c)
    outgoing_h = incoming_h + load_kw / air_flow
    outgoing_wb = brentq(lambda t: saturated_enthalpy_kj_kg(t) - outgoing_h, wet_bulb_c, 95)
    cpe = (
        (outgoing_h - incoming_h) / (outgoing_wb - wet_bulb_c)
        if load_kw > 1e-8
        else (saturated_enthalpy_kj_kg(wet_bulb_c + 0.001) - incoming_h) / 0.001
    )
    effective_ua = ua * speed**0.8 * cpe / 1.006
    smallest, largest = sorted([water_capacity, air_flow * cpe])
    ntu = effective_ua / smallest
    effectiveness = counterflow_effectiveness(ntu, smallest / largest)
    entering = wet_bulb_c + load_kw / (effectiveness * smallest)
    leaving = entering - load_kw / water_capacity
    return TowerResult(leaving, entering, load_kw, design_fan_kw * speed**3, ntu)


def merkel_tower(load_kw: float, wet_bulb_c: float, config: TowerConfig) -> TowerResult:
    return _tower(
        max(0, load_kw),
        wet_bulb_c,
        config.ua_kw_k,
        config.water_mass_flow_kg_s,
        config.air_mass_flow_kg_s,
        config.design_fan_kw,
        config.fan_speed_pct,
    )
