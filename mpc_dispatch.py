import numpy as np
import cvxpy as cp
from typing import List, Tuple

__all__ = ["mpc_battery_action"]


def _price_vector(
    horizon: int,
    hp_hours: List[int],
    buy_hp: float,
    buy_hc: float,
    current_hour: int = 0,
) -> np.ndarray:
    """Return length-*horizon* vector of buy prices (€/kWh)."""
    return np.array(
        [
            buy_hp if ((current_hour + t) % 24) in hp_hours else buy_hc
            for t in range(horizon)
        ]
    )


def mpc_battery_action(
    *,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    price_buy_hp: float,
    price_buy_hc: float,
    price_sell: float,
    hp_hours: List[int],
    soc_init: float,
    batt_capacity: float,
    batt_power: float,
    soc_min: float,
    soc_max: float,
    eta_roundtrip: float,
    current_hour: int = 0,
) -> Tuple[float, float]:
    """Solve the MPC and return (charge_kwh, discharge_kwh) for given hour."""
    
    horizon = len(load_forecast)
    eta_c = np.sqrt(eta_roundtrip)
    eta_d = np.sqrt(eta_roundtrip)

    # Decision variables
    ch = cp.Variable(horizon, nonneg=True)  # charge (kWh)
    dis = cp.Variable(horizon, nonneg=True)  # discharge (kWh)
    soc = cp.Variable(horizon + 1)  # state of charge (kWh)
    imp = cp.Variable(horizon, nonneg=True)  # grid import (kWh)
    exp = cp.Variable(horizon, nonneg=True)  # grid export (kWh)

    # Constraints
    cons = [soc[0] == soc_init]
    for t in range(horizon):
        # Power limits (energy within the hour)
        cons += [ch[t] <= batt_power, dis[t] <= batt_power]
        # SoC dynamics
        cons += [soc[t + 1] == soc[t] + eta_c * ch[t] - dis[t] / eta_d]
        # SoC bounds
        cons += [
            soc[t + 1] >= soc_min * batt_capacity,
            soc[t + 1] <= soc_max * batt_capacity,
        ]
        # Energy balance
        cons += [pv_forecast[t] + dis[t] + imp[t] == load_forecast[t] + ch[t] + exp[t]]

    # Objective
    price_buy = _price_vector(
        horizon, hp_hours, price_buy_hp, price_buy_hc, current_hour
    )
    price_sell_vec = price_sell * np.ones(horizon)

    cost = cp.sum(cp.multiply(price_buy, imp) - cp.multiply(price_sell_vec, exp))
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-9)

    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        # Fallback – do nothing this hour
        return 0.0, 0.0

    charge_cmd = float(ch.value[0])
    discharge_cmd = float(dis.value[0])

    eps = 1e-6
    if charge_cmd > discharge_cmd + eps:
        discharge_cmd = 0.0
    elif discharge_cmd > charge_cmd + eps:
        charge_cmd = 0.0
    else:  # both are virtually zero
        charge_cmd = discharge_cmd = 0.0

    return charge_cmd, discharge_cmd
