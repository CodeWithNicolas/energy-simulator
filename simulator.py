import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pathlib
import matplotlib.pyplot as plt
from get_irradiation import get_hourly_irradiation
from mpc_dispatch import mpc_battery_action
from plot_raw_data import plot_grid_import_bar, plot_self_consumption_bar
from plot_raw_data import plot_single_series_bar


@dataclass
class CustomerSite:
    latitude: float
    longitude: float
    load_curve_kwh: np.ndarray


@dataclass
class PVSystem:
    peak_power_kw: float
    dc_ac_eff: float
    capex_eur: float


@dataclass
class BatterySystem:
    capacity_kwh: float
    max_power_kw: float
    soc_min: float = 0.1
    soc_max: float = 0.9
    roundtrip_eff: float = 0.92
    capex_eur: float = 0.0


@dataclass
class EconomicParams:
    buy_price_hp: float
    buy_price_hc: float
    hp_hours: List[int]
    sell_price_premium: float
    sell_price_oa: float
    inflation_rate: float = 0.02
    service_revenue_eur: float = 120


@dataclass
class SimulationConfig:
    years: int = 20
    pv_degradation: float = 0.005
    batt_degradation: float = 0.02
    batt_replace_year: int = 12


def pv_production_hourly(
    irradiance_kw_m2: np.ndarray, pv: PVSystem, year_idx: int, cfg: SimulationConfig
) -> np.ndarray:
    """Return hourly AC PV production (kWh) for a given calendar *year_idx*."""
    derate = (1.0 - cfg.pv_degradation) ** year_idx
    return irradiance_kw_m2 * pv.peak_power_kw * pv.dc_ac_eff * derate


def simulate_period(
    site: CustomerSite,
    pv: PVSystem,
    batt: BatterySystem,
    econ: EconomicParams,
    irradiance_kw_m2: np.ndarray,
    cfg: SimulationConfig,
    period_hours: int,
    year_idx: int = 0,
    smart: bool = False,
    simple_hp_discharge: bool = False,
) -> Tuple[float, float, dict]:
    """Simulate a period (year or 24h) -> (net_cash €, self‑consumption ratio, details)."""

    load = site.load_curve_kwh[:period_hours]
    pv_hour = pv_production_hourly(irradiance_kw_m2[:period_hours], pv, year_idx, cfg)

    # Price escalation
    price_factor = (1 + econ.inflation_rate) ** year_idx
    buy_hp = econ.buy_price_hp * price_factor
    buy_hc = econ.buy_price_hc * price_factor
    sell_price = (
        econ.sell_price_oa
    ) * price_factor

    # Battery state
    soc = batt.capacity_kwh * (batt.soc_min + batt.soc_max) / 2  # start mid‑range
    capacity_end_of_year = batt.capacity_kwh * (1 - cfg.batt_degradation) ** max(
        0, year_idx - 1
    )

    cash = 0.0
    pv_self = 0.0
    soc_series = []
    grid_import_series = []
    grid_export_series = []
    batt_charge_series = []
    batt_discharge_series = []

    for h in range(period_hours):
        pv_gen = pv_hour[h]
        demand = load[h]

        # Step 1 – direct self‑consumption
        direct_use = min(pv_gen, demand)
        surplus = pv_gen - direct_use
        residual = demand - direct_use
        pv_self += direct_use

        batt_charge = 0.0
        batt_discharge = 0.0

        # Step 2 – battery dispatch
        if batt.capacity_kwh > 0:
            hour = h % 24
            if smart:
                if simple_hp_discharge and hour in econ.hp_hours:
                    # Simple favour discharge during HP
                    discharge_possible = min(
                        batt.max_power_kw, soc - batt.soc_min * capacity_end_of_year
                    )
                    discharge = min(discharge_possible, residual)
                    residual -= discharge
                    soc -= discharge / batt.roundtrip_eff
                    batt_discharge = discharge
                else:
                    H = min(24, period_hours - h)  # horizon
                    load_fc = (
                        load[h : h + H]
                        if h + H <= period_hours
                        else np.pad(load[h:], (0, h + H - period_hours))
                    )
                    pv_fc = (
                        pv_hour[h : h + H]
                        if h + H <= period_hours
                        else np.pad(pv_hour[h:], (0, h + H - period_hours))
                    )

                    charge_cmd, discharge_cmd = mpc_battery_action(
                        load_forecast=load_fc,
                        pv_forecast=pv_fc,
                        price_buy_hp=buy_hp,
                        price_buy_hc=buy_hc,
                        price_sell=sell_price,
                        hp_hours=econ.hp_hours,
                        soc_init=soc,
                        batt_capacity=capacity_end_of_year,
                        batt_power=batt.max_power_kw,
                        soc_min=batt.soc_min,
                        soc_max=batt.soc_max,
                        eta_roundtrip=batt.roundtrip_eff,
                        current_hour=h % 24,
                    )

                    # Execute the first-hour decision
                    if discharge_cmd > 0:
                        discharge_possible = min(
                            discharge_cmd,
                            residual,
                            batt.max_power_kw,
                            soc - batt.soc_min * capacity_end_of_year,
                        )
                        residual -= discharge_possible
                        soc -= discharge_possible / batt.roundtrip_eff
                        batt_discharge = discharge_possible
                    elif charge_cmd > 0:
                        charge_room = capacity_end_of_year * batt.soc_max - soc
                        charge_possible = min(
                            charge_cmd, surplus, batt.max_power_kw, charge_room
                        )
                        surplus -= charge_possible
                        soc += charge_possible * batt.roundtrip_eff
                        batt_charge = charge_possible
            else:
                # charge with surplus if room
                charge_room = capacity_end_of_year * batt.soc_max - soc
                charge_possible = min(batt.max_power_kw, surplus, charge_room)
                surplus -= charge_possible
                soc += charge_possible * batt.roundtrip_eff
                batt_charge = charge_possible

        # Step 3 – grid interaction
        price_buy = buy_hp if (h % 24) in econ.hp_hours else buy_hc
        cash -= residual * price_buy  # grid import costs
        cash += surplus * sell_price  # grid export revenue
        grid_import_series.append(residual)
        grid_export_series.append(surplus)
        soc_series.append(soc)
        batt_charge_series.append(batt_charge)
        batt_discharge_series.append(batt_discharge)

    # Annual fixed items (only for year simulation)
    services = econ.service_revenue_eur if smart and period_hours >= 8760 else 0.0
    cash += services

    sc_ratio = pv_self / load.sum() if load.sum() > 0 else 0.0
    details = {
        "soc": np.array(soc_series),
        "grid_import": np.array(grid_import_series),
        "grid_export": np.array(grid_export_series),
        "batt_charge": np.array(batt_charge_series),
        "batt_discharge": np.array(batt_discharge_series),
        "pv_self": pv_self,
        "load": load,
        "pv": pv_hour,
    }
    return cash, sc_ratio, details

def simulate_24h_mpc(
    site: CustomerSite,
    pv: PVSystem,
    batt: BatterySystem,
    econ: EconomicParams,
    irradiance_kw_m2: np.ndarray,
    cfg: SimulationConfig,
    smart: bool = True,
    simple_hp_discharge: bool = False,
    soc_init: float = None,
) -> dict:
    """Simulate a single 24h period using MPC. Returns detailed time series."""
    # Optionally allow custom initial SoC
    if soc_init is not None:
        # Patch the site object for this run
        batt_soc_save = batt.soc_min
        batt.soc_min = soc_init / batt.capacity_kwh if batt.capacity_kwh > 0 else 0.0
        cash, sc_ratio, details = simulate_period(
            site,
            pv,
            batt,
            econ,
            irradiance_kw_m2,
            cfg,
            24,
            0,
            smart,
            simple_hp_discharge,
        )
        batt.soc_min = batt_soc_save
    else:
        cash, sc_ratio, details = simulate_period(
            site,
            pv,
            batt,
            econ,
            irradiance_kw_m2,
            cfg,
            24,
            0,
            smart,
            simple_hp_discharge,
        )
    details["cash"] = cash
    details["self_consumption_ratio"] = sc_ratio
    return details


def _load_slp_csv(path: str, season_keyword: str = "winter") -> np.ndarray:
    """Flexible loader for the SLP CSV. Always returns hourly data for a full year (8760 values)"""

    df_raw = pd.read_csv(path, sep=";", header=None, dtype=str)
    if df_raw.empty:
        raise ValueError("CSV empty or wrong separator; expected ';' as delimiter.")

    header = [str(c).strip() for c in df_raw.iloc[0].tolist()]
    body = df_raw.iloc[1:].copy()
    body.columns = header

    # Columns to scan, prioritise those containing the keyword
    priority = [c for c in body.columns if season_keyword.lower() in c.lower()]
    fallback = [c for c in body.columns if c not in priority and c.lower() != "time"]
    search_cols = priority + fallback

    def _clean_numeric(col_name: str) -> pd.Series:
        col_data = body[col_name]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        ser = col_data.astype(str).str.replace(",", ".", regex=False)
        ser = pd.to_numeric(ser, errors="coerce").dropna().astype(float)
        return ser

    chosen = None
    for col in search_cols:
        cleaned = _clean_numeric(col)
        n = len(cleaned)
        # Accept 24, 96, 24*365, or 96*365
        if n in (24, 96, 24 * 365, 96 * 365):
            chosen = cleaned.reset_index(drop=True)
            break
    if chosen is None:
        raise ValueError(
            "No column with 24, 96, 8760, or 35040 numeric values found in CSV."
        )

    arr = chosen.to_numpy(dtype=float)
    n = len(arr)
    if n == 24:
        # 1 day, hourly: tile to 8760
        arr_hourly = np.tile(arr, 365)[:8760]
    elif n == 96:
        # 1 day, 15-min: aggregate to 24, then tile
        arr_hourly = np.tile(arr.reshape(24, 4).sum(axis=1), 365)[:8760]
    elif n == 24 * 365:
        arr_hourly = arr[:8760]
    elif n == 96 * 365:
        arr_hourly = arr.reshape(365, 24, 4).sum(axis=2).reshape(-1)[:8760]
    else:
        raise ValueError(f"Unexpected number of values in selected column: {n}")
    return arr_hourly


if __name__ == "__main__":
    _smoke_tests()

    csv_path = pathlib.Path("slpe.csv")
    # Create site_demo with dummy load for coordinates
    site_demo = CustomerSite(48.8, 2.3, np.zeros(8760))

    # Fetch irradiation using site_demo coordinates
    try:
        irr_year = get_hourly_irradiation(site_demo.latitude, site_demo.longitude)
        if len(irr_year) != 8760:
            print(
                f" Irradiation data length is {len(irr_year)}, expected 8760. Using dummy data."
            )
            irr_year = np.full(8760, 0.25)
        else:
            print("Loaded hourly irradiation from PVGIS API.")
    except Exception as e:
        print(f" Could not fetch irradiation from API: {e}")
        irr_year = np.full(8760, 0.25)

    # Now load the real load profile
    if csv_path.exists():
        print("Loading consumption profile from slpe.csv …")
        try:
            normalization_factor = 700
            load_year = _load_slp_csv(str(csv_path)) / normalization_factor
            print("Loaded consumption profile from dataset.")
        except Exception as exc:
            print(" Could not parse CSV:", exc)
            load_year = np.full(8760, 0.5)
    else:
        load_year = np.full(8760, 0.5)
    # Update site_demo with the real load
    site_demo.load_curve_kwh = load_year

    pv_demo = PVSystem(6, 0.9, 10800)
    batt_demo = BatterySystem(7, 4, capex_eur=5000)
    econ_demo = EconomicParams(
        0.27,
        0.17,
        list(range(6, 12)) + list(range(17, 21)),
        0.13,
        0.04,
        service_revenue_eur=120,
    )
    cfg_demo = SimulationConfig()

    # # Make a copy so you don't affect the simulation
    load_for_plot = site_demo.load_curve_kwh.copy()

    # Add random daily variation (e.g., ±10%)
    np.random.seed(42)  # for reproducibility
    variation = 1 + 0.7 * (np.random.rand(len(load_for_plot)) - 0.5) * 2
    load_for_plot = load_for_plot * variation

    # Temporarily assign for plotting
    site_demo_for_plot = CustomerSite(
        site_demo.latitude, site_demo.longitude, load_for_plot
    )

    # Last 3 months
    plot_single_series_bar(
        site_demo, irr_year, pv_demo, cfg_demo, variable="pv", period="day", length=7
    )

    # Last 7 days
    plot_single_series_bar(
        site_demo_for_plot,
        irr_year,
        pv_demo,
        cfg_demo,
        variable="load",
        period="day",
        length=7,
    )

    # Last 24 hours
    plot_single_series_bar(
        site_demo,
        irr_year,
        pv_demo,
        cfg_demo,
        variable="irradiation",
        period="hour",
        length=24,
    )

    # Plot last 7 days of grid import
    plot_grid_import_bar(
        site_demo,
        irr_year,
        pv_demo,
        batt_demo,
        econ_demo,
        cfg_demo,
        period="day",
        length=7,
    )

    # Plot last 7 days of self-consumption ratio (%)
    plot_self_consumption_bar(
        site_demo,
        irr_year,
        pv_demo,
        batt_demo,
        econ_demo,
        cfg_demo,
        period="day",
        length=7,
    )

    print("\nRunning 24h MPC simulation...")

    # Take a summer day (around day 180) for the simulation
    day_idx = 180
    start_hour = day_idx * 24
    site_24h = CustomerSite(
        site_demo.latitude,
        site_demo.longitude,
        site_demo.load_curve_kwh[start_hour : start_hour + 24],
    )

    results_24h = simulate_24h_mpc(
        site_24h,
        pv_demo,
        batt_demo,
        econ_demo,
        irr_year[start_hour : start_hour + 24],
        cfg_demo,
        smart=True,
        simple_hp_discharge=False,
    )

    # --- MPC Simulation Checks ---
    print("\nMPC Simulation Energy Conservation Check (per hour):")
    ok = True
    for t in range(24):
        lhs = results_24h["load"][t]
        rhs = (
            results_24h["pv"][t]
            + results_24h["grid_import"][t]
            + results_24h["batt_discharge"][t]
            - results_24h["batt_charge"][t]
            - results_24h["grid_export"][t]
        )
        if not np.isclose(lhs, rhs, atol=1e-6):
            print(
                f"  [!] Energy not conserved at hour {t}: load={lhs:.4f} != inflows={rhs:.4f}"
            )
            ok = False
    if ok:
        print("  All hours: Energy conserved (load == inflows/outflows)")

    total_grid_import = np.sum(results_24h["grid_import"])
    print(f"\nTotal grid import (MPC): {total_grid_import:.2f} kWh")

    print("\nFirst 5 hours SoC dynamics (should match MPC predictions):")
    for t in range(5):
        print(f"  Hour {t}: SoC={results_24h['soc'][t]:.3f} kWh")

    # --- MPC Behavior Analysis ---
    print(f"\nMPC Settings Check:")
    print(f"  Battery capacity: {batt_demo.capacity_kwh} kWh")
    print(f"  Battery efficiency: {batt_demo.roundtrip_eff:.1%}")
    print(f"  SoC limits: {batt_demo.soc_min:.1%} - {batt_demo.soc_max:.1%}")
    print(f"  HP price: €{econ_demo.buy_price_hp:.3f}/kWh")
    print(f"  HC price: €{econ_demo.buy_price_hc:.3f}/kWh")
    print(f"  HP hours: {econ_demo.hp_hours}")

    print(f"\nEvening Hours Analysis (should prefer battery over grid):")
    for t in range(18, 24):
        hour_of_day = t % 24
        is_hp = hour_of_day in econ_demo.hp_hours
        price_type = "HP" if is_hp else "HC"
        print(
            f"  Hour {t} ({hour_of_day}:00, {price_type}): "
            f"Grid={results_24h['grid_import'][t]:.3f} kWh, "
            f"Batt_discharge={results_24h['batt_discharge'][t]:.3f} kWh, "
            f"SoC={results_24h['soc'][t]:.3f} kWh"
        )

    soc_min_abs = batt_demo.soc_min * batt_demo.capacity_kwh
    soc_max_abs = batt_demo.soc_max * batt_demo.capacity_kwh
    print(f"\nSoC bounds check:")
    print(f"  Min SoC: {soc_min_abs:.3f} kWh, Max SoC: {soc_max_abs:.3f} kWh")
    print(
        f"  SoC range in simulation: {np.min(results_24h['soc']):.3f} - {np.max(results_24h['soc']):.3f} kWh"
    )

    # Create time vector for x-axis
    hours = np.arange(24)

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("24-hour MPC Simulation Results")

    # Plot 1: Load, PV, and Battery Power
    axs[0].plot(hours, results_24h["load"], "b-", label="Load")
    axs[0].plot(hours, results_24h["pv"], "y-", label="PV Generation")
    axs[0].plot(hours, results_24h["batt_discharge"], "g-", label="Battery Discharge")
    axs[0].plot(hours, -results_24h["batt_charge"], "r-", label="Battery Charge")
    axs[0].set_ylabel("Power (kW)")
    axs[0].set_title("Power Flows")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Grid Interaction
    axs[1].plot(hours, results_24h["grid_import"], "r-", label="Grid Import")
    axs[1].plot(hours, results_24h["grid_export"], "g-", label="Grid Export")
    axs[1].set_ylabel("Power (kW)")
    axs[1].set_title("Grid Interaction")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Battery State of Charge
    axs[2].plot(hours, results_24h["soc"], "b-", label="State of Charge")
    axs[2].set_ylabel("Energy (kWh)")
    axs[2].set_xlabel("Hour of Day")
    axs[2].set_title("Battery State of Charge")
    axs[2].grid(True)
    axs[2].legend()

    # Highlight HP hours
    for ax in axs:
        for hp_hour in econ_demo.hp_hours:
            ax.axvspan(hp_hour, hp_hour + 1, color="gray", alpha=0.2)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n24h Simulation Summary:")
    print(f"Total load: {results_24h['load'].sum():.1f} kWh")
    print(f"Total PV generation: {results_24h['pv'].sum():.1f} kWh")
    print(f"Self-consumption ratio: {results_24h['self_consumption_ratio']*100:.1f}%")
    print(f"Net cash flow: {results_24h['cash']:.2f} €")
