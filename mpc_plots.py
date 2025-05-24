import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pathlib
from simulator import (
    CustomerSite,
    PVSystem,
    BatterySystem,
    EconomicParams,
    SimulationConfig,
)
from simulator import simulate_24h_mpc, _load_slp_csv
from get_irradiation import get_hourly_irradiation
from mpc_dispatch import mpc_battery_action

# Colors
RED = "#D71700"
GRAY = "#929292"
GREEN = "#1FB100"
BLACK = "#000000"
LIGHT_GRAY = "#F5F5F5"
DARK_GRAY = "#333333"


def load_simulation_data():
    """Load the same data as the main simulation"""
    csv_path = pathlib.Path("slpe.csv")
    site_demo = CustomerSite(48.8, 2.3, np.zeros(8760))

    try:
        irr_year = get_hourly_irradiation(site_demo.latitude, site_demo.longitude)
        if len(irr_year) != 8760:
            irr_year = np.full(8760, 0.25)
    except:
        irr_year = np.full(8760, 0.25)

    if csv_path.exists():
        try:
            load_year = _load_slp_csv(str(csv_path)) / 700
        except:
            load_year = np.full(8760, 0.5)
    else:
        load_year = np.full(8760, 0.5)

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

    return site_demo, pv_demo, batt_demo, econ_demo, cfg_demo, irr_year


def get_mpc_forecast(
    hour_t, site, pv, batt, econ, irr_year, cfg, soc_current, horizon=6
):
    """Get MPC forecast for next 'horizon' hours starting from hour_t"""
    day_idx = 180
    start_hour = day_idx * 24
    abs_hour = start_hour + hour_t

    # Get forecasts - use the remaining hours from current position
    if abs_hour + horizon <= len(site.load_curve_kwh):
        load_fc = site.load_curve_kwh[
            abs_hour + 1 : abs_hour + 1 + horizon
        ]  # +1 to get NEXT hours
        irr_fc = irr_year[abs_hour + 1 : abs_hour + 1 + horizon]
    else:
        # Handle wrap-around (shouldn't happen in 24h simulation)
        remaining = len(site.load_curve_kwh) - (abs_hour + 1)
        load_fc = np.concatenate(
            [
                site.load_curve_kwh[abs_hour + 1 :],
                site.load_curve_kwh[: horizon - remaining],
            ]
        )
        irr_fc = np.concatenate(
            [irr_year[abs_hour + 1 :], irr_year[: horizon - remaining]]
        )

    # Get PV forecast
    from simulator import pv_production_hourly

    pv_fc = pv_production_hourly(irr_fc, pv, 0, cfg)

    # Create a temporary site for the forecast period
    forecast_site = CustomerSite(site.latitude, site.longitude, load_fc)

    # Use the actual simulation logic to get the forecast
    from simulator import simulate_period

    # Temporarily set initial SoC
    original_soc_range = (batt.soc_min + batt.soc_max) / 2
    temp_soc_min = soc_current / batt.capacity_kwh
    temp_soc_max = batt.soc_max

    # Create temporary battery with current SoC as starting point
    temp_batt = BatterySystem(
        capacity_kwh=batt.capacity_kwh,
        max_power_kw=batt.max_power_kw,
        soc_min=temp_soc_min,
        soc_max=temp_soc_max,
        roundtrip_eff=batt.roundtrip_eff,
        capex_eur=batt.capex_eur,
    )

    # Run simulation for forecast period
    cash, sc_ratio, forecast_details = simulate_period(
        forecast_site,
        pv,
        temp_batt,
        econ,
        irr_fc,
        cfg,
        period_hours=len(load_fc),
        year_idx=0,
        smart=True,
        simple_hp_discharge=False,
    )

    return {
        "load": load_fc,
        "pv": pv_fc,
        "charge": forecast_details["batt_charge"],
        "discharge": forecast_details["batt_discharge"],
        "grid_import": forecast_details["grid_import"],
        "grid_export": forecast_details["grid_export"],
        "soc": forecast_details["soc"],
    }


def create_animated_plot():
    """Create the animated MPC visualization"""
    # Load data
    site, pv, batt, econ, cfg, irr_year = load_simulation_data()

    # Run full 24h simulation
    day_idx = 180
    start_hour = day_idx * 24
    site_24h = CustomerSite(
        site.latitude, site.longitude, site.load_curve_kwh[start_hour : start_hour + 24]
    )

    results_24h = simulate_24h_mpc(
        site_24h,
        pv,
        batt,
        econ,
        irr_year[start_hour : start_hour + 24],
        cfg,
        smart=True,
        simple_hp_discharge=False,
    )

    # Setup the figure with mobile-friendly aspect ratio and high resolution
    fig = plt.figure(figsize=(10, 14), facecolor="white", dpi=150)
    fig.patch.set_facecolor("white")

    # Create subplots - removed battery power subplot
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

    ax1 = fig.add_subplot(gs[0])  # Load & PV
    ax2 = fig.add_subplot(gs[1])  # Battery SoC
    ax3 = fig.add_subplot(gs[2])  # Grid Import/Export

    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3, color=GRAY, axis="y")  # Only horizontal grid lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)  # Remove y-axis line
        ax.spines["bottom"].set_visible(False)  # Remove x-axis line
        ax.tick_params(
            colors=BLACK, labelsize=22, bottom=True, left=True, top=False, right=False
        )
        ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks

    def animate(frame):
        # Clear all axes
        for ax in [ax1, ax2, ax3]:
            ax.clear()

        # Animation parameters - predictions extend from hour 14 forward
        start_hour = 14  # Fixed starting point
        max_prediction_hours = 10  # Maximum hours to predict (14h + 10h = 24h)
        steps_per_hour = 16

        # Calculate current prediction extent based on frame
        prediction_progress = frame / (max_prediction_hours * steps_per_hour)
        current_prediction_hours = prediction_progress * max_prediction_hours
        prediction_end_time = min(start_hour + current_prediction_hours, 24)

        # Historical data (static) - always show from 0 to start_hour
        hours_hist = np.arange(start_hour + 1)  # 0 to 14 (inclusive)
        load_hist = results_24h["load"][: start_hour + 1]
        pv_hist = results_24h["pv"][: start_hour + 1]
        soc_hist = results_24h["soc"][: start_hour + 1]
        grid_import_hist = results_24h["grid_import"][: start_hour + 1]
        grid_export_hist = results_24h["grid_export"][: start_hour + 1]

        # Prediction data (animated) - extend from start_hour forward
        if prediction_end_time > start_hour:
            # Create smooth prediction timeline
            n_prediction_points = int(
                (prediction_end_time - start_hour) * steps_per_hour
            )
            if n_prediction_points > 0:
                forecast_hours = np.linspace(
                    start_hour, prediction_end_time, n_prediction_points + 1
                )[
                    1:
                ]  # Exclude start_hour

                # Ensure we don't go beyond available data
                valid_indices = forecast_hours < 24
                forecast_hours = forecast_hours[valid_indices]

                if len(forecast_hours) > 0:
                    # Interpolate prediction data smoothly
                    load_forecast = np.interp(
                        forecast_hours, np.arange(24), results_24h["load"]
                    )
                    pv_forecast = np.interp(
                        forecast_hours, np.arange(24), results_24h["pv"]
                    )
                    soc_forecast = np.interp(
                        forecast_hours, np.arange(24), results_24h["soc"]
                    )
                    grid_import_forecast = np.interp(
                        forecast_hours, np.arange(24), results_24h["grid_import"]
                    )
                    grid_export_forecast = np.interp(
                        forecast_hours, np.arange(24), results_24h["grid_export"]
                    )
                else:
                    forecast_hours = np.array([])
                    load_forecast = np.array([])
                    pv_forecast = np.array([])
                    soc_forecast = np.array([])
                    grid_import_forecast = np.array([])
                    grid_export_forecast = np.array([])
            else:
                forecast_hours = np.array([])
                load_forecast = np.array([])
                pv_forecast = np.array([])
                soc_forecast = np.array([])
                grid_import_forecast = np.array([])
                grid_export_forecast = np.array([])
        else:
            forecast_hours = np.array([])
            load_forecast = np.array([])
            pv_forecast = np.array([])
            soc_forecast = np.array([])
            grid_import_forecast = np.array([])
            grid_export_forecast = np.array([])

        # Add HP hour shading to all subplots
        def add_hp_shading(ax):
            for hp_hour in econ.hp_hours:
                ax.axvspan(
                    hp_hour,
                    hp_hour + 1,
                    facecolor=GRAY,
                    alpha=0.15,
                    zorder=0,
                    edgecolor="none",
                    linewidth=0,
                )

        # Plot 1: Load & PV
        ax1.set_xlim(0, 24)
        ax1.set_ylim(0, 3)
        add_hp_shading(ax1)

        # Static historical data
        ax1.plot(hours_hist, load_hist, color=BLACK, linewidth=5, label="Consumption")
        ax1.plot(hours_hist, pv_hist, color=GREEN, linewidth=5, label="PV Production")

        # Animated predictions
        if len(load_forecast) > 0:
            ax1.plot(
                forecast_hours,
                load_forecast,
                color=BLACK,
                linewidth=3,
                alpha=0.7,
                linestyle="--",
            )
            ax1.plot(
                forecast_hours,
                pv_forecast,
                color=GREEN,
                linewidth=3,
                alpha=0.7,
                linestyle="--",
            )

        ax1.set_ylabel("kW", fontsize=24, color=BLACK)
        ax1.set_title("Energy Flow", fontsize=26, color=BLACK, pad=20)
        ax1.grid(True, alpha=0.3, color=GRAY, axis="y")
        ax1.set_xticklabels([])  # Remove x-axis labels
        ax1.legend(
            loc="upper right", fontsize=20, frameon=False, bbox_to_anchor=(0.98, 0.98)
        )

        # Plot 2: Battery SoC
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0, 7)
        add_hp_shading(ax2)

        # Static historical data
        ax2.plot(hours_hist, soc_hist, color=GRAY, linewidth=5, label="SoC")

        # Animated predictions with seamless connection
        if len(soc_forecast) > 0:
            # Connect current SoC to forecast seamlessly
            soc_connected = np.concatenate([[soc_hist[-1]], soc_forecast])
            hours_connected = np.concatenate([[start_hour], forecast_hours])
            ax2.plot(
                hours_connected,
                soc_connected,
                color=GRAY,
                linewidth=3,
                alpha=0.7,
                linestyle="--",
            )

        ax2.set_ylabel("kWh", fontsize=24, color=BLACK)
        ax2.set_title("Battery State of Charge", fontsize=26, color=BLACK, pad=20)
        ax2.grid(True, alpha=0.3, color=GRAY, axis="y")
        ax2.set_xticklabels([])  # Remove x-axis labels
        # No legend needed - only one curve

        # Plot 3: Grid Import/Export
        ax3.set_xlim(0, 24)
        ax3.set_ylim(-1, 2)
        add_hp_shading(ax3)

        # Static historical data
        ax3.plot(hours_hist, grid_import_hist, color=RED, linewidth=5, label="Import")
        ax3.plot(
            hours_hist, -grid_export_hist, color=GREEN, linewidth=5, label="Export"
        )

        # Animated predictions
        if len(grid_import_forecast) > 0:
            ax3.plot(
                forecast_hours,
                grid_import_forecast,
                color=RED,
                linewidth=3,
                alpha=0.7,
                linestyle="--",
            )
            ax3.plot(
                forecast_hours,
                -grid_export_forecast,
                color=GREEN,
                linewidth=3,
                alpha=0.7,
                linestyle="--",
            )

        ax3.axhline(y=0, color=BLACK, linewidth=1, alpha=0.5)
        ax3.set_ylabel("kW", fontsize=24, color=BLACK)
        ax3.set_title("Grid Interaction", fontsize=26, color=BLACK, pad=20)
        ax3.set_xlabel("Hour of Day", fontsize=24, color=BLACK)
        ax3.grid(True, alpha=0.3, color=GRAY, axis="y")
        ax3.legend(
            loc="upper right", fontsize=20, frameon=False, bbox_to_anchor=(0.98, 0.98)
        )

        # Style all axes
        for ax in [ax1, ax2, ax3]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)  # Remove y-axis line
            ax.spines["bottom"].set_visible(False)  # Remove x-axis line
            ax.tick_params(
                colors=BLACK,
                labelsize=22,
                bottom=True,
                left=True,
                top=False,
                right=False,
            )
            ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks

        # Add overall legend at the bottom with better visibility (just for line styles and HP hours)
        legend_elements = [
            plt.Line2D([0], [0], color=BLACK, linewidth=5, label="Measurement"),
            plt.Line2D(
                [0],
                [0],
                color=BLACK,
                linewidth=3,
                linestyle="--",
                alpha=0.7,
                label="Prediction",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=GRAY,
                alpha=0.15,
                edgecolor="none",
                label="HP Hours",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=22,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )

        # Apply consistent spacing with more bottom margin for legend and x-axis label
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.18, hspace=0.5)

    # Create animation - predictions extend from hour 14 to 24 (10 hours total)
    max_prediction_hours = 10
    total_frames = max_prediction_hours * 16  # 16 sub-steps per hour
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=150, repeat=True, blit=False
    )

    # Save as GIF with maximum resolution and faster speed
    print("Creating ultra-smooth animated GIF... This may take a moment.")
    anim.save("mpc_animation.gif", writer="pillow", fps=6.67, dpi=150)
    print("Animation saved as 'mpc_animation.gif'")

    plt.close()


if __name__ == "__main__":
    create_animated_plot()
