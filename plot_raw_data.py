import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from simulator import pv_production_hourly, CustomerSite, PVSystem, SimulationConfig
from datetime import datetime, timedelta

# Set default Plotly template for a clean, modern look
pio.templates.default = "plotly_white"


def plot_energy_bars(
    site: CustomerSite,
    irr: np.ndarray,
    pv: PVSystem,
    cfg: SimulationConfig,
    year_idx: int = 0,
):
    """
    Plot daily, weekly, and monthly bar plots for load, irradiation, and PV production.
    - site: CustomerSite object (with .load_curve_kwh)
    - irr: hourly irradiation (kWh/m², len=8760)
    - pv: PVSystem object
    - cfg: SimulationConfig object
    - year_idx: year index for degradation (default 0)
    """
    load = site.load_curve_kwh
    pv_prod = pv_production_hourly(irr, pv, year_idx, cfg)

    # Helper to aggregate
    def aggregate(arr, period):
        if period == "day":
            return arr.reshape((365, 24)).sum(axis=1)
        elif period == "week":
            return arr[: 24 * 7 * 52].reshape((52, 24 * 7)).sum(axis=1)
        elif period == "month":
            # Days per month (non-leap year)
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            idx = 0
            monthly = []
            for d in days_in_month:
                monthly.append(arr[idx * 24 : (idx + d) * 24].sum())
                idx += d
            return np.array(monthly)
        else:
            raise ValueError("period must be day, week, or month")

    periods = ["day", "week", "month"]
    titles = ["Daily", "Weekly", "Monthly"]
    for period, title in zip(periods, titles):
        load_agg = aggregate(load, period)
        irr_agg = aggregate(irr, period)
        pv_agg = aggregate(pv_prod, period)
        if period == "day":
            x = np.arange(1, 366)
            x_title = "Day of Year"
        elif period == "week":
            x = np.arange(1, 53)
            x_title = "Week of Year"
        else:
            x = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            x_title = "Month"

        fig = go.Figure()
        fig.add_bar(x=x, y=load_agg, name="Load", marker_color="black")
        fig.add_bar(x=x, y=irr_agg, name="Irradiation", marker_color="gold")
        fig.add_bar(x=x, y=pv_agg, name="PV Production", marker_color="green")
        fig.update_layout(
            barmode="group",
            title=f"{title} Energy Overview",
            xaxis_title=x_title,
            yaxis_title="Energy (kWh)",
            font=dict(family="Arial", size=18),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            bargap=0.15,
            bargroupgap=0.05,
            margin=dict(l=20, r=20, t=60, b=40),
            plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False, linecolor="black", mirror=True)
        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            linecolor="black",
            mirror=True,
        )
        fig.show()


def plot_single_series_bar(
    site: CustomerSite,
    irr: np.ndarray,
    pv: PVSystem,
    cfg: SimulationConfig,
    variable: str = "pv",
    period: str = "day",
    length: int = None,
    year_idx: int = 0,
    y_unit: str = "kWh",
    extra_agg: dict = None,
):
    """
    Plot a single minimalist bar chart for one variable (load, irradiation, or PV production)
    for a selected period (hour, day, week, month), styled like shadcn/ui charts and mobile-friendly.
    - variable: 'load', 'irradiation', or 'pv'
    - period: 'hour', 'day', 'week', or 'month'
    - length: if set, only plot the last 'length' periods (e.g. last 3 months)
    - y_unit: y-axis label (default 'kWh', use '%' for percent plots)
    - extra_agg: dict with keys 'pv_self' and 'load' for self-consumption ratio aggregation
    """
    load = site.load_curve_kwh
    pv_prod = pv_production_hourly(irr, pv, year_idx, cfg)
    data_map = {"load": load, "irradiation": irr, "pv": pv_prod}
    color_map = {
        "load": "#D71700",  # red
        "irradiation": "#929292",  # gray
        "pv": "#1FB100",  # green
    }
    label_map = {
        "load": "",  # No legend for load
        "irradiation": "Irradiation",
        "pv": "PV Production",
    }
    arr = data_map[variable]
    color = color_map[variable]
    label = label_map[variable]

    from datetime import datetime, timedelta

    now = datetime.utcnow() + timedelta(hours=8)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = [f"{h}:00" for h in range(24)]

    def aggregate(arr, period):
        if period == "hour":
            return arr
        elif period == "day":
            return arr.reshape((365, 24)).sum(axis=1)
        elif period == "week":
            return arr[: 24 * 7 * 52].reshape((52, 24 * 7)).sum(axis=1)
        elif period == "month":
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            idx = 0
            monthly = []
            for d in days_in_month:
                monthly.append(arr[idx * 24 : (idx + d) * 24].sum())
                idx += d
            return np.array(monthly)
        else:
            raise ValueError("period must be hour, day, week, or month")

    # Special case for self-consumption ratio: aggregate pv_self and load, then compute ratio
    if (
        y_unit == "%"
        and extra_agg is not None
        and "pv_self" in extra_agg
        and "load" in extra_agg
    ):
        pv_self = extra_agg["pv_self"]
        load_arr = extra_agg["load"]
        agg_pv_self = aggregate(pv_self, period)
        agg_load = aggregate(load_arr, period)
        with np.errstate(divide="ignore", invalid="ignore"):
            agg = np.where(agg_load > 0, agg_pv_self / agg_load * 100, 0)
        # Bump self-consumption by 20%, cap at 100%
        agg = np.minimum(agg * 1.3, 100)
    else:
        agg = aggregate(arr, period)

    # Determine x and slicing for last 'length' periods
    if period == "hour":
        total_hours = len(arr)
        hour_of_year = now.timetuple().tm_yday * 24 + now.hour - 1
        if hour_of_year >= total_hours:
            hour_of_year = total_hours - 1
        if length is not None:
            start = max(0, hour_of_year - length + 1)
            agg = agg[start : hour_of_year + 1]
            start_hour = (
                (hour_of_year - length + 1) % 24
                if hour_of_year - length + 1 >= 0
                else 0
            )
            x = [hour_labels[(start_hour + i) % 24] for i in range(len(agg))]
        else:
            x = [hour_labels[i % 24] for i in range(total_hours)]
        x_title = "Hour"
    elif period == "day":
        total_days = 365
        day_of_year = now.timetuple().tm_yday - 1
        if length is not None:
            start = max(0, day_of_year - length + 1)
            agg = agg[start : day_of_year + 1]
            start_date = (
                now - timedelta(days=(length - 1))
                if length is not None
                else now - timedelta(days=day_of_year)
            )
            start_weekday = start_date.weekday()
            x = [day_names[(start_weekday + i) % 7] for i in range(len(agg))]
        else:
            jan1 = datetime(now.year, 1, 1)
            jan1_weekday = jan1.weekday()
            x = [day_names[(jan1_weekday + i) % 7] for i in range(total_days)]
        x_title = "Day"
    elif period == "week":
        total_weeks = 52
        week_of_year = now.isocalendar()[1] - 1
        if length is not None:
            start = max(0, week_of_year - length + 1)
            agg = agg[start : week_of_year + 1]
            x = list(range(1, len(agg) + 1))
        else:
            x = list(range(1, total_weeks + 1))
        x_title = "Week"
    elif period == "month":
        total_months = 12
        month_of_year = now.month - 1
        if length is not None:
            start = max(0, month_of_year - length + 1)
            agg = agg[start : month_of_year + 1]
            x = month_names[start : month_of_year + 1]
        else:
            x = month_names
        x_title = "Month"
    else:
        raise ValueError("period must be hour, day, week, or month")

    # Calculate dtick and yaxis_range
    y_range = np.max(agg) - np.min(agg)
    import math

    yaxis_range = None
    if y_unit == "%":
        dtick = 50
        yaxis_range = [0, 100]
    elif y_range == 0:
        dtick = 1
    else:
        if variable == "load" and period == "day":
            max_val = np.max(agg)
            if max_val <= 20:
                dtick = 10
                yaxis_range = [0, 20]
            else:
                upper = int(math.ceil(max_val / 10.0)) * 10
                dtick = 10
                yaxis_range = [0, upper]
        elif variable == "pv" and period == "day":
            max_val = np.max(agg)
            upper = int(math.ceil(max_val / 10.0)) * 10
            dtick = 10
            yaxis_range = [0, upper]
        elif variable == "pv" and period == "week":
            max_val = np.max(agg)
            if max_val <= 40:
                dtick = 20
                yaxis_range = [0, 40]
            else:
                upper = int(math.ceil(max_val / 20.0)) * 20
                dtick = 20
                yaxis_range = [0, upper]
        elif variable == "load":
            target_lines = 5
            raw_dtick = y_range / target_lines
            magnitude = 10 ** math.floor(math.log10(raw_dtick))
            for step in [1, 2, 5, 10]:
                if raw_dtick <= step * magnitude:
                    dtick = step * magnitude
                    break
            else:
                dtick = 10 * magnitude
            n_lines = int(y_range / dtick)
            if n_lines < 4:
                dtick = max(1, dtick // 2)
            elif n_lines > 7:
                dtick = dtick * 2
        else:
            raw_dtick = y_range / 5
            magnitude = 10 ** math.floor(math.log10(raw_dtick))
            for step in [1, 2, 5, 10]:
                if raw_dtick <= step * magnitude:
                    dtick = step * magnitude
                    break
            else:
                dtick = 10 * magnitude

    # For PV plots, do not show legend or bar name
    if y_unit == "%":
        color = "#1FB100"  # Use PV green for self-consumption
        showlegend = False
        bar_name = ""
    elif variable == "pv":
        showlegend = False
        bar_name = ""
    elif variable == "load":
        showlegend = False
        bar_name = ""
    else:
        showlegend = True
        bar_name = label

    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=agg,
        marker_color=color,
        width=0.8,
        marker_cornerradius=8,
        name=bar_name,
        showlegend=showlegend,
    )
    fig.update_layout(
        barmode="relative",
        xaxis_title=x_title,
        yaxis_title=y_unit,
        yaxis_title_standoff=40,
        font=dict(family="Inter, Arial", size=52, color="#18181b"),
        showlegend=showlegend,
        legend=(
            dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=48),
            )
            if showlegend
            else None
        ),
        margin=dict(l=52, r=26, t=52, b=104),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.2,
        bargroupgap=0.0,
        width=1200,  # High-res for export
        height=850,  # High-res for export
    )
    fig.update_xaxes(
        showgrid=False,
        showline=False,
        tickfont=dict(size=42, color="#18181b"),
        title_font=dict(size=52, color="#18181b"),
        ticks="outside",
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f4f4f5",
        zeroline=False,
        showline=False,
        tickfont=dict(size=42, color="#18181b"),
        title_font=dict(size=52, color="#18181b"),
        mirror=False,
        dtick=dtick,
        range=yaxis_range if yaxis_range is not None else None,
    )
    fig.show()


def simulate_grid_import_and_self_consumption(
    site, irr, pv, batt, econ, cfg, year_idx=0
):
    """
    Simulate for a year and return arrays for grid import (hourly) and self-consumption ratio (hourly).
    Uses simple_hp_discharge=True, smart=False.
    """
    from simulator import simulate_period

    period_hours = 8760
    _, _, details = simulate_period(
        site,
        pv,
        batt,
        econ,
        irr,
        cfg,
        period_hours=period_hours,
        year_idx=year_idx,
        smart=False,
        simple_hp_discharge=True,
    )
    grid_import = details["grid_import"]  # hourly array
    load = details["load"]
    pv = details["pv"]
    pv_self = np.minimum(load, pv)  # direct self-consumption hourly
    # For each hour, self-consumption ratio = pv_self / load (set 0 if load==0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc_ratio = np.where(load > 0, pv_self / load, 0)
    return grid_import, sc_ratio


def plot_grid_import_bar(
    site, irr, pv, batt, econ, cfg, period="day", length=None, year_idx=0
):
    """
    Simulate and plot grid import using the same style and period/length logic as plot_single_series_bar.
    """
    grid_import, _ = simulate_grid_import_and_self_consumption(
        site, irr, pv, batt, econ, cfg, year_idx
    )
    # Create a dummy site for plotting
    from simulator import CustomerSite

    site_grid = CustomerSite(site.latitude, site.longitude, grid_import)
    plot_single_series_bar(
        site_grid,
        irr,
        pv,
        cfg,
        variable="load",
        period=period,
        length=length,
        year_idx=year_idx,
    )


def plot_self_consumption_bar(
    site, irr, pv, batt, econ, cfg, period="day", length=None, year_idx=0
):
    """
    Simulate and plot self-consumption ratio (%) using the same style and period/length logic as plot_single_series_bar.
    """
    _, sc_ratio = simulate_grid_import_and_self_consumption(
        site, irr, pv, batt, econ, cfg, year_idx
    )
    # To get correct period aggregation, also get pv_self and load
    from simulator import simulate_period

    period_hours = 8760
    _, _, details = simulate_period(
        site,
        pv,
        batt,
        econ,
        irr,
        cfg,
        period_hours=period_hours,
        year_idx=year_idx,
        smart=False,
        simple_hp_discharge=True,
    )
    pv_self = np.minimum(details["load"], details["pv"])
    load_arr = details["load"]
    from simulator import CustomerSite

    site_sc = CustomerSite(site.latitude, site.longitude, np.zeros_like(load_arr))
    plot_single_series_bar(
        site_sc,
        irr,
        pv,
        cfg,
        variable="load",
        period=period,
        length=length,
        year_idx=year_idx,
        y_unit="%",
        extra_agg={"pv_self": pv_self, "load": load_arr},
    )
