from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. FIXED SYSTEM CONFIGURATION
# ==========================================
NUM_PANELS = 42
WP_PER_PANEL = 470  # Total: 19.74 kWp
PANEL_TILT = 35.0  # Optimum South tilt (degrees)
PANEL_AZIMUTH = 180.0  # 180° = South

BATTERY_CAPACITY_KWH = 30.0  # Storage capacity in kWh
BATTERY_MAX_POWER_KW = 10.0  # Inverter continuous AC rating in kW
ROUND_TRIP_EFFICIENCY = 0.92  # 92% round-trip efficiency
MIN_SOC = 0.05  # 5% reserve limit (95% DoD)
MAX_SOC = 1.00  # 100% max charge
INITIAL_SOC = 0.50

INSTALLATION_COST = 30000.0  # Turnkey Capex in €
BASE_ELEC_PRICE = 0.28  # All-in grid import tariff (€/kWh)
BASE_INJ_TARIFF = 0.05  # Grid injection tariff (€/kWh)
CAPACITY_TARIFF_EUR_PER_KW = 48.0  # Flanders reference capacity tariff (€/kW/yr)

LATITUDE = 51.05  # Ghent region
LONGITUDE = 3.72

FILE_NAME = "Kwartierwaarden verbruik - Opdeforten MDH.xlsx"
FILE_PATH = Path(__file__).resolve().parent / FILE_NAME


# ==========================================
# 2. PV PRODUCTION (DST-AWARE & PVGIS CALIBRATED)
# ==========================================
def calculate_pv_generation(
    timestamps: pd.DatetimeIndex,
    peak_kwp: float,
    tilt: float = 35.0,
    azimuth: float = 180.0,
    lat: float = 51.05,
    lon: float = 3.72,
) -> pd.Series:
    """Calculates 15-minute synthetic PV generation calibrated to Flemish reference yields (~1,000 kWh/kWp/year)."""
    # Dynamic Daylight Saving Time Offset for Belgium (CET = UTC+1, CEST = UTC+2)
    is_dst = (timestamps >= pd.Timestamp("2026-03-29 03:00:00")) & (
        timestamps < pd.Timestamp("2026-10-25 02:00:00")
    )
    dst_offset = np.where(is_dst, 2.0, 1.0)
    local_hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    utc_hour = local_hour - dst_offset

    day_of_year = timestamps.dayofyear.to_numpy()
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
    decl_rad = np.radians(declination)
    lat_rad = np.radians(lat)

    b = np.radians(360 / 365 * (day_of_year - 81))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    # True solar time from UTC
    solar_time = utc_hour + (lon / 15.0) + (eot / 60.0)
    omega = np.radians(15.0 * (solar_time - 12.0))

    sin_elev = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(omega)
    elevation = np.arcsin(np.clip(sin_elev, -1.0, 1.0))
    zenith_rad = np.pi / 2 - elevation

    tilt_rad = np.radians(tilt)
    surface_azimuth_rad = np.radians(azimuth - 180.0)

    cos_azimuth = np.clip(
        (
            np.sin(decl_rad) * np.cos(lat_rad)
            - np.cos(decl_rad) * np.sin(lat_rad) * np.cos(omega)
        )
        / np.cos(elevation),
        -1.0,
        1.0,
    )
    solar_azimuth = np.where(
        omega > 0, np.pi - np.arccos(cos_azimuth), np.pi + np.arccos(cos_azimuth)
    )
    cos_theta = np.cos(zenith_rad) * np.cos(tilt_rad) + np.sin(
        zenith_rad
    ) * np.sin(tilt_rad) * np.cos(solar_azimuth - surface_azimuth_rad)

    # PVGIS regional monthly insolation reference (kWh/kWp for South 35° in Flanders)
    monthly_targets = {
        1: 30.0,
        2: 48.0,
        3: 86.0,
        4: 118.0,
        5: 135.0,
        6: 136.0,
        7: 135.0,
        8: 120.0,
        9: 92.0,
        10: 58.0,
        11: 32.0,
        12: 22.0,
    }

    geom_curve = np.where(
        (elevation > 0) & (cos_theta > 0),
        (np.maximum(0.0, sin_elev) ** 0.85) * np.maximum(0.0, cos_theta),
        0.0,
    )

    df_temp = pd.DataFrame({"month": timestamps.month, "geom": geom_curve})
    monthly_sums = df_temp.groupby("month")["geom"].transform("sum")
    month_target_series = df_temp["month"].map(monthly_targets)

    # Scale 15-minute generation so monthly totals match actual Flemish solar irradiance
    pv_kwh_15m = (df_temp["geom"] / monthly_sums) * month_target_series * peak_kwp
    return pd.Series(pv_kwh_15m.to_numpy(), index=timestamps, name="PV_Production_kWh")


# ==========================================
# 3. FULL YEAR RECONSTRUCTION
# ==========================================
def build_full_year_dataset(
    df_measured: pd.DataFrame, peak_kwp: float
) -> pd.DataFrame:
    year = df_measured.index.min().year
    full_year_idx = pd.date_range(
        f"{year}-01-01 00:00:00", f"{year}-12-31 23:45:00", freq="15min"
    )

    df_fy = pd.DataFrame(index=full_year_idx)
    df_fy["PV_Production_kWh"] = calculate_pv_generation(
        full_year_idx, peak_kwp=peak_kwp
    )

    df_fy["Consumption_kWh"] = np.nan
    common_idx = df_measured.index.intersection(full_year_idx)
    df_fy.loc[common_idx, "Consumption_kWh"] = df_measured.loc[
        common_idx, "Consumption_kWh"
    ]

    # Handle spring DST jump: March 29 02:00-02:45 does not exist in local time
    march_jump = [
        ts
        for ts in df_fy[df_fy["Consumption_kWh"].isna()].index
        if ts.month == 3
    ]
    df_fy.loc[march_jump, "Consumption_kWh"] = 0.0

    # Impute missing autumn months (Sep-Dec) by analogous seasonal profile
    month_map = {9: 5, 10: 4, 11: 2, 12: 1}
    missing_idx = df_fy[df_fy["Consumption_kWh"].isna()].index

    if len(missing_idx) > 0:
        known = df_measured.copy()
        known["DayOfWeek"] = known.index.dayofweek
        known["Time"] = known.index.time
        known["Month"] = known.index.month

        lookup = known.groupby(["Month", "DayOfWeek", "Time"])[
            "Consumption_kWh"
        ].mean()
        dow_lookup = known.groupby(["DayOfWeek", "Time"])[
            "Consumption_kWh"
        ].mean()

        fill_vals = []
        for ts in missing_idx:
            target_m = month_map.get(ts.month, ts.month)
            val = lookup.get((target_m, ts.dayofweek, ts.time()), np.nan)
            if np.isnan(val):
                val = dow_lookup.get((ts.dayofweek, ts.time()), 0.0)
            fill_vals.append(val)
        df_fy.loc[missing_idx, "Consumption_kWh"] = fill_vals

    return df_fy


# ==========================================
# 4. BATTERY DISPATCH SIMULATION
# ==========================================
def simulate_battery_storage(
    cons: np.ndarray,
    pv: np.ndarray,
    capacity_kwh: float = 30.0,
    max_power_kw: float = 10.0,
    efficiency: float = 0.92,
    min_soc: float = 0.05,
    max_soc: float = 1.0,
    initial_soc: float = 0.5,
) -> dict:
    charge_eff = np.sqrt(efficiency)
    discharge_eff = np.sqrt(efficiency)
    max_step = max_power_kw * 0.25  # Rated AC energy per 15 min (2.5 kWh AC)

    u_min = capacity_kwh * min_soc
    u_max = capacity_kwh * max_soc
    stored = capacity_kwh * initial_soc

    n = len(cons)
    direct_solar = np.zeros(n)
    bat_discharge = np.zeros(n)
    bat_charge = np.zeros(n)
    grid_imp = np.zeros(n)
    grid_exp = np.zeros(n)
    soc_pct = np.zeros(n)

    for i in range(n):
        c = cons[i]
        p = pv[i]
        direct = min(c, p)
        direct_solar[i] = direct
        surplus = p - direct
        deficit = c - direct

        if surplus > 0:
            space = (u_max - stored) / charge_eff
            chg = min(surplus, max_step, space)
            bat_charge[i] = chg
            stored += chg * charge_eff
            grid_exp[i] = surplus - chg
        elif deficit > 0:
            avail = (stored - u_min) * discharge_eff
            dis = min(deficit, max_step, avail)  # Capped by true AC inverter limit
            bat_discharge[i] = dis
            stored -= dis / discharge_eff
            grid_imp[i] = deficit - dis

        soc_pct[i] = (stored / capacity_kwh) * 100.0

    return {
        "Direct_PV_kWh": direct_solar,
        "Battery_Charge_kWh": bat_charge,
        "Battery_Discharge_kWh": bat_discharge,
        "Battery_SoC_pct": soc_pct,
        "Total_Self_Consumption_kWh": direct_solar + bat_discharge,
        "Grid_Import_kWh": grid_imp,
        "Grid_Export_kWh": grid_exp,
    }


# ==========================================
# 5. SPLIT SENSITIVITY & FINANCIAL PLOTTER
# ==========================================
def plot_split_tariff_sensitivity(
    self_cons_kwh: float,
    grid_exp_kwh: float,
    cap_tariff_savings: float,
    capex: float = 30000.0,
    elec_range: np.ndarray = np.arange(0.18, 0.42, 0.02),
    inj_range: np.ndarray = np.arange(0.00, 0.16, 0.02),
):
    payback_no_cap = np.zeros((len(elec_range), len(inj_range)))
    payback_with_cap = np.zeros((len(elec_range), len(inj_range)))

    for i, ep in enumerate(elec_range):
        for j, ip in enumerate(inj_range):
            s1 = self_cons_kwh * ep + grid_exp_kwh * ip
            s2 = s1 + cap_tariff_savings
            payback_no_cap[i, j] = capex / s1
            payback_with_cap[i, j] = capex / s2

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Heatmap without Capacity Tariff
    im1 = axes[0, 0].imshow(
        payback_no_cap,
        cmap="RdYlGn_r",
        aspect="auto",
        origin="lower",
        extent=[
            inj_range[0] - 0.01,
            inj_range[-1] + 0.01,
            elec_range[0] - 0.01,
            elec_range[-1] + 0.01,
        ],
        vmin=5.5,
        vmax=13.5,
    )
    axes[0, 0].set_title(
        "1. Payback (Years): EXCLUDING Capacity Tariff",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Injection Tariff (€/kWh)", fontsize=10)
    axes[0, 0].set_ylabel("Electricity Import Tariff (€/kWh)", fontsize=10)
    axes[0, 0].set_xticks(inj_range)
    axes[0, 0].set_yticks(elec_range)
    axes[0, 0].set_xticklabels([f"€{x:.2f}" for x in inj_range])
    axes[0, 0].set_yticklabels([f"€{x:.2f}" for x in elec_range])
    for i, ep in enumerate(elec_range):
        for j, ip in enumerate(inj_range):
            val = payback_no_cap[i, j]
            axes[0, 0].text(
                ip,
                ep,
                f"{val:.1f}y",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if val < 7.0 or val > 11.0 else "black",
            )
    fig.colorbar(im1, ax=axes[0, 0], label="Years")

    # Panel 2: Heatmap with Capacity Tariff
    im2 = axes[0, 1].imshow(
        payback_with_cap,
        cmap="RdYlGn_r",
        aspect="auto",
        origin="lower",
        extent=[
            inj_range[0] - 0.01,
            inj_range[-1] + 0.01,
            elec_range[0] - 0.01,
            elec_range[-1] + 0.01,
        ],
        vmin=5.5,
        vmax=13.5,
    )
    axes[0, 1].set_title(
        "2. Payback (Years): INCLUDING Capacity Tariff",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Injection Tariff (€/kWh)", fontsize=10)
    axes[0, 1].set_ylabel("Electricity Import Tariff (€/kWh)", fontsize=10)
    axes[0, 1].set_xticks(inj_range)
    axes[0, 1].set_yticks(elec_range)
    axes[0, 1].set_xticklabels([f"€{x:.2f}" for x in inj_range])
    axes[0, 1].set_yticklabels([f"€{x:.2f}" for x in elec_range])
    for i, ep in enumerate(elec_range):
        for j, ip in enumerate(inj_range):
            val = payback_with_cap[i, j]
            axes[0, 1].text(
                ip,
                ep,
                f"{val:.1f}y",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if val < 7.0 or val > 11.0 else "black",
            )
    fig.colorbar(im2, ax=axes[0, 1], label="Years")

    # Panel 3: Curve Comparison vs Import Price
    ep_cont = np.linspace(0.18, 0.40, 60)
    for ip, col in zip([0.00, 0.05, 0.10], ["#264653", "#2a9d8f", "#e76f51"]):
        s1 = self_cons_kwh * ep_cont + grid_exp_kwh * ip
        s2 = s1 + cap_tariff_savings
        axes[1, 0].plot(
            ep_cont,
            capex / s1,
            lw=2,
            linestyle="--",
            color=col,
            label=f"Excl. Cap (Inj €{ip:.2f})",
        )
        axes[1, 0].plot(
            ep_cont,
            capex / s2,
            lw=2.2,
            linestyle="-",
            color=col,
            label=f"Incl. Cap (Inj €{ip:.2f})",
        )

    axes[1, 0].axvline(
        BASE_ELEC_PRICE,
        color="gray",
        linestyle=":",
        alpha=0.8,
        label=f"Ref (€{BASE_ELEC_PRICE:.2f})",
    )
    axes[1, 0].set_title(
        "3. Payback vs Import Price (Solid = Incl. Cap, Dashed = Excl.)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Electricity Import Tariff (€/kWh)", fontsize=10)
    axes[1, 0].set_ylabel("Simple Payback Period (Years)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(fontsize=8, loc="upper right")

    # Panel 4: 15-Year Cash Flow Projection
    years = np.arange(0, 16)
    base_s1 = self_cons_kwh * BASE_ELEC_PRICE + grid_exp_kwh * BASE_INJ_TARIFF
    base_s2 = base_s1 + cap_tariff_savings
    cf1 = -capex + base_s1 * years
    cf2 = -capex + base_s2 * years

    axes[1, 1].axhline(0, color="black", linestyle="-", lw=1, alpha=0.7)
    axes[1, 1].plot(
        years,
        cf1,
        marker="o",
        color="#e76f51",
        lw=2.2,
        label=f"Excl. Cap (€{base_s1:,.0f}/yr, {capex/base_s1:.1f}y)",
    )
    axes[1, 1].plot(
        years,
        cf2,
        marker="s",
        color="#2a9d8f",
        lw=2.2,
        label=f"Incl. Cap (€{base_s2:,.0f}/yr, {capex/base_s2:.1f}y)",
    )
    axes[1, 1].set_title(
        "4. Cumulative 15-Year Cash Flow (@ Base Reference Tariffs)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Years in Operation", fontsize=10)
    axes[1, 1].set_ylabel("Net Cumulative Cash Flow (€)", fontsize=10)
    axes[1, 1].set_xticks(years)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def run_padel_energy_model(filepath: Path):
    try:
        raw_df = pd.read_excel(filepath, header=1)
    except PermissionError:
        print(
            f"\n[ERROR] File '{filepath.name}' is open in Excel. Close it and re-run.\n"
        )
        return None

    timestamps = pd.to_datetime(raw_df.iloc[:, 1], errors="coerce")
    # Read Column F directly as provided in the Excel file
    consumption = pd.to_numeric(raw_df.iloc[:, 5], errors="coerce").fillna(0.0)

    df_sample = (
        pd.DataFrame({"Timestamp": timestamps, "Consumption_kWh": consumption})
        .dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    df_sample.set_index("Timestamp", inplace=True)
    df_sample = df_sample[~df_sample.index.duplicated(keep="first")]

    total_kwp = (NUM_PANELS * WP_PER_PANEL) / 1000.0

    # Build 365-day energy series
    df_fy = build_full_year_dataset(df_sample, peak_kwp=total_kwp)

    # Simulate physical battery dispatch
    results = simulate_battery_storage(
        cons=df_fy["Consumption_kWh"].to_numpy(),
        pv=df_fy["PV_Production_kWh"].to_numpy(),
        capacity_kwh=BATTERY_CAPACITY_KWH,
        max_power_kw=BATTERY_MAX_POWER_KW,
        efficiency=ROUND_TRIP_EFFICIENCY,
        min_soc=MIN_SOC,
        max_soc=MAX_SOC,
        initial_soc=INITIAL_SOC,
    )

    for k, v in results.items():
        df_fy[k] = v

    tot_cons = df_fy["Consumption_kWh"].sum()
    tot_pv = df_fy["PV_Production_kWh"].sum()
    self_cons = results["Total_Self_Consumption_kWh"].sum()
    grid_imp = results["Grid_Import_kWh"].sum()
    grid_exp = results["Grid_Export_kWh"].sum()
    direct_pv = results["Direct_PV_kWh"].sum()
    bat_discharge = results["Battery_Discharge_kWh"].sum()

    # Calculate Flemish Capacity Tariff peak reductions
    df_fy["Baseline_Power_kW"] = df_fy["Consumption_kWh"] * 4.0
    df_fy["Grid_Import_Power_kW"] = df_fy["Grid_Import_kWh"] * 4.0

    # Monthly peaks with Flemish 2.5 kW floor
    monthly_peak_baseline = np.maximum(
        2.5, df_fy.groupby(df_fy.index.month)["Baseline_Power_kW"].max()
    )
    monthly_peak_system = np.maximum(
        2.5, df_fy.groupby(df_fy.index.month)["Grid_Import_Power_kW"].max()
    )

    avg_peak_baseline = monthly_peak_baseline.mean()
    avg_peak_system = monthly_peak_system.mean()

    cap_cost_baseline = avg_peak_baseline * CAPACITY_TARIFF_EUR_PER_KW
    cap_cost_system = avg_peak_system * CAPACITY_TARIFF_EUR_PER_KW
    cap_tariff_savings = cap_cost_baseline - cap_cost_system

    # Case 1: Excluding Capacity Tariff
    savings_case1 = self_cons * BASE_ELEC_PRICE + grid_exp * BASE_INJ_TARIFF
    payback_case1 = INSTALLATION_COST / savings_case1

    # Case 2: Including Capacity Tariff
    savings_case2 = savings_case1 + cap_tariff_savings
    payback_case2 = INSTALLATION_COST / savings_case2

    print("=" * 68)
    print("PHYSICAL DISPATCH SUMMARY (42 PANELS / 30 kWh BATTERY)")
    print("=" * 68)
    print(f"Annual Club Consumption:      {tot_cons:>10,.1f} kWh")
    print(
        f"Annual Solar Generation:      {tot_pv:>10,.1f} kWh ({tot_pv/total_kwp:.1f} kWh/kWp)"
    )
    print(
        f"Direct Solar Used On-Site:    {direct_pv:>10,.1f} kWh ({direct_pv/tot_cons*100:.1f}% of Demand)"
    )
    print(
        f"Battery Energy Supplied:      {bat_discharge:>10,.1f} kWh ({bat_discharge/tot_cons*100:.1f}% of Demand)"
    )
    print(
        f"Total On-Site Self-Consumption: {self_cons:>10,.1f} kWh ({self_cons/tot_cons*100:.1f}% Autarky)"
    )
    print(f"Grid Import Required:         {grid_imp:>10,.1f} kWh")
    print(f"Grid Feed-in Surplus Sold:    {grid_exp:>10,.1f} kWh")
    print("-" * 68)
    print("FLEMISH CAPACITY TARIFF (CAPACITEITSTARIEF PEAK IMPACT)")
    print(
        f"Average Billed Peak Baseline: {avg_peak_baseline:>10.2f} kW  (Cost: €{cap_cost_baseline:,.2f}/yr)"
    )
    print(
        f"Average Billed Peak With Bat: {avg_peak_system:>10.2f} kW  (Cost: €{cap_cost_system:,.2f}/yr)"
    )
    print(f"Peak Demand Savings:          €{cap_tariff_savings:>10.2f} / year")
    print("=" * 68)
    print("SPLIT BUSINESS CASE COMPARISON (€30,000 CAPEX)")
    print("=" * 68)
    print("CASE 1: EXCLUDING CAPACITY TARIFF (Energy Commodity Only)")
    print(f"  * Annual Savings:           €{savings_case1:>10,.2f} / year")
    print(f"  * Simple Payback Period:     {payback_case1:>10.2f} YEARS")
    print("-" * 68)
    print("CASE 2: INCLUDING CAPACITY TARIFF (Comprehensive Case)")
    print(f"  * Annual Savings:           €{savings_case2:>10,.2f} / year")
    print(f"  * Simple Payback Period:     {payback_case2:>10.2f} YEARS")
    print("=" * 68)

    plot_split_tariff_sensitivity(
        self_cons_kwh=self_cons,
        grid_exp_kwh=grid_exp,
        cap_tariff_savings=cap_tariff_savings,
        capex=INSTALLATION_COST,
    )

    return df_fy


if __name__ == "__main__":
    df_results = run_padel_energy_model(FILE_PATH)