from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. FIXED SYSTEM CONFIGURATION
# ==========================================
NUM_PANELS = 42
WP_PER_PANEL = 470  # Total: 19.74 kWp
SYSTEM_LOSSES = 0.14
PANEL_TILT = 35.0
PANEL_AZIMUTH = 180.0

BATTERY_CAPACITY_KWH = 30.0  # Fixed capacity
BATTERY_MAX_POWER_KW = 10.0  # Fixed inverter power
ROUND_TRIP_EFFICIENCY = 0.92
MIN_SOC = 0.05
MAX_SOC = 1.00
INITIAL_SOC = 0.50

INSTALLATION_COST = 30000.0  # Fixed Capex (€)
BASE_ELEC_PRICE = 0.28  # Reference grid import price (€/kWh)
BASE_INJ_TARIFF = 0.05  # Reference grid injection price (€/kWh)

LATITUDE = 51.05
LONGITUDE = 3.72
CLIMATIC_YIELD_FACTOR = 0.65

FILE_NAME = "Kwartierwaarden verbruik - Opdeforten MDH.xlsx"
FILE_PATH = Path(__file__).resolve().parent / FILE_NAME


# ==========================================
# 2. PV PRODUCTION & DATASET ASSEMBLY
# ==========================================
def calculate_pv_generation(
    timestamps: pd.DatetimeIndex, peak_kwp: float
) -> pd.Series:
    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0

    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
    decl_rad = np.radians(declination)
    lat_rad = np.radians(LATITUDE)

    b = np.radians(360 / 365 * (day_of_year - 81))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    solar_time = hour + (4 * (LONGITUDE - 15 * 1) + eot) / 60.0
    omega = np.radians(15.0 * (solar_time - 12.0))

    sin_elev = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(omega)
    elevation = np.arcsin(np.clip(sin_elev, -1.0, 1.0))
    zenith_rad = np.pi / 2 - elevation

    tilt_rad = np.radians(PANEL_TILT)
    surface_azimuth_rad = np.radians(PANEL_AZIMUTH - 180.0)

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

    safe_sin_elev = np.maximum(0.0, sin_elev)
    safe_cos_theta = np.maximum(0.0, cos_theta)
    gti = np.where(
        (elevation > 0) & (cos_theta > 0),
        1050 * (safe_sin_elev**1.1) * safe_cos_theta,
        0.0,
    )

    pv_power_kw = (
        peak_kwp
        * (gti * CLIMATIC_YIELD_FACTOR / 1000.0)
        * (1.0 - SYSTEM_LOSSES)
    )
    return pd.Series(
        np.maximum(0.0, pv_power_kw * 0.25),
        index=timestamps,
        name="PV_Production_kWh",
    )


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
# 3. BATTERY DISPATCH (30 kWh / 10 kW)
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
    max_step = max_power_kw * 0.25

    u_min = capacity_kwh * min_soc
    u_max = capacity_kwh * max_soc
    stored = capacity_kwh * initial_soc

    direct_solar = 0.0
    bat_discharge = 0.0
    grid_imp = 0.0
    grid_exp = 0.0

    for i in range(len(cons)):
        c = cons[i]
        p = pv[i]
        direct = c if c < p else p
        direct_solar += direct
        surplus = p - direct
        deficit = c - direct

        if surplus > 0:
            space = (u_max - stored) / charge_eff
            chg = surplus if surplus < max_step else max_step
            if chg > space:
                chg = space
            stored += chg * charge_eff
            grid_exp += surplus - chg
        elif deficit > 0:
            avail = (stored - u_min) * discharge_eff
            max_d = max_step * discharge_eff
            dis = deficit if deficit < max_d else max_d
            if dis > avail:
                dis = avail
            bat_discharge += dis
            stored -= dis / discharge_eff
            grid_imp += deficit - dis

    return {
        "Direct_PV_kWh": direct_solar,
        "Battery_Discharge_kWh": bat_discharge,
        "Total_Self_Consumption_kWh": direct_solar + bat_discharge,
        "Grid_Import_kWh": grid_imp,
        "Grid_Export_kWh": grid_exp,
    }


# ==========================================
# 4. TARIFF SENSITIVITY PLOTTER
# ==========================================
def plot_tariff_sensitivity(
    self_cons_kwh: float,
    grid_exp_kwh: float,
    capex: float = 30000.0,
    elec_range: np.ndarray = np.arange(0.18, 0.42, 0.02),
    inj_range: np.ndarray = np.arange(0.00, 0.16, 0.02),
):
    payback_grid = np.zeros((len(elec_range), len(inj_range)))
    savings_grid = np.zeros((len(elec_range), len(inj_range)))

    for i, ep in enumerate(elec_range):
        for j, ip in enumerate(inj_range):
            annual_savings = self_cons_kwh * ep + grid_exp_kwh * ip
            savings_grid[i, j] = annual_savings
            payback_grid[i, j] = capex / annual_savings

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Heatmap
    im = axes[0].imshow(
        payback_grid,
        cmap="RdYlGn_r",
        aspect="auto",
        origin="lower",
        extent=[
            inj_range[0] - 0.01,
            inj_range[-1] + 0.01,
            elec_range[0] - 0.01,
            elec_range[-1] + 0.01,
        ],
    )
    axes[0].set_title(
        f"1. Payback Period (Years) [Fixed 30 kWh / 10 kW, €{capex:,.0f} Capex]",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Injection Tariff (€/kWh)", fontsize=11)
    axes[0].set_ylabel("Electricity Import Tariff (€/kWh)", fontsize=11)
    axes[0].set_xticks(inj_range)
    axes[0].set_yticks(elec_range)
    axes[0].set_xticklabels([f"€{x:.2f}" for x in inj_range])
    axes[0].set_yticklabels([f"€{x:.2f}" for x in elec_range])

    for i, ep in enumerate(elec_range):
        for j, ip in enumerate(inj_range):
            val = payback_grid[i, j]
            axes[0].text(
                ip,
                ep,
                f"{val:.1f}y",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if val < 8.5 or val > 12.5 else "black",
            )

    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label("Payback (Years)", fontsize=10)

    # Panel 2: Curves vs Import Price
    key_injections = [0.00, 0.04, 0.08, 0.12]
    curve_colors = ["#264653", "#2a9d8f", "#e76f51", "#e63946"]
    ep_continuous = np.linspace(0.18, 0.40, 60)

    for ip, col in zip(key_injections, curve_colors):
        sav = self_cons_kwh * ep_continuous + grid_exp_kwh * ip
        axes[1].plot(
            ep_continuous,
            capex / sav,
            lw=2.2,
            color=col,
            label=f"Injection: €{ip:.2f}/kWh",
        )

    axes[1].axvline(
        BASE_ELEC_PRICE,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Base Ref (€{BASE_ELEC_PRICE:.2f})",
    )
    axes[1].axhline(10.0, color="darkred", linestyle=":", alpha=0.6, label="10-Year Mark")

    axes[1].set_title(
        "2. Payback Period vs Electricity Price by Injection Tariff",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Electricity Import Tariff (€/kWh)", fontsize=11)
    axes[1].set_ylabel("Simple Payback Period (Years)", fontsize=11)
    axes[1].set_xticks(np.arange(0.18, 0.42, 0.04))
    axes[1].set_xticklabels([f"€{x:.2f}" for x in np.arange(0.18, 0.42, 0.04)])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.show()

    # Formatted terminal display
    inspect_inj = [0.00, 0.03, 0.05, 0.08, 0.10]
    inspect_elec = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]
    df_table = pd.DataFrame(
        index=[f"Import_€{ep:.2f}" for ep in inspect_elec],
        columns=[f"Inj_€{ip:.2f}" for ip in inspect_inj],
    )
    for ep in inspect_elec:
        for ip in inspect_inj:
            sav = self_cons_kwh * ep + grid_exp_kwh * ip
            df_table.loc[f"Import_€{ep:.2f}", f"Inj_€{ip:.2f}"] = (
                f"{capex / sav:.1f} yrs (€{sav:,.0f}/yr)"
            )
    return df_table


# ==========================================
# 5. MAIN EXECUTION PIPELINE
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

    tot_cons = df_fy["Consumption_kWh"].sum()
    tot_pv = df_fy["PV_Production_kWh"].sum()
    self_cons = results["Total_Self_Consumption_kWh"]
    grid_imp = results["Grid_Import_kWh"]
    grid_exp = results["Grid_Export_kWh"]

    base_savings = self_cons * BASE_ELEC_PRICE + grid_exp * BASE_INJ_TARIFF
    base_payback = INSTALLATION_COST / base_savings

    print("=" * 65)
    print("PHYSICAL DISPATCH SUMMARY (30 kWh / 10 kW SYSTEM)")
    print("=" * 65)
    print(f"Annual Club Consumption:    {tot_cons:>10,.1f} kWh")
    print(f"Annual Solar Generation:    {tot_pv:>10,.1f} kWh")
    print(
        f"Total Self-Consumption:     {self_cons:>10,.1f} kWh ({self_cons/tot_cons*100:.1f}% Autarky)"
    )
    print(f"Grid Import Required:       {grid_imp:>10,.1f} kWh")
    print(f"Grid Feed-in Surplus:       {grid_exp:>10,.1f} kWh")
    print("-" * 65)
    print(
        f"Reference Annual Savings:   €{base_savings:>10,.2f} / yr  (@ €{BASE_ELEC_PRICE:.2f} imp / €{BASE_INJ_TARIFF:.2f} inj)"
    )
    print(
        f"Reference Payback Period:    {base_payback:>10.1f} YEARS  (@ €{INSTALLATION_COST:,.0f} Capex)"
    )
    print("=" * 65)

    # Launch price sensitivity
    sensitivity_table = plot_tariff_sensitivity(
        self_cons_kwh=self_cons,
        grid_exp_kwh=grid_exp,
        capex=INSTALLATION_COST,
    )

    print("\nSummary Tariff Matrix:\n")
    print(sensitivity_table)
    return df_fy


if __name__ == "__main__":
    df_results = run_padel_energy_model(FILE_PATH)