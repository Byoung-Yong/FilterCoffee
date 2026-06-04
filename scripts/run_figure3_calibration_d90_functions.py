"""Generate Figure 3 calibration outputs for the JFE manuscript.

This analysis uses only the matched in-house pour-over experiment and the
measured PSD file. It does not use public recipe records and does not refit
coefficients. The simulator is run with the current D90-conditioned closure
functions from ``scripts/calibrated_v60_model.py``.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrated_v60_model import (  # noqa: E402
    CALIBRATED_COEFFICIENTS,
    D90_CLOSURE_FITS,
    d90_closure_coefficients_for_scenario,
    d90_closure_config_for_scenario,
    load_calibrated_psd_scenarios,
)
from run_measured_psd_analysis import mass_percentile_diameter_um  # noqa: E402
from v60_physics.parameters import ModelConfig, Scenario, load_config  # noqa: E402
from v60_physics.solver import run_simulation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "figure3_calibration_d90_functions"
EXPERIMENT_FILE = ROOT / "data" / "pour-over data.csv"
CALIBRATION_CSV = OUTPUT_DIR / "figure3_calibration_outputs.csv"
D90_CURVE_CSV = OUTPUT_DIR / "figure3_d90_function_curves.csv"
SUMMARY_MD = OUTPUT_DIR / "figure3_calibration_summary.md"
FIG_PNG = OUTPUT_DIR / "figure3_calibration_d90_functions.png"
FIG_PDF = OUTPUT_DIR / "figure3_calibration_d90_functions.pdf"
CAPTION_MD = OUTPUT_DIR / "figure3_caption.md"

PSD_ORDER = ("coarse", "medium", "fine")
CLASS_LABELS = {"coarse": "coarse", "medium": "medium", "fine": "fine"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    base = reference_config()
    scenarios = {scenario.name: scenario for scenario in load_calibrated_psd_scenarios()}
    experiment = load_experiment()

    calibration_rows = []
    for name in PSD_ORDER:
        scenario = scenarios[name]
        calibration_rows.append(run_calibration_case(base, scenario, experiment[name]))

    d90_curve_rows = d90_function_curve_rows()
    anchor_check_rows = anchor_check(calibration_rows)

    write_csv(CALIBRATION_CSV, calibration_rows)
    write_csv(D90_CURVE_CSV, d90_curve_rows)
    SUMMARY_MD.write_text(summary_markdown(calibration_rows, anchor_check_rows), encoding="utf-8")
    CAPTION_MD.write_text(caption_text(), encoding="utf-8")
    make_figure(calibration_rows, d90_curve_rows)

    max_water = max(float(row["max_water_balance_residual_g"]) for row in calibration_rows)
    max_solids = max(
        float(row["max_dissolved_solids_balance_residual_g"]) for row in calibration_rows
    )
    max_anchor_rel = max(abs(float(row["relative_difference_percent"])) for row in anchor_check_rows)

    print(f"Maximum water balance residual: {max_water:.3e} g")
    print(f"Maximum dissolved-solids balance residual: {max_solids:.3e} g")
    print(f"Maximum anchor/function relative difference: {max_anchor_rel:.2f} %")
    print("Generated files:")
    for path in (CALIBRATION_CSV, D90_CURVE_CSV, SUMMARY_MD, FIG_PNG, FIG_PDF, CAPTION_MD):
        print(path)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )


def reference_config() -> ModelConfig:
    config = load_config(ROOT / "configs" / "default_v60.json")
    return replace(
        config,
        recipe=replace(config.recipe, total_time_s=700.0, dt_s=0.05, sample_every_s=1.0),
        geometry=replace(config.geometry, axial_layers=24, radial_bins=6),
        scenarios=load_calibrated_psd_scenarios(),
    )


def load_experiment() -> dict[str, dict[str, float]]:
    data = pd.read_csv(EXPERIMENT_FILE)
    result = {}
    for name, group in data.groupby(data["Grind"].str.lower()):
        result[str(name)] = {
            "n": int(len(group)),
            "drawdown_mean": float(group["Extraction time (s)"].mean()),
            "drawdown_sd": float(group["Extraction time (s)"].std(ddof=1)),
            "beverage_mean": float(group["Beverage mass (g)"].mean()),
            "beverage_sd": float(group["Beverage mass (g)"].std(ddof=1)),
            "tds_mean": float(group["TDS (%)"].mean()),
            "tds_sd": float(group["TDS (%)"].std(ddof=1)),
            "ey_mean": float(group["EY (%)"].mean()),
            "ey_sd": float(group["EY (%)"].std(ddof=1)),
        }
    return result


def run_calibration_case(
    base: ModelConfig,
    scenario: Scenario,
    measured: dict[str, float],
) -> dict[str, object]:
    run_config = d90_closure_config_for_scenario(base, scenario)
    coefficients = d90_closure_coefficients_for_scenario(scenario)
    result = run_simulation(run_config, scenario.name)
    summary = result.summary
    simulated_drawdown = float(summary["drawdown_time_s"])
    simulated_beverage = float(summary["cup_mass_g"])
    simulated_tds = float(summary["tds_percent"])
    simulated_ey = float(summary["ey_percent"])
    return {
        "PSD_class": scenario.name,
        "D90_um": mass_percentile_diameter_um(scenario, 90.0),
        "measured_mean_drawdown_time_s": measured["drawdown_mean"],
        "measured_sd_drawdown_time_s": measured["drawdown_sd"],
        "simulated_drawdown_time_s": simulated_drawdown,
        "drawdown_error_s": simulated_drawdown - measured["drawdown_mean"],
        "measured_mean_beverage_mass_g": measured["beverage_mean"],
        "measured_sd_beverage_mass_g": measured["beverage_sd"],
        "simulated_beverage_mass_g": simulated_beverage,
        "beverage_mass_error_g": simulated_beverage - measured["beverage_mean"],
        "measured_mean_TDS_percent": measured["tds_mean"],
        "measured_sd_TDS_percent": measured["tds_sd"],
        "simulated_TDS_percent": simulated_tds,
        "TDS_error_percentage_points": simulated_tds - measured["tds_mean"],
        "measured_mean_extraction_yield_percent": measured["ey_mean"],
        "measured_sd_extraction_yield_percent": measured["ey_sd"],
        "simulated_extraction_yield_percent": simulated_ey,
        "extraction_yield_error_percentage_points": simulated_ey - measured["ey_mean"],
        "retained_water_capacity_g_g": coefficients["retained_water_capacity_g_per_g_coffee"],
        "hydraulic_correction": coefficients["hydraulic_correction_multiplier"],
        "diffusion_like_coefficient_s_1": coefficients["diffusion_rate_ref_s_inv"],
        "surface_like_coefficient_s_1": coefficients["surface_rate_ref_s_inv"],
        "max_water_balance_residual_g": summary["max_water_residual_abs_g"],
        "max_dissolved_solids_balance_residual_g": summary["max_solids_residual_abs_g"],
        "solver_status": solver_status(summary),
    }


def d90_function_curve_rows() -> list[dict[str, float]]:
    rows = []
    for d90_um in np.linspace(700.0, 1700.0, 301):
        values = values_from_d90(float(d90_um))
        rows.append({"D90_um": float(d90_um), **values})
    return rows


def values_from_d90(d90_um: float) -> dict[str, float]:
    return {
        "retained_water_capacity_g_g": D90_CLOSURE_FITS[
            "retained_water_capacity_g_per_g_coffee"
        ]["prefactor"]
        * d90_um
        ** D90_CLOSURE_FITS["retained_water_capacity_g_per_g_coffee"]["exponent"],
        "hydraulic_correction": D90_CLOSURE_FITS["hydraulic_correction_multiplier"][
            "prefactor"
        ]
        * d90_um
        ** D90_CLOSURE_FITS["hydraulic_correction_multiplier"]["exponent"],
        "diffusion_like_coefficient_s_1": D90_CLOSURE_FITS["diffusion_rate_ref_s_inv"][
            "prefactor"
        ]
        * d90_um
        ** D90_CLOSURE_FITS["diffusion_rate_ref_s_inv"]["exponent"],
        "surface_like_coefficient_s_1": D90_CLOSURE_FITS["surface_rate_ref_s_inv"][
            "prefactor"
        ]
        * d90_um
        ** D90_CLOSURE_FITS["surface_rate_ref_s_inv"]["exponent"],
    }


def anchor_check(calibration_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    map_names = {
        "retained_water_capacity_g_g": "retained_water_capacity_g_per_g_coffee",
        "hydraulic_correction": "hydraulic_correction_multiplier",
        "diffusion_like_coefficient_s_1": "diffusion_rate_ref_s_inv",
        "surface_like_coefficient_s_1": "surface_rate_ref_s_inv",
    }
    rows = []
    for row in calibration_rows:
        cls = str(row["PSD_class"])
        for output_name, anchor_name in map_names.items():
            anchor_value = float(CALIBRATED_COEFFICIENTS[cls][anchor_name])
            function_value = float(row[output_name])
            rows.append(
                {
                    "PSD_class": cls,
                    "parameter": output_name,
                    "anchor_value": anchor_value,
                    "D90_function_value": function_value,
                    "absolute_difference": function_value - anchor_value,
                    "relative_difference_percent": 100.0
                    * (function_value - anchor_value)
                    / anchor_value,
                }
            )
    return rows


def solver_status(summary: dict[str, object]) -> str:
    drawdown = float(summary["drawdown_time_s"])
    if not math.isfinite(drawdown):
        return "no_drawdown"
    if float(summary["max_water_residual_abs_g"]) > 1e-6:
        return "water_balance_warning"
    if float(summary["max_solids_residual_abs_g"]) > 1e-8:
        return "solids_balance_warning"
    return "ok"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    calibration_rows: list[dict[str, object]],
    d90_curve_rows: list[dict[str, float]],
) -> None:
    curve = pd.DataFrame(d90_curve_rows)
    rows = pd.DataFrame(calibration_rows)
    rows["label"] = rows["PSD_class"].map(CLASS_LABELS)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.6, 5.7),
        gridspec_kw={"wspace": 0.42, "hspace": 0.52},
    )
    axes_flat = axes.flatten()

    metric_panels = [
        (
            "A",
            "measured_mean_drawdown_time_s",
            "measured_sd_drawdown_time_s",
            "simulated_drawdown_time_s",
            "Drawdown time (s)",
        ),
        (
            "B",
            "measured_mean_beverage_mass_g",
            "measured_sd_beverage_mass_g",
            "simulated_beverage_mass_g",
            "Beverage mass (g)",
        ),
        (
            "C",
            "measured_mean_TDS_percent",
            "measured_sd_TDS_percent",
            "simulated_TDS_percent",
            "TDS (%)",
        ),
        (
            "D",
            "measured_mean_extraction_yield_percent",
            "measured_sd_extraction_yield_percent",
            "simulated_extraction_yield_percent",
            "Extraction yield (%)",
        ),
    ]
    x = np.arange(len(rows))
    for ax, (label, mean_col, sd_col, sim_col, ylabel) in zip(axes_flat[:4], metric_panels):
        add_panel_label(ax, label)
        measured_artist = ax.errorbar(
            x - 0.06,
            rows[mean_col],
            yerr=rows[sd_col],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=3,
            elinewidth=0.8,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Measured mean +/- SD",
        )
        simulated_artist = ax.scatter(
            x + 0.06,
            rows[sim_col],
            marker="s",
            s=28,
            color="#E69F00",
            edgecolor="black",
            linewidth=0.5,
            label="D90-conditioned model",
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rows["label"])
        ax.set_ylabel(ylabel)
        finish_axis(ax)
    axes_flat[0].legend(
        handles=[measured_artist, simulated_artist],
        labels=["Measured mean +/- SD", "D90-conditioned model"],
        frameon=False,
        loc="best",
        handlelength=1.4,
    )

    make_parameter_panel_e(axes_flat[4], rows, curve)
    make_parameter_panel_f(axes_flat[5], rows, curve)

    fig.savefig(FIG_PNG, dpi=300)
    fig.savefig(FIG_PDF)
    plt.close(fig)


def make_parameter_panel_e(ax, rows: pd.DataFrame, curve: pd.DataFrame) -> None:
    add_panel_label(ax, "E")
    medium_retained = float(
        CALIBRATED_COEFFICIENTS["medium"]["retained_water_capacity_g_per_g_coffee"]
    )
    medium_hydraulic = float(CALIBRATED_COEFFICIENTS["medium"]["hydraulic_correction_multiplier"])
    ax.plot(
        curve["D90_um"],
        curve["retained_water_capacity_g_g"] / medium_retained,
        color="#0072B2",
        linestyle="-",
        label="Retained-water capacity",
    )
    ax.plot(
        curve["D90_um"],
        curve["hydraulic_correction"] / medium_hydraulic,
        color="#009E73",
        linestyle="--",
        label="Hydraulic correction",
    )
    for _, row in rows.iterrows():
        cls = str(row["PSD_class"])
        anchor = CALIBRATED_COEFFICIENTS[cls]
        ax.scatter(
            float(row["D90_um"]),
            anchor["retained_water_capacity_g_per_g_coffee"] / medium_retained,
            color="#0072B2",
            marker="o",
            edgecolor="black",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
        ax.scatter(
            float(row["D90_um"]),
            anchor["hydraulic_correction_multiplier"] / medium_hydraulic,
            color="#009E73",
            marker="s",
            edgecolor="black",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
    ax.set_xlabel("Mass-fraction D90 (um)")
    ax.set_ylabel("Parameter value relative to medium")
    ax.set_xlim(690, 1710)
    ax.set_ylim(0.0, 2.3)
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax)


def make_parameter_panel_f(ax, rows: pd.DataFrame, curve: pd.DataFrame) -> None:
    add_panel_label(ax, "F")
    ax.plot(
        curve["D90_um"],
        curve["diffusion_like_coefficient_s_1"],
        color="#CC79A7",
        linestyle="-",
        label="Diffusion-like coefficient",
    )
    ax.plot(
        curve["D90_um"],
        curve["surface_like_coefficient_s_1"],
        color="#D55E00",
        linestyle="--",
        label="Surface-like coefficient",
    )
    for _, row in rows.iterrows():
        cls = str(row["PSD_class"])
        anchor = CALIBRATED_COEFFICIENTS[cls]
        ax.scatter(
            float(row["D90_um"]),
            anchor["diffusion_rate_ref_s_inv"],
            color="#CC79A7",
            marker="o",
            edgecolor="black",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
        ax.scatter(
            float(row["D90_um"]),
            anchor["surface_rate_ref_s_inv"],
            color="#D55E00",
            marker="s",
            edgecolor="black",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Mass-fraction D90 (um)")
    ax.set_ylabel("Coefficient (s$^{-1}$)")
    ax.set_xlim(690, 1710)
    ax.set_ylim(1.0e-4, 7.0e-3)
    ax.legend(frameon=False, loc="upper left")
    finish_axis(ax)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.20,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def finish_axis(ax) -> None:
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def summary_markdown(
    calibration_rows: list[dict[str, object]],
    anchor_check_rows: list[dict[str, object]],
) -> str:
    max_water = max(float(row["max_water_balance_residual_g"]) for row in calibration_rows)
    max_solids = max(
        float(row["max_dissolved_solids_balance_residual_g"]) for row in calibration_rows
    )
    max_anchor_rel = max(abs(float(row["relative_difference_percent"])) for row in anchor_check_rows)
    lines = [
        "# Figure 3 calibration and D90-conditioned closure summary",
        "",
        "Public recipe data were not used. Reported public finish time was not used.",
        "The calculation uses only `data/pour-over data.csv`, `data/PSD.csv`, and the current calibrated model configuration.",
        "",
        "This is matched-experiment calibration evidence, not independent validation.",
        "The D90 functions are descriptive PSD-conditioned closures from three measured PSD anchors, not universal material laws.",
        "",
        "## Measured PSD D90 values and D90-function coefficients used in simulation",
        "",
        "| PSD class | D90 (um) | retained-water capacity (g/g) | hydraulic correction | diffusion-like coefficient (s^-1) | surface-like coefficient (s^-1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in calibration_rows:
        lines.append(
            "| {PSD_class} | {D90_um:.2f} | {retained_water_capacity_g_g:.3f} | {hydraulic_correction:.3f} | {diffusion_like_coefficient_s_1:.6g} | {surface_like_coefficient_s_1:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Class-conditioned calibration anchors",
            "",
            "| PSD class | D90 (um) | retained-water capacity (g/g) | hydraulic correction | diffusion-like coefficient (s^-1) | surface-like coefficient (s^-1) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    d90_by_class = {str(row["PSD_class"]): float(row["D90_um"]) for row in calibration_rows}
    for cls in PSD_ORDER:
        anchor = CALIBRATED_COEFFICIENTS[cls]
        lines.append(
            f"| {cls} | {d90_by_class[cls]:.2f} | "
            f"{anchor['retained_water_capacity_g_per_g_coffee']:.3f} | "
            f"{anchor['hydraulic_correction_multiplier']:.3f} | "
            f"{anchor['diffusion_rate_ref_s_inv']:.6g} | "
            f"{anchor['surface_rate_ref_s_inv']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## D90-conditioned functions",
            "",
            "```text",
            "retained-water capacity = 61.1115 * D90_um^-0.4828",
            "hydraulic correction    = 868951  * D90_um^-1.990",
            "diffusion-like rate     = 3.81942e-08 * D90_um^1.238",
            "surface-like rate       = 4.77428e-07 * D90_um^1.238",
            "```",
            "",
            f"The current D90 functions approximate the class-conditioned anchors; the maximum relative anchor/function difference is {max_anchor_rel:.2f}%. This difference is reported explicitly because the present implementation uses the D90 functions rather than a categorical coefficient lookup.",
            "",
            "## Calibrated output residuals",
            "",
            "| PSD class | drawdown error (s) | beverage mass error (g) | TDS error (percentage points) | extraction-yield error (percentage points) | solver status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in calibration_rows:
        lines.append(
            "| {PSD_class} | {drawdown_error_s:+.1f} | {beverage_mass_error_g:+.2f} | {TDS_error_percentage_points:+.3f} | {extraction_yield_error_percentage_points:+.2f} | {solver_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Maximum water balance residual: {max_water:.3e} g",
            f"- Maximum dissolved-solids balance residual: {max_solids:.3e} g",
            "",
            "The measured coarse, medium, and fine labels are PSD inputs. They are not treated as independent recipe-specific fitting categories in the plotted D90-conditioned implementation.",
        ]
    )
    return "\n".join(lines) + "\n"


def caption_text() -> str:
    return (
        "Figure 3. Matched-experiment calibration and D90-conditioned parameter "
        "functions. (A-D) Measured replicate means +/- SD and calibrated "
        "D90-function model outputs for drawdown time, beverage mass, TDS, and "
        "extraction yield for the measured coarse, medium, and fine PSD inputs. "
        "(E-F) Calibration anchors converted into common PSD-conditioned functions "
        "of mass-fraction D90 for retained-water capacity, hydraulic correction, "
        "and dissolved-solids release coefficients; Panel E values are normalized "
        "to the medium anchor. Public recipe data were not used for calibration. "
        "The D90 functions are descriptive closures from the present measured PSD "
        "anchors, not universal laws.\n"
    )


if __name__ == "__main__":
    main()
