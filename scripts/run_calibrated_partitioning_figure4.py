"""Generate calibrated water and dissolved-solids partitioning for Figure 4."""

from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrated_v60_model import (  # noqa: E402
    d90_closure_coefficients_for_scenario,
    d90_closure_config_for_scenario,
    load_calibrated_psd_scenarios,
)
from run_measured_psd_analysis import mass_percentile_diameter_um  # noqa: E402
from v60_physics.parameters import load_config  # noqa: E402
from v60_physics.solver import run_simulation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "calibrated_partitioning"
PARTITION_CSV = OUTPUT_DIR / "calibrated_water_solids_partitioning.csv"
SUMMARY_MD = OUTPUT_DIR / "calibrated_water_solids_partitioning_summary.md"
FIG_OUT = OUTPUT_DIR / "figure_calibrated_partitioning.png"
TRACE_CSV = OUTPUT_DIR / "calibrated_partitioning_time_traces.csv"

PSD_ORDER = ("coarse", "medium", "fine")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = reference_config()
    scenarios = {scenario.name: scenario for scenario in load_calibrated_psd_scenarios()}
    rows = []
    traces = []
    for name in PSD_ORDER:
        scenario = scenarios[name]
        run_config = d90_closure_config_for_scenario(base, scenario)
        result = run_simulation(run_config, name)
        endpoint = endpoint_row(result.timeseries, float(result.summary["drawdown_time_s"]))
        rows.append(partition_row(name, scenario, run_config, result.summary, endpoint))
        traces.extend(trace_rows(name, result.timeseries))

    write_csv(PARTITION_CSV, rows)
    write_csv(TRACE_CSV, traces)
    SUMMARY_MD.write_text(summary_markdown(rows), encoding="utf-8")
    make_figure(rows)

    for path in (PARTITION_CSV, SUMMARY_MD, FIG_OUT, TRACE_CSV):
        print(path)


def reference_config():
    config = load_config(ROOT / "configs" / "default_v60.json")
    return replace(
        config,
        recipe=replace(config.recipe, total_time_s=700.0, dt_s=0.05, sample_every_s=1.0),
        geometry=replace(config.geometry, axial_layers=24, radial_bins=6),
        scenarios=load_calibrated_psd_scenarios(),
    )


def endpoint_row(timeseries: list[dict[str, object]], drawdown_time_s: float) -> dict[str, object]:
    if not math.isfinite(drawdown_time_s):
        return timeseries[-1]
    return min(timeseries, key=lambda row: abs(float(row["time_s"]) - drawdown_time_s))


def partition_row(name, scenario, config, summary, endpoint) -> dict[str, object]:
    coeff = d90_closure_coefficients_for_scenario(scenario)
    input_water = float(endpoint["total_input_water_g"])
    cup_water = float(endpoint["cup_water_g"])
    retained = float(endpoint["retained_water_g"])
    mobile = float(endpoint["pore_water_g"])
    pooled = float(endpoint["pool_water_g"])
    accounted_water = cup_water + retained + mobile + pooled
    water_residual = input_water - accounted_water

    initial_solids = float(endpoint["initial_extractable_solids_g"])
    cup_solids = float(endpoint["cup_solids_g"])
    bed_liquid_solids = float(endpoint["bed_liquid_solids_g"])
    remaining_solids = float(endpoint["remaining_extractable_solids_g"])
    accounted_solids = cup_solids + bed_liquid_solids + remaining_solids
    solids_residual = initial_solids - accounted_solids
    coffee_mass = float(config.recipe.coffee_mass_g)

    return {
        "PSD_class": name,
        "D90_um": mass_percentile_diameter_um(scenario, 90.0),
        "retained_water_capacity_g_per_g_coffee": coeff[
            "retained_water_capacity_g_per_g_coffee"
        ],
        "hydraulic_correction_multiplier": coeff["hydraulic_correction_multiplier"],
        "diffusion_rate_ref_s_inv": coeff["diffusion_rate_ref_s_inv"],
        "surface_rate_ref_s_inv": coeff["surface_rate_ref_s_inv"],
        "input_water_g": input_water,
        "cup_water_g": cup_water,
        "retained_water_g": retained,
        "mobile_bed_water_g": mobile,
        "pooled_water_g": pooled,
        "total_accounted_water_g": accounted_water,
        "water_balance_residual_g": water_residual,
        "drawdown_time_s": float(summary["drawdown_time_s"]),
        "initial_extractable_solids_g": initial_solids,
        "cup_dissolved_solids_g": cup_solids,
        "bed_liquid_dissolved_solids_g": bed_liquid_solids,
        "remaining_extractable_solids_g": remaining_solids,
        "total_accounted_solids_g": accounted_solids,
        "dissolved_solids_balance_residual_g": solids_residual,
        "TDS_percent": 100.0 * cup_solids / cup_water if cup_water > 0 else 0.0,
        "extraction_yield_percent": 100.0 * cup_solids / coffee_mass,
        "retained_water_per_coffee_g_g": retained / coffee_mass,
        "cup_water_fraction_of_input_percent": 100.0 * cup_water / input_water,
        "retained_water_fraction_of_input_percent": 100.0 * retained / input_water,
        "mobile_bed_water_fraction_of_input_percent": 100.0 * mobile / input_water,
        "pooled_water_fraction_of_input_percent": 100.0 * pooled / input_water,
        "cup_solids_fraction_of_initial_extractable_percent": 100.0 * cup_solids / initial_solids,
        "bed_liquid_solids_fraction_of_initial_extractable_percent": 100.0
        * bed_liquid_solids
        / initial_solids,
        "remaining_solids_fraction_of_initial_extractable_percent": 100.0
        * remaining_solids
        / initial_solids,
        "max_water_balance_residual_g": summary["max_water_residual_abs_g"],
        "max_dissolved_solids_balance_residual_g": summary["max_solids_residual_abs_g"],
        "solver_status": solver_status(summary),
    }


def trace_rows(name: str, timeseries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in timeseries:
        cup_water = float(row["cup_water_g"])
        cup_solids = float(row["cup_solids_g"])
        rows.append(
            {
                "time_s": row["time_s"],
                "PSD_class": name,
                "input_water_g": float(row["total_input_water_g"]),
                "cup_water_g": row["cup_water_g"],
                "retained_water_g": row["retained_water_g"],
                "mobile_bed_water_g": row["pore_water_g"],
                "pooled_water_g": row["pool_water_g"],
                "cup_dissolved_solids_g": row["cup_solids_g"],
                "bed_liquid_dissolved_solids_g": row["bed_liquid_solids_g"],
                "remaining_extractable_solids_g": row["remaining_extractable_solids_g"],
                "instantaneous_outflow_g_s": row["outlet_flow_g_s"],
                "TDS_percent": 100.0 * cup_solids / cup_water if cup_water > 0 else 0.0,
                "extraction_yield_percent": row["ey_percent"],
            }
        )
    return rows


def solver_status(summary: dict[str, object]) -> str:
    if not math.isfinite(float(summary["drawdown_time_s"])):
        return "no_drawdown"
    if float(summary["max_water_residual_abs_g"]) > 1e-6:
        return "water_balance_warning"
    if float(summary["max_solids_residual_abs_g"]) > 1e-8:
        return "solids_balance_warning"
    return "ok"


def summary_markdown(rows: list[dict[str, object]]) -> str:
    max_water = max(abs(float(row["water_balance_residual_g"])) for row in rows)
    max_solids = max(abs(float(row["dissolved_solids_balance_residual_g"])) for row in rows)
    lines = [
        "# Calibrated water and dissolved-solids partitioning",
        "",
        "Reference recipe: 20 g coffee and 300 g water with pours of 60 g from 0-15 s, 120 g from 30-70 s, and 120 g from 80-120 s.",
        "PSD classes: measured coarse, medium, and fine PSD inputs from `data/PSD.csv`.",
        "Calibration: D90-conditioned parameter functions. Public recipe data were not used.",
        "",
        "These are calibrated model-reconstructed internal inventories, not independently measured internal states.",
        "",
        "## D90-conditioned parameter values",
        "",
        "| PSD | D90 (um) | retained capacity (g/g) | hydraulic correction | diffusion coefficient (s^-1) | surface coefficient (s^-1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {PSD_class} | {D90_um:.2f} | {retained_water_capacity_g_per_g_coffee:.3f} | {hydraulic_correction_multiplier:.3f} | {diffusion_rate_ref_s_inv:.6g} | {surface_rate_ref_s_inv:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Water partitioning",
            "",
            "| PSD | cup water (g) | retained water (g) | mobile bed water (g) | pooled water (g) | drawdown (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {PSD_class} | {cup_water_g:.2f} | {retained_water_g:.2f} | {mobile_bed_water_g:.4f} | {pooled_water_g:.4f} | {drawdown_time_s:.1f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Dissolved-solids partitioning",
            "",
            "| PSD | cup dissolved solids (g) | bed-liquid dissolved solids (g) | remaining extractable solids (g) | TDS (%) | EY (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {PSD_class} | {cup_dissolved_solids_g:.4f} | {bed_liquid_dissolved_solids_g:.4f} | {remaining_extractable_solids_g:.4f} | {TDS_percent:.3f} | {extraction_yield_percent:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Maximum water balance residual at drawdown endpoint: {max_water:.3e} g",
            f"- Maximum dissolved-solids balance residual at drawdown endpoint: {max_solids:.3e} g",
        ]
    )
    return "\n".join(lines) + "\n"


def make_figure(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    labels = [str(row["PSD_class"]) for row in rows]
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    stack_bars(
        ax_a,
        x,
        rows,
        [
            ("cup_water_g", "Cup water", "#0072B2"),
            ("retained_water_g", "Retained water", "#E69F00"),
            ("mobile_bed_water_g", "Mobile bed water", "#009E73"),
            ("pooled_water_g", "Pooled water", "#CC79A7"),
        ],
        "Water mass (g)",
    )
    ax_a.set_xticks(x, labels)
    ax_a.text(0.02, 0.95, "A", transform=ax_a.transAxes, fontweight="bold", va="top")

    stack_bars(
        ax_b,
        x,
        rows,
        [
            ("cup_dissolved_solids_g", "Cup dissolved solids", "#0072B2"),
            ("bed_liquid_dissolved_solids_g", "Bed-liquid dissolved solids", "#009E73"),
            ("remaining_extractable_solids_g", "Remaining extractable solids", "#999999"),
        ],
        "Dissolved-solids inventory (g)",
    )
    ax_b.set_xticks(x, labels)
    ax_b.text(0.02, 0.95, "B", transform=ax_b.transAxes, fontweight="bold", va="top")

    output_metrics = [
        ("drawdown_time_s", "Drawdown time (s)", "#0072B2"),
        ("TDS_percent", "TDS (%)", "#D55E00"),
        ("extraction_yield_percent", "Extraction yield (%)", "#009E73"),
    ]
    width = 0.24
    for offset, (key, label, color) in zip((-width, 0.0, width), output_metrics):
        values = np.array([float(row[key]) for row in rows], dtype=float)
        normalized = values / values[1] * 100.0
        ax_c.bar(x + offset, normalized, width, label=label, color=color, edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(x, labels)
    ax_c.set_ylabel("Output relative to medium (%)")
    ax_c.legend(frameon=False, loc="best")
    ax_c.text(0.02, 0.95, "C", transform=ax_c.transAxes, fontweight="bold", va="top")

    water_res = [max(abs(float(row["water_balance_residual_g"])), 1e-16) for row in rows]
    solids_res = [max(abs(float(row["dissolved_solids_balance_residual_g"])), 1e-16) for row in rows]
    ax_d.bar(x - 0.18, water_res, 0.36, label="Water", color="#56B4E9", edgecolor="black", linewidth=0.5)
    ax_d.bar(x + 0.18, solids_res, 0.36, label="Dissolved solids", color="#F0E442", edgecolor="black", linewidth=0.5)
    ax_d.set_yscale("log")
    ax_d.set_xticks(x, labels)
    ax_d.set_ylabel("Absolute balance residual (g)")
    ax_d.legend(frameon=False, loc="best")
    ax_d.text(0.02, 0.95, "D", transform=ax_d.transAxes, fontweight="bold", va="top")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        clean_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def stack_bars(ax, x, rows, components, ylabel):
    bottom = np.zeros(len(rows))
    for key, label, color in components:
        values = np.array([max(float(row[key]), 0.0) for row in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.5)
        bottom += values
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
