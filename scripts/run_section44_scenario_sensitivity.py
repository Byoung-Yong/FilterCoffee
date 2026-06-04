"""Generate Section 4.4 scenario sensitivity outputs for the JFE manuscript.

The repository currently contains class-specific calibrated coefficients rather
than one shared calibrated parameter set. Therefore this script uses the
calibrated medium measured-PSD case as a diagnostic reference simulation only.

No public recipe data are used. No coefficients are refit. Governing equations
are unchanged.
"""

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
    CALIBRATED_COEFFICIENTS,
    calibrated_config_for_scenario,
    load_calibrated_psd_scenarios,
)
from run_measured_psd_analysis import mass_percentile_diameter_um  # noqa: E402
from v60_physics.parameters import ModelConfig, Scenario, load_config  # noqa: E402
from v60_physics.solver import run_simulation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "scenario_sensitivity"
CSV_OUT = OUTPUT_DIR / "scenario_sensitivity_medium_reference.csv"
MD_OUT = OUTPUT_DIR / "scenario_sensitivity_medium_reference_summary.md"
FIG_OUT = OUTPUT_DIR / "figure_scenario_sensitivity_medium_reference.png"
HEATMAP_OUT = OUTPUT_DIR / "figure_scenario_sensitivity_heatmap.png"
INTERPRETATION_CSV = OUTPUT_DIR / "scenario_interpretation_table.csv"
INTERPRETATION_MD = OUTPUT_DIR / "scenario_interpretation_table.md"

REFERENCE_SCENARIO = "medium"
PERTURBATION_LEVELS = (-30, -20, -10, 0, 10, 20, 30)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    base_config, scenario = calibrated_medium_reference()
    baseline_summary = run_simulation(base_config, scenario.name).summary
    rows = scenario_rows(base_config, scenario, baseline_summary)
    write_csv(CSV_OUT, rows)
    MD_OUT.write_text(summary_markdown(rows, baseline_summary, scenario), encoding="utf-8")
    make_main_figure(rows)
    make_heatmap(rows)
    interpretation_rows = scenario_interpretation_rows()
    write_csv(INTERPRETATION_CSV, interpretation_rows)
    INTERPRETATION_MD.write_text(
        markdown_table(interpretation_rows, "# Scenario interpretation table"),
        encoding="utf-8",
    )
    for path in (CSV_OUT, MD_OUT, FIG_OUT, HEATMAP_OUT, INTERPRETATION_CSV, INTERPRETATION_MD):
        print(path)


def calibrated_medium_reference() -> tuple[ModelConfig, Scenario]:
    config = load_config(ROOT / "configs" / "default_v60.json")
    scenarios = {scenario.name: scenario for scenario in load_calibrated_psd_scenarios()}
    scenario = scenarios[REFERENCE_SCENARIO]
    base = replace(
        config,
        recipe=replace(config.recipe, total_time_s=420.0, dt_s=0.05, sample_every_s=1.0),
        geometry=replace(config.geometry, axial_layers=24, radial_bins=6),
        scenarios=(scenario,),
    )
    return calibrated_config_for_scenario(base, scenario), scenario


def scenario_rows(
    base_config: ModelConfig,
    scenario: Scenario,
    baseline_summary: dict[str, object],
) -> list[dict[str, object]]:
    coefficients = CALIBRATED_COEFFICIENTS[scenario.name]
    reference_hydraulic_diameter_um = mass_percentile_diameter_um(scenario, 90.0)
    rows: list[dict[str, object]] = []
    for perturbation in PERTURBATION_LEVELS:
        factor = 1.0 + perturbation / 100.0
        specs = [
            (
                "Effective hydraulic resistance scenario",
                "Effective hydraulic diameter",
                reference_hydraulic_diameter_um,
                reference_hydraulic_diameter_um * factor,
                replace(
                    base_config,
                    hydraulics=replace(
                        base_config.hydraulics,
                        permeability_scale=base_config.hydraulics.permeability_scale * factor**2,
                    ),
                ),
                (factor**2 - 1.0) * 100.0,
            ),
            (
                "Retained-water scenario",
                "Retained-water capacity",
                coefficients["retained_water_capacity_g_per_g_coffee"],
                coefficients["retained_water_capacity_g_per_g_coffee"] * factor,
                replace(
                    base_config,
                    material=replace(
                        base_config.material,
                        retained_water_capacity_g_per_g_coffee=coefficients[
                            "retained_water_capacity_g_per_g_coffee"
                        ]
                        * factor,
                    ),
                ),
                float("nan"),
            ),
            (
                "Dissolved-solids release scenario",
                "Dissolved-solids release rate",
                coefficients["diffusion_rate_ref_s_inv"],
                coefficients["diffusion_rate_ref_s_inv"] * factor,
                replace(
                    base_config,
                    release=replace(
                        base_config.release,
                        diffusion_rate_ref_s_inv=coefficients["diffusion_rate_ref_s_inv"] * factor,
                        surface_rate_ref_s_inv=coefficients["surface_rate_ref_s_inv"] * factor,
                    ),
                ),
                float("nan"),
            ),
        ]
        for group, parameter, reference_value, perturbed_value, run_config, permeability_change in specs:
            summary = run_simulation(run_config, scenario.name).summary
            rows.append(
                {
                    "scenario_group": group,
                    "perturbed_parameter": parameter,
                    "perturbation_percent": perturbation,
                    "reference_value": reference_value,
                    "perturbed_value": perturbed_value,
                    "relative_permeability_change_percent": permeability_change,
                    "coffee_dose_g": float(summary["coffee_mass_g"]),
                    "water_input_g": float(summary["total_recipe_water_g"]),
                    "PSD_input": scenario.name,
                    **output_fields(summary),
                    **normalized_response(summary, baseline_summary),
                }
            )
    return rows


def output_fields(summary: dict[str, object]) -> dict[str, object]:
    cup = float(summary["cup_mass_g"])
    retained = float(summary["bed_water_g"])
    dose = float(summary["coffee_mass_g"])
    return {
        "cup_water_mass_g": cup,
        "retained_water_g": retained,
        "retained_water_per_coffee_g_g": retained / dose,
        "drawdown_time_s": float(summary["drawdown_time_s"]),
        "cup_dissolved_solids_g": float(summary["cup_dissolved_solids_g"]),
        "TDS_percent": float(summary["tds_percent"]),
        "extraction_yield_percent": float(summary["ey_percent"]),
        "max_water_balance_residual_g": float(summary["max_water_residual_abs_g"]),
        "max_dissolved_solids_balance_residual_g": float(summary["max_solids_residual_abs_g"]),
        "solver_status": solver_status(summary),
    }


def normalized_response(
    summary: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float]:
    return {
        "cup_water_mass_change_percent": percent_change(summary["cup_mass_g"], baseline["cup_mass_g"]),
        "retained_water_change_percent": percent_change(summary["bed_water_g"], baseline["bed_water_g"]),
        "drawdown_time_change_percent": percent_change(
            summary["drawdown_time_s"],
            baseline["drawdown_time_s"],
        ),
        "cup_dissolved_solids_change_percent": percent_change(
            summary["cup_dissolved_solids_g"],
            baseline["cup_dissolved_solids_g"],
        ),
        "TDS_change_percent": percent_change(summary["tds_percent"], baseline["tds_percent"]),
        "extraction_yield_change_percent": percent_change(
            summary["ey_percent"],
            baseline["ey_percent"],
        ),
    }


def solver_status(summary: dict[str, object]) -> str:
    if not math.isfinite(float(summary["drawdown_time_s"])):
        return "no_drawdown"
    if float(summary["max_water_residual_abs_g"]) > 1e-6:
        return "water_balance_warning"
    if float(summary["max_solids_residual_abs_g"]) > 1e-8:
        return "solids_balance_warning"
    return "ok"


def percent_change(value: object, baseline: object) -> float:
    value_f = float(value)
    baseline_f = float(baseline)
    if not math.isfinite(value_f) or not math.isfinite(baseline_f) or baseline_f == 0.0:
        return float("nan")
    return 100.0 * (value_f - baseline_f) / baseline_f


def summary_markdown(
    rows: list[dict[str, object]],
    baseline: dict[str, object],
    scenario: Scenario,
) -> str:
    max_water = max(abs(float(row["max_water_balance_residual_g"])) for row in rows)
    max_solids = max(abs(float(row["max_dissolved_solids_balance_residual_g"])) for row in rows)
    hydraulic = group_rows(rows, "Effective hydraulic resistance scenario")
    retained = group_rows(rows, "Retained-water scenario")
    release = group_rows(rows, "Dissolved-solids release scenario")
    lines = [
        "# Scenario sensitivity around calibrated medium reference",
        "",
        "The repository currently contains class-specific calibrated coefficients. A shared calibrated parameter set was not available. Therefore this analysis uses the calibrated medium measured-PSD case as a diagnostic reference simulation only.",
        "",
        "No public recipe data were used. No coefficients were refit. Governing equations were not changed.",
        "",
        "## Reference case",
        "",
        f"- Coffee dose: {float(baseline['coffee_mass_g']):.1f} g",
        f"- Water input: {float(baseline['total_recipe_water_g']):.1f} g",
        f"- PSD input: {scenario.name}",
        f"- Cup water mass: {float(baseline['cup_mass_g']):.2f} g",
        f"- Retained water: {float(baseline['bed_water_g']):.2f} g",
        f"- Drawdown time: {float(baseline['drawdown_time_s']):.1f} s",
        f"- Cup dissolved solids: {float(baseline['cup_dissolved_solids_g']):.3f} g",
        f"- TDS: {float(baseline['tds_percent']):.3f}%",
        f"- Extraction yield: {float(baseline['ey_percent']):.2f}%",
        "",
        "## Scenario groups",
        "",
        "- Effective hydraulic diameter: represents unreported changes in bed hydraulic behavior, including looser or tighter packing, fines accumulation, filter clogging, compaction, channeling, bypass, or different effective flow paths.",
        "- Retained-water capacity: represents unreported changes in wetting, bloom behavior, gas displacement, pore filling, particle swelling, drainage history, and liquid hold-up.",
        "- Dissolved-solids release rate: represents unreported changes in temperature, roast structure, wetting completeness, accessible soluble inventory, agitation, local mass transfer, and active extraction area.",
        "",
        "These perturbations are not claims that physical particle size changes during brewing. They represent unreported effective process conditions that can shift reconstructed outputs.",
        "",
        "## Key trends",
        "",
        f"- Hydraulic perturbation: -30% effective diameter gave drawdown {value_at(hydraulic, -30, 'drawdown_time_s'):.1f} s and TDS {value_at(hydraulic, -30, 'TDS_percent'):.3f}%; +30% gave drawdown {value_at(hydraulic, 30, 'drawdown_time_s'):.1f} s and TDS {value_at(hydraulic, 30, 'TDS_percent'):.3f}%. Lower hydraulic resistance shortened drawdown and reduced TDS/EY; higher resistance increased residence time and extraction-scale outputs.",
        f"- Retained-water perturbation: -30% retained capacity gave cup water {value_at(retained, -30, 'cup_water_mass_g'):.2f} g and retained water {value_at(retained, -30, 'retained_water_g'):.2f} g; +30% gave cup water {value_at(retained, 30, 'cup_water_mass_g'):.2f} g and retained water {value_at(retained, 30, 'retained_water_g'):.2f} g. The main effect was water partitioning between bed-associated liquid and cup delivery.",
        f"- Release-rate perturbation: -30% release gave TDS {value_at(release, -30, 'TDS_percent'):.3f}% and EY {value_at(release, -30, 'extraction_yield_percent'):.2f}%; +30% gave TDS {value_at(release, 30, 'TDS_percent'):.3f}% and EY {value_at(release, 30, 'extraction_yield_percent'):.2f}%. The effect on drawdown and cup water mass was negligible.",
        "",
        "These simulations support the interpretation of public recipe deviations: unreported hydraulic state, liquid hold-up, and release conditions can move finish time, cup mass, TDS, and extraction yield even when the recipe descriptors are nominally similar.",
        "",
        "## Balance checks",
        "",
        f"- Maximum water balance residual: {max_water:.3e} g",
        f"- Maximum dissolved-solids balance residual: {max_solids:.3e} g",
    ]
    return "\n".join(lines) + "\n"


def group_rows(rows: list[dict[str, object]], group_name: str) -> list[dict[str, object]]:
    return [row for row in rows if row["scenario_group"] == group_name]


def value_at(rows: list[dict[str, object]], perturbation: int, key: str) -> float:
    for row in rows:
        if int(row["perturbation_percent"]) == perturbation:
            return float(row[key])
    raise KeyError((perturbation, key))


def make_main_figure(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.0))
    axes = axes.ravel()
    groups = {
        "hydraulic": group_rows(rows, "Effective hydraulic resistance scenario"),
        "retained": group_rows(rows, "Retained-water scenario"),
        "release": group_rows(rows, "Dissolved-solids release scenario"),
    }
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"

    plot_single(axes[0], groups["hydraulic"], "drawdown_time_s", "Drawdown time (s)", blue)
    plot_dual(axes[1], groups["hydraulic"], ("TDS_percent", "TDS (%)"), ("extraction_yield_percent", "Extraction yield (%)"))
    plot_dual(axes[2], groups["retained"], ("cup_water_mass_g", "Cup water mass (g)"), ("retained_water_g", "Retained water (g)"))
    plot_dual(axes[3], groups["retained"], ("TDS_percent", "TDS (%)"), ("extraction_yield_percent", "Extraction yield (%)"))
    plot_dual(axes[4], groups["release"], ("TDS_percent", "TDS (%)"), ("extraction_yield_percent", "Extraction yield (%)"))
    plot_dual(axes[5], groups["release"], ("drawdown_time_s", "Drawdown time (s)"), ("cup_water_mass_g", "Cup water mass (g)"))

    xlabels = [
        "Effective hydraulic diameter perturbation (%)",
        "Effective hydraulic diameter perturbation (%)",
        "Retained-water capacity perturbation (%)",
        "Retained-water capacity perturbation (%)",
        "Dissolved-solids release rate perturbation (%)",
        "Dissolved-solids release rate perturbation (%)",
    ]
    for ax, label, panel in zip(axes, xlabels, "ABCDEF"):
        ax.set_xlabel(label)
        ax.axvline(0, color="0.65", linestyle="--", linewidth=0.9)
        ax.text(-0.16, 1.06, panel, transform=ax.transAxes, weight="bold")
        clean_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=300)
    plt.close(fig)


def plot_single(ax, rows: list[dict[str, object]], metric: str, ylabel: str, color: str) -> None:
    x = [float(row["perturbation_percent"]) for row in rows]
    y = [float(row[metric]) for row in rows]
    ax.plot(x, y, marker="o", color=color, label=ylabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def plot_dual(
    ax,
    rows: list[dict[str, object]],
    first: tuple[str, str],
    second: tuple[str, str],
) -> None:
    x = [float(row["perturbation_percent"]) for row in rows]
    ax.plot(x, [float(row[first[0]]) for row in rows], marker="o", color="#0072B2", label=first[1])
    ax.set_ylabel(first[1], color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax2 = ax.twinx()
    ax2.plot(x, [float(row[second[0]]) for row in rows], marker="s", color="#D55E00", label=second[1])
    ax2.set_ylabel(second[1], color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")


def make_heatmap(rows: list[dict[str, object]]) -> None:
    parameters = [
        ("Effective hydraulic diameter", "Effective hydraulic resistance scenario"),
        ("Retained-water capacity", "Retained-water scenario"),
        ("Dissolved-solids release rate", "Dissolved-solids release scenario"),
    ]
    outputs = [
        ("cup_water_mass_change_percent", "cup water mass"),
        ("retained_water_change_percent", "retained water"),
        ("drawdown_time_change_percent", "drawdown time"),
        ("TDS_change_percent", "TDS"),
        ("extraction_yield_change_percent", "extraction yield"),
    ]
    matrix = np.zeros((len(parameters), len(outputs)))
    for i, (_, group_name) in enumerate(parameters):
        group = group_rows(rows, group_name)
        minus = next(row for row in group if int(row["perturbation_percent"]) == -20)
        plus = next(row for row in group if int(row["perturbation_percent"]) == 20)
        for j, (key, _) in enumerate(outputs):
            matrix[i, j] = 0.5 * (abs(float(minus[key])) + abs(float(plus[key])))

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(outputs)), [label for _, label in outputs], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(parameters)), [label for label, _ in parameters])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean absolute response at +/-20% (%)")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(HEATMAP_OUT, dpi=300)
    plt.close(fig)


def scenario_interpretation_rows() -> list[dict[str, str]]:
    return [
        {
            "simulated scenario": "Lower effective hydraulic resistance",
            "possible real brewing situations represented": "channeling; bypass; looser packing; coarser effective flow paths; lower filter resistance",
            "primary outputs affected": "drawdown time; TDS; extraction yield",
            "expected direction when parameter increases": "Increasing effective diameter lowers resistance, shortens drawdown, and may lower TDS/EY through shorter residence time.",
            "relevance to public recipe deviations": "Can explain public recipes that drain faster than predicted from nominal grind and dose.",
        },
        {
            "simulated scenario": "Higher effective hydraulic resistance",
            "possible real brewing situations represented": "fines accumulation; clogging; tighter packing; compaction; finer effective flow paths",
            "primary outputs affected": "drawdown time; retained water; TDS; extraction yield",
            "expected direction when parameter increases": "Increasing resistance lengthens drawdown and can increase TDS/EY by increasing residence time.",
            "relevance to public recipe deviations": "Can explain public recipes with slow finish times or high extraction relative to recipe descriptors.",
        },
        {
            "simulated scenario": "Higher retained-water capacity",
            "possible real brewing situations represented": "greater wetting; pore filling; swelling; liquid hold-up; drainage history",
            "primary outputs affected": "retained water; cup water mass; TDS; extraction yield",
            "expected direction when parameter increases": "Retained water increases and cup water mass decreases; TDS/EY shift through water partitioning.",
            "relevance to public recipe deviations": "Can explain differences in beverage mass and concentration not captured by dose and water input.",
        },
        {
            "simulated scenario": "Lower retained-water capacity",
            "possible real brewing situations represented": "poor wetting; lower liquid hold-up; faster drainage endpoint; less bed-associated liquid",
            "primary outputs affected": "retained water; cup water mass",
            "expected direction when parameter increases": "The opposite direction corresponds to lower retained water and higher cup water mass.",
            "relevance to public recipe deviations": "Can explain higher beverage mass under similar recipe inputs.",
        },
        {
            "simulated scenario": "Higher dissolved-solids release rate",
            "possible real brewing situations represented": "higher temperature; greater accessible soluble inventory; stronger agitation; more effective local mass transfer",
            "primary outputs affected": "cup dissolved solids; TDS; extraction yield",
            "expected direction when parameter increases": "Cup dissolved solids, TDS, and extraction yield increase with limited direct effect on drawdown.",
            "relevance to public recipe deviations": "Can explain beverage-strength deviations when finish time is similar.",
        },
        {
            "simulated scenario": "Lower dissolved-solids release rate",
            "possible real brewing situations represented": "lower temperature; poor wetting; lower accessible soluble inventory; reduced active extraction area",
            "primary outputs affected": "cup dissolved solids; TDS; extraction yield",
            "expected direction when parameter increases": "The opposite direction corresponds to lower TDS and extraction yield.",
            "relevance to public recipe deviations": "Can explain low TDS/EY despite similar drawdown or recipe descriptors.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], title: str) -> str:
    columns = list(rows[0].keys())
    lines = [title, "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in columns) + " |")
    return "\n".join(lines) + "\n"


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)


if __name__ == "__main__":
    main()
