"""Generate JFE revision outputs with the D90-conditioned closure model."""

from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
import statistics
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
    HYDRAULIC_MASS_PERCENTILE,
    calibrated_scenario_from_grind_description,
    d90_closure_coefficients_for_scenario,
    d90_closure_config_for_scenario,
    load_calibrated_psd_scenarios,
)
from run_calibrated_public_recipe_test import (  # noqa: E402
    config_from_public_recipe,
    read_csv,
    normalize_public_data_row,
    truthy,
    validation_row,
)
from run_measured_psd_analysis import mass_percentile_diameter_um  # noqa: E402
from v60_physics.geometry import build_grid  # noqa: E402
from v60_physics.parameters import GeometryConfig, ModelConfig, Scenario, load_config  # noqa: E402
from v60_physics.solver import derive_scenario, run_simulation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "d90_closure"
PUBLIC_DATA_FILE = ROOT / "data" / "public_data.csv"
EXPERIMENT_FILE = ROOT / "data" / "pour-over data.csv"

MATCHED_CSV = OUTPUT_DIR / "d90_closure_matched_experiment.csv"
MATCHED_MD = OUTPUT_DIR / "d90_closure_matched_experiment_summary.md"
MATCHED_FIG = OUTPUT_DIR / "figure_d90_closure_matched_experiment.png"
PUBLIC_CSV = OUTPUT_DIR / "d90_closure_public_recipe_reconstruction.csv"
PUBLIC_MD = OUTPUT_DIR / "d90_closure_public_recipe_reconstruction_summary.md"
PUBLIC_FIG = OUTPUT_DIR / "figure_d90_closure_public_recipe_reconstruction.png"
HYDRAULIC_CSV = OUTPUT_DIR / "d90_closure_hydraulic_percentile_sensitivity.csv"
HYDRAULIC_MD = OUTPUT_DIR / "d90_closure_hydraulic_percentile_sensitivity_summary.md"
HYDRAULIC_FIG = OUTPUT_DIR / "figure_d90_closure_hydraulic_percentile_sensitivity.png"
GEOMETRY_CSV = OUTPUT_DIR / "d90_closure_geometry_module_swap.csv"
GEOMETRY_MD = OUTPUT_DIR / "d90_closure_geometry_module_swap_summary.md"
GEOMETRY_FIG = OUTPUT_DIR / "figure_d90_closure_geometry_module_swap.png"
REPORT_MD = OUTPUT_DIR / "d90_closure_revision_report.md"

PSD_CLASSES = ("coarse", "medium", "fine")
HYDRAULIC_PERCENTILES = (50.0, 80.0, 90.0, 95.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    base = analysis_config(total_time_s=700.0)
    scenarios = load_calibrated_psd_scenarios()
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}

    matched_rows = matched_experiment_rows(base, scenarios)
    write_csv(MATCHED_CSV, matched_rows)
    MATCHED_MD.write_text(matched_summary(matched_rows), encoding="utf-8")
    make_matched_figure(matched_rows, MATCHED_FIG)

    public_rows = public_reconstruction_rows(base, scenario_by_name)
    write_csv(PUBLIC_CSV, public_rows)
    PUBLIC_MD.write_text(public_summary(public_rows), encoding="utf-8")
    make_public_figure(public_rows, PUBLIC_FIG)

    hydraulic_rows = hydraulic_sensitivity_rows(base, scenarios)
    write_csv(HYDRAULIC_CSV, hydraulic_rows)
    HYDRAULIC_MD.write_text(hydraulic_summary(hydraulic_rows), encoding="utf-8")
    make_hydraulic_figure(hydraulic_rows, HYDRAULIC_FIG)

    geometry_rows = geometry_swap_rows(base, scenario_by_name["medium"])
    write_csv(GEOMETRY_CSV, geometry_rows)
    GEOMETRY_MD.write_text(geometry_summary(geometry_rows), encoding="utf-8")
    make_geometry_figure(geometry_rows, GEOMETRY_FIG)

    REPORT_MD.write_text(
        final_report(matched_rows, public_rows, hydraulic_rows, geometry_rows),
        encoding="utf-8",
    )

    for path in (
        MATCHED_CSV,
        MATCHED_MD,
        MATCHED_FIG,
        PUBLIC_CSV,
        PUBLIC_MD,
        PUBLIC_FIG,
        HYDRAULIC_CSV,
        HYDRAULIC_MD,
        HYDRAULIC_FIG,
        GEOMETRY_CSV,
        GEOMETRY_MD,
        GEOMETRY_FIG,
        REPORT_MD,
    ):
        print(path)


def set_style() -> None:
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
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def analysis_config(total_time_s: float = 700.0) -> ModelConfig:
    base = load_config(ROOT / "configs" / "default_v60.json")
    return replace(
        base,
        recipe=replace(base.recipe, total_time_s=total_time_s, dt_s=0.05, sample_every_s=1.0),
        geometry=replace(base.geometry, axial_layers=24, radial_bins=6),
        scenarios=load_calibrated_psd_scenarios(),
    )


def experiment_means() -> dict[str, dict[str, float]]:
    data = pd.read_csv(EXPERIMENT_FILE)
    data["cup_dissolved_solids_g"] = data["Beverage mass (g)"] * data["TDS (%)"] / 100.0
    result = {}
    for name, group in data.groupby(data["Grind"].str.lower()):
        result[name] = {
            "drawdown_time_s": float(group["Extraction time (s)"].mean()),
            "beverage_mass_g": float(group["Beverage mass (g)"].mean()),
            "tds_percent": float(group["TDS (%)"].mean()),
            "ey_percent": float(group["EY (%)"].mean()),
        }
    return result


def matched_experiment_rows(
    base: ModelConfig,
    scenarios: tuple[Scenario, ...],
) -> list[dict[str, object]]:
    exp = experiment_means()
    rows = []
    for scenario in scenarios:
        run_config = d90_closure_config_for_scenario(base, scenario)
        derived = derive_scenario(run_config, scenario)
        coeff = d90_closure_coefficients_for_scenario(scenario)
        result = run_simulation(run_config, scenario.name)
        summary = result.summary
        row = {
            "PSD_class": scenario.name,
            "D90_um": mass_percentile_diameter_um(scenario, 90.0),
            "retained_water_capacity_g_per_g_coffee": coeff[
                "retained_water_capacity_g_per_g_coffee"
            ],
            "hydraulic_correction_multiplier": coeff["hydraulic_correction_multiplier"],
            "diffusion_rate_ref_s_inv": coeff["diffusion_rate_ref_s_inv"],
            "surface_rate_ref_s_inv": coeff["surface_rate_ref_s_inv"],
            "permeability_m2": derived.permeability_m2,
            "experimental_drawdown_time_s": exp[scenario.name]["drawdown_time_s"],
            "simulated_drawdown_time_s": summary["drawdown_time_s"],
            "error_drawdown_time_s": float(summary["drawdown_time_s"])
            - exp[scenario.name]["drawdown_time_s"],
            "experimental_beverage_mass_g": exp[scenario.name]["beverage_mass_g"],
            "simulated_beverage_mass_g": summary["cup_mass_g"],
            "error_beverage_mass_g": float(summary["cup_mass_g"])
            - exp[scenario.name]["beverage_mass_g"],
            "simulated_retained_water_g": summary["bed_water_g"],
            "experimental_TDS_percent": exp[scenario.name]["tds_percent"],
            "simulated_TDS_percent": summary["tds_percent"],
            "error_TDS_percentage_points": float(summary["tds_percent"])
            - exp[scenario.name]["tds_percent"],
            "experimental_extraction_yield_percent": exp[scenario.name]["ey_percent"],
            "simulated_extraction_yield_percent": summary["ey_percent"],
            "error_extraction_yield_percentage_points": float(summary["ey_percent"])
            - exp[scenario.name]["ey_percent"],
            "max_water_balance_residual_g": summary["max_water_residual_abs_g"],
            "max_dissolved_solids_balance_residual_g": summary["max_solids_residual_abs_g"],
            "solver_status": solver_status(summary),
        }
        rows.append(row)
    return rows


def public_rows_input() -> list[dict[str, str]]:
    return [
        normalize_public_data_row(row)
        for row in read_csv(PUBLIC_DATA_FILE)
        if row.get("recipe_id", "") and truthy(row.get("has_reported_finish_time", ""))
    ]


def public_reconstruction_rows(
    base: ModelConfig,
    scenario_by_name: dict[str, Scenario],
) -> list[dict[str, object]]:
    rows = public_rows_input()
    finish_values = [
        float(row["drawdown_or_finish_s"])
        for row in rows
        if row.get("drawdown_or_finish_s", "").strip()
        and math.isfinite(float(row["drawdown_or_finish_s"]))
    ]
    finish_min = min(finish_values)
    finish_max = max(finish_values)
    output = []
    for row in rows:
        config, schedule_quality = config_from_public_recipe(base, row)
        scenario_name = calibrated_scenario_from_grind_description(
            row.get("recipe_id", ""),
            row.get("grind_description", ""),
        )
        scenario = scenario_by_name[scenario_name]
        run_config = d90_closure_config_for_scenario(config, scenario)
        result = run_simulation(run_config, scenario_name)
        out = validation_row(
            row=row,
            config=run_config,
            scenario_name=scenario_name,
            recipe_notes=schedule_quality,
            summary=result.summary,
            public_finish_min_s=finish_min,
            public_finish_max_s=finish_max,
        )
        coeff = d90_closure_coefficients_for_scenario(scenario)
        derived = derive_scenario(run_config, scenario)
        out.update(
            {
                "grind_description": row.get("grind_description", ""),
                "D90_um": mass_percentile_diameter_um(scenario, 90.0),
                "retained_water_capacity_g_per_g_coffee": coeff[
                    "retained_water_capacity_g_per_g_coffee"
                ],
                "hydraulic_correction_multiplier": coeff["hydraulic_correction_multiplier"],
                "diffusion_rate_ref_s_inv": coeff["diffusion_rate_ref_s_inv"],
                "surface_rate_ref_s_inv": coeff["surface_rate_ref_s_inv"],
                "permeability_m2": derived.permeability_m2,
                "retained_water_per_coffee_g_g": float(result.summary["bed_water_g"])
                / float(config.recipe.coffee_mass_g),
            }
        )
        output.append(out)
    return output


def d90_config_with_hydraulic_percentile(
    config: ModelConfig,
    scenario: Scenario,
    percentile: float,
) -> ModelConfig:
    coeff = d90_closure_coefficients_for_scenario(scenario)
    run_config = replace(
        config,
        scenarios=(scenario,),
        material=replace(
            config.material,
            retained_water_capacity_g_per_g_coffee=coeff[
                "retained_water_capacity_g_per_g_coffee"
            ],
        ),
        release=replace(
            config.release,
            diffusion_rate_ref_s_inv=coeff["diffusion_rate_ref_s_inv"],
            surface_rate_ref_s_inv=coeff["surface_rate_ref_s_inv"],
        ),
    )
    derived = derive_scenario(run_config, scenario)
    sauter_um = derived.characteristic_diameter_m * 1e6
    hydraulic_um = mass_percentile_diameter_um(scenario, percentile)
    multiplier = (hydraulic_um / sauter_um) ** 2 * coeff["hydraulic_correction_multiplier"]
    return replace(
        run_config,
        hydraulics=replace(
            run_config.hydraulics,
            permeability_scale=run_config.hydraulics.permeability_scale * multiplier,
        ),
    )


def hydraulic_sensitivity_rows(
    base: ModelConfig,
    scenarios: tuple[Scenario, ...],
) -> list[dict[str, object]]:
    rows = []
    for scenario in scenarios:
        for percentile in HYDRAULIC_PERCENTILES:
            run_config = d90_config_with_hydraulic_percentile(base, scenario, percentile)
            result = run_simulation(run_config, scenario.name)
            summary = result.summary
            rows.append(
                {
                    "PSD_class": scenario.name,
                    "hydraulic_descriptor": f"D{percentile:.0f}",
                    "hydraulic_diameter_um": mass_percentile_diameter_um(scenario, percentile),
                    "drawdown_time_s": summary["drawdown_time_s"],
                    "cup_water_mass_g": summary["cup_mass_g"],
                    "retained_water_g": summary["bed_water_g"],
                    "TDS_percent": summary["tds_percent"],
                    "extraction_yield_percent": summary["ey_percent"],
                    "max_water_balance_residual_g": summary["max_water_residual_abs_g"],
                    "max_dissolved_solids_balance_residual_g": summary["max_solids_residual_abs_g"],
                    "solver_status": solver_status(summary),
                }
            )
    return rows


def geometry_swap_rows(base: ModelConfig, scenario: Scenario) -> list[dict[str, object]]:
    geometries = [
        ("V60 conical", base.geometry),
        ("cylindrical", GeometryConfig(0.032, 0.0, base.geometry.axial_layers, base.geometry.radial_bins)),
        ("flat-bottom shallow", GeometryConfig(0.045, 0.0, base.geometry.axial_layers, base.geometry.radial_bins)),
        ("truncated cone", GeometryConfig(0.018, 20.0, base.geometry.axial_layers, base.geometry.radial_bins)),
    ]
    rows = []
    for name, geometry in geometries:
        geom_config = replace(base, geometry=geometry)
        run_config = d90_closure_config_for_scenario(geom_config, scenario)
        derived = derive_scenario(run_config, scenario)
        grid = build_grid(
            coffee_mass_g=run_config.recipe.coffee_mass_g,
            bulk_density_kg_m3=derived.bulk_density_kg_m3,
            porosity=derived.porosity,
            geometry=run_config.geometry,
            water_density_kg_m3=run_config.material.water_density_kg_m3,
        )
        result = run_simulation(run_config, scenario.name)
        summary = result.summary
        rows.append(
            {
                "geometry_name": name,
                "bed_volume_mL": float(grid.cell_volume_m3.sum()) * 1e6,
                "bed_height_m": grid.bed_height_m,
                "top_radius_m": float(grid.outer_radius_m[-1]),
                "pore_volume_g": float(grid.pore_capacity_g.sum()),
                "cup_water_mass_g": summary["cup_mass_g"],
                "retained_water_g": summary["bed_water_g"],
                "drawdown_time_s": summary["drawdown_time_s"],
                "cup_dissolved_solids_g": summary["cup_dissolved_solids_g"],
                "TDS_percent": summary["tds_percent"],
                "extraction_yield_percent": summary["ey_percent"],
                "max_water_balance_residual_g": summary["max_water_residual_abs_g"],
                "max_dissolved_solids_balance_residual_g": summary["max_solids_residual_abs_g"],
                "solver_status": solver_status(summary),
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


def matched_summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# D90-closure matched-experiment reconstruction",
        "",
        "Coefficients were calculated from mass D90 closure functions. Public recipe data were not used.",
        "",
        "| PSD | D90 (um) | drawdown error (s) | beverage error (g) | TDS error (pp) | EY error (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cls} | {d90:.2f} | {dt:+.1f} | {mass:+.2f} | {tds:+.3f} | {ey:+.2f} |".format(
                cls=row["PSD_class"],
                d90=float(row["D90_um"]),
                dt=float(row["error_drawdown_time_s"]),
                mass=float(row["error_beverage_mass_g"]),
                tds=float(row["error_TDS_percentage_points"]),
                ey=float(row["error_extraction_yield_percentage_points"]),
            )
        )
    lines.extend(balance_lines(rows))
    return "\n".join(lines) + "\n"


def public_summary(rows: list[dict[str, object]]) -> str:
    finite = [row for row in rows if math.isfinite(float(row["finish_abs_error_s"]))]
    passed = [row for row in finite if bool_value(row["inside_local_finish_window"])]
    mae = statistics.mean(float(row["finish_abs_error_s"]) for row in finite)
    med = statistics.median(float(row["finish_abs_error_s"]) for row in finite)
    lines = [
        "# D90-closure public recipe reconstruction",
        "",
        "Public recipe data were not used for coefficient fitting. Reported finish time was used only as a comparison output.",
        "",
        f"- Rows with finish-time comparison: {len(finite)}",
        f"- Local-window agreement: {len(passed)}/{len(finite)} ({100 * len(passed) / len(finite):.1f}%)",
        f"- Mean absolute finish-time error: {mae:.1f} s",
        f"- Median absolute finish-time error: {med:.1f} s",
    ]
    lines.extend(balance_lines(rows, water_key="water_balance_residual_g", solids_key="solids_balance_residual_g"))
    return "\n".join(lines) + "\n"


def hydraulic_summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# D90-closure hydraulic percentile sensitivity",
        "",
        "D90 closure coefficients were fixed. Only the hydraulic percentile used in the permeability calculation was changed.",
        "",
        "| PSD | descriptor | drawdown (s) | cup water (g) | TDS (%) | EY (%) | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {cls} | {desc} | {drawdown} | {cup:.2f} | {tds:.3f} | {ey:.2f} | {status} |".format(
                cls=row["PSD_class"],
                desc=row["hydraulic_descriptor"],
                drawdown=format_number(row["drawdown_time_s"], 1),
                cup=float(row["cup_water_mass_g"]),
                tds=float(row["TDS_percent"]),
                ey=float(row["extraction_yield_percent"]),
                status=row["solver_status"],
            )
        )
    lines.extend(balance_lines(rows))
    return "\n".join(lines) + "\n"


def geometry_summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# D90-closure geometry module swap",
        "",
        "The medium measured PSD and D90 closure coefficients were kept fixed. Only the geometry module was changed.",
        "",
        "| geometry | bed height (m) | drawdown (s) | cup water (g) | TDS (%) | EY (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {height:.4f} | {drawdown:.1f} | {cup:.2f} | {tds:.3f} | {ey:.2f} |".format(
                name=row["geometry_name"],
                height=float(row["bed_height_m"]),
                drawdown=float(row["drawdown_time_s"]),
                cup=float(row["cup_water_mass_g"]),
                tds=float(row["TDS_percent"]),
                ey=float(row["extraction_yield_percent"]),
            )
        )
    lines.extend(balance_lines(rows))
    return "\n".join(lines) + "\n"


def final_report(
    matched_rows: list[dict[str, object]],
    public_rows: list[dict[str, object]],
    hydraulic_rows: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
) -> str:
    return "\n".join(
        [
            "# D90-closure revision report",
            "",
            "The D90-conditioned closure replaces direct coarse/medium/fine coefficient lookup with continuous coefficient functions of measured mass D90.",
            "",
            "Public recipe records were not used for coefficient fitting. They were used only for incomplete-input reconstruction diagnostics after the D90 closure had been fixed.",
            "",
            "## Matched experiment",
            "",
            matched_summary(matched_rows),
            "## Public recipe reconstruction",
            "",
            public_summary(public_rows),
            "## Hydraulic percentile sensitivity",
            "",
            hydraulic_summary(hydraulic_rows),
            "## Geometry module swap",
            "",
            geometry_summary(geometry_rows),
        ]
    )


def balance_lines(
    rows: list[dict[str, object]],
    water_key: str = "max_water_balance_residual_g",
    solids_key: str = "max_dissolved_solids_balance_residual_g",
) -> list[str]:
    water = max(float(row[water_key]) for row in rows)
    solids = max(float(row[solids_key]) for row in rows)
    return [
        "",
        f"- Maximum water balance residual: {water:.3e} g",
        f"- Maximum dissolved-solids balance residual: {solids:.3e} g",
    ]


def make_matched_figure(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["PSD_class"]) for row in rows]
    metrics = [
        ("drawdown_time_s", "Drawdown time (s)"),
        ("beverage_mass_g", "Beverage mass (g)"),
        ("TDS_percent", "TDS (%)"),
        ("extraction_yield_percent", "Extraction yield (%)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))
    x = np.arange(len(labels))
    width = 0.34
    for ax, (key, ylabel) in zip(axes, metrics):
        exp = [float(row[f"experimental_{key}"]) for row in rows]
        sim = [float(row[f"simulated_{key}"]) for row in rows]
        ax.bar(x - width / 2, exp, width, label="Measured", color="#9E9E9E", edgecolor="black", linewidth=0.6)
        ax.bar(x + width / 2, sim, width, label="Simulated", color="#0072B2", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        clean_axis(ax)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_public_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    classes = ["coarse", "medium", "fine"]
    colors = {"coarse": "#0072B2", "medium": "#E69F00", "fine": "#009E73"}
    for idx, cls in enumerate(classes):
        subset = [row for row in rows if row["scenario_used"] == cls]
        x = np.full(len(subset), idx, dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(subset)) if subset else []
        axes[0].scatter(
            x + jitter,
            [float(row["finish_abs_error_s"]) for row in subset],
            s=18,
            color=colors[cls],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
        axes[1].scatter(
            x + jitter,
            [float(row["simulated_drawdown_time_s"]) for row in subset],
            s=18,
            color=colors[cls],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
    axes[0].set_ylabel("Absolute finish-time error (s)")
    axes[1].set_ylabel("Simulated drawdown time (s)")
    for ax in axes:
        ax.set_xticks(range(len(classes)), classes)
        clean_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_hydraulic_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    colors = {"coarse": "#0072B2", "medium": "#E69F00", "fine": "#009E73"}
    for cls in PSD_CLASSES:
        subset = [row for row in rows if row["PSD_class"] == cls]
        x = [float(str(row["hydraulic_descriptor"]).replace("D", "")) for row in subset]
        y = [float(row["drawdown_time_s"]) if math.isfinite(float(row["drawdown_time_s"])) else np.nan for row in subset]
        ax.plot(x, y, marker="o", label=cls, color=colors[cls])
    ax.set_xlabel("Hydraulic mass percentile")
    ax.set_ylabel("Drawdown time (s)")
    clean_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_geometry_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    labels = [str(row["geometry_name"]) for row in rows]
    x = np.arange(len(labels))
    axes[0].bar(x, [float(row["drawdown_time_s"]) for row in rows], color="#0072B2", edgecolor="black", linewidth=0.6)
    axes[1].bar(x, [float(row["extraction_yield_percent"]) for row in rows], color="#E69F00", edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Drawdown time (s)")
    axes[1].set_ylabel("Extraction yield (%)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        clean_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def format_number(value: object, digits: int) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "not reached"
    return f"{number:.{digits}f}"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
