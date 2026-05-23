"""Run baseline simulations using measured surface-area PSD data.

The input file ``data/PSD.csv`` contains surface-area relative-bin
distributions for fine, medium, and coarse grind classes. The solver assigns
extractable solids by coffee mass, so surface-area bins are converted to mass
fractions using the geometric relation mass fraction proportional to
surface-area fraction times particle diameter.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v60_physics.parameters import ModelConfig, ParticleClass, Scenario, load_config  # noqa: E402
from v60_physics.solver import derive_scenario, run_simulation  # noqa: E402


PSD_FILE = ROOT / "data" / "PSD.csv"
OUTPUT_DIR = ROOT / "outputs" / "psd"
CLASSES_CSV = OUTPUT_DIR / "measured_psd_classes.csv"
SIMULATION_CSV = OUTPUT_DIR / "measured_psd_simulation.csv"
SUMMARY_MD = OUTPUT_DIR / "measured_psd_simulation_summary.md"

PSD_CLASSES = ("fine", "medium", "coarse")
COARSENED_CLASS_COUNT = 24
PSD_TOTAL_TIME_S = 900.0
HYDRAULIC_MASS_PERCENTILE = 90.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_config(ROOT / "configs" / "default_v60.json")
    base_config = replace(
        base_config,
        recipe=replace(
            base_config.recipe,
            total_time_s=max(base_config.recipe.total_time_s, PSD_TOTAL_TIME_S),
        ),
    )
    scenarios = tuple(load_psd_scenario(name) for name in PSD_CLASSES)
    config = replace(base_config, scenarios=scenarios)
    write_class_csv(scenarios)

    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        derived = derive_scenario(config, scenario)
        hydraulic_diameter_um = mass_percentile_diameter_um(scenario, HYDRAULIC_MASS_PERCENTILE)
        permeability_multiplier = (
            hydraulic_diameter_um / (derived.characteristic_diameter_m * 1e6)
        ) ** 2
        run_config = replace(
            config,
            hydraulics=replace(
                config.hydraulics,
                permeability_scale=config.hydraulics.permeability_scale * permeability_multiplier,
            ),
        )
        run_derived = derive_scenario(run_config, scenario)
        result = run_simulation(run_config, scenario.name)
        row = {
            "scenario": scenario.name,
            "class_count": len(scenario.particle_classes),
            "sauter_diameter_um": derived.characteristic_diameter_m * 1e6,
            "hydraulic_diameter_um": hydraulic_diameter_um,
            "hydraulic_mass_percentile": HYDRAULIC_MASS_PERCENTILE,
            "permeability_scale_multiplier": permeability_multiplier,
            "porosity": run_derived.porosity,
            "permeability_m2": run_derived.permeability_m2,
            "cup_water_mass_g": result.summary["cup_mass_g"],
            "cup_dissolved_solids_g": result.summary["cup_dissolved_solids_g"],
            "tds_percent": result.summary["tds_percent"],
            "extraction_yield_percent": result.summary["ey_percent"],
            "retained_water_g": result.summary["bed_water_g"],
            "pooled_water_g": result.summary["pool_water_g"],
            "drawdown_time_s": result.summary["drawdown_time_s"],
            "max_water_balance_residual_g": result.summary["max_water_residual_abs_g"],
            "max_dissolved_solids_balance_residual_g": result.summary[
                "max_solids_residual_abs_g"
            ],
        }
        rows.append(row)
        print(
            f"{scenario.name}: d_sauter={row['sauter_diameter_um']:.2f} um, "
            f"d_h={row['hydraulic_diameter_um']:.2f} um, "
            f"drawdown={row['drawdown_time_s']} s, "
            f"TDS={row['tds_percent']:.3f} %, EY={row['extraction_yield_percent']:.2f} %",
            flush=True,
        )

    write_csv(SIMULATION_CSV, rows)
    write_summary(rows)
    print(f"Wrote {CLASSES_CSV}")
    print(f"Wrote {SIMULATION_CSV}")
    print(f"Wrote {SUMMARY_MD}")


def load_psd_scenario(name: str) -> Scenario:
    rows = []
    with PSD_FILE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            size_um = float(row["size"])
            surface_fraction = float(row[name])
            if size_um <= 0.0 or surface_fraction <= 0.0:
                continue
            rows.append((size_um, surface_fraction))
    if not rows:
        raise ValueError(f"No positive PSD bins found for {name}.")

    mass_weights = [(size_um, surface_fraction * size_um) for size_um, surface_fraction in rows]
    total_mass_weight = sum(weight for _, weight in mass_weights)
    mass_bins = [(size_um, weight / total_mass_weight) for size_um, weight in mass_weights]
    return Scenario(name=name, particle_classes=coarsen_mass_psd(mass_bins, COARSENED_CLASS_COUNT))


def coarsen_mass_psd(
    mass_bins: list[tuple[float, float]],
    target_count: int,
) -> tuple[ParticleClass, ...]:
    """Aggregate a mass-fraction PSD into ordered equal-mass classes."""

    classes: list[ParticleClass] = []
    current_mass = 0.0
    current_diameter_mass = 0.0
    target_mass = 1.0 / target_count
    remaining_groups = target_count

    for size_um, mass_fraction in mass_bins:
        remaining = mass_fraction
        while remaining > 1e-15:
            room = target_mass - current_mass
            take = min(remaining, room)
            current_mass += take
            current_diameter_mass += take * size_um
            remaining -= take
            if current_mass >= target_mass - 1e-14 and remaining_groups > 1:
                classes.append(
                    ParticleClass(
                        radius_um=0.5 * current_diameter_mass / current_mass,
                        mass_fraction=current_mass,
                    )
                )
                remaining_groups -= 1
                current_mass = 0.0
                current_diameter_mass = 0.0
    if current_mass > 1e-14:
        classes.append(
            ParticleClass(
                radius_um=0.5 * current_diameter_mass / current_mass,
                mass_fraction=current_mass,
            )
        )

    total = sum(p.mass_fraction for p in classes)
    normalized = tuple(
        ParticleClass(radius_um=p.radius_um, mass_fraction=p.mass_fraction / total) for p in classes
    )
    return normalized


def mass_percentile_diameter_um(scenario: Scenario, percentile: float) -> float:
    target = percentile / 100.0
    cumulative = 0.0
    for particle in sorted(scenario.particle_classes, key=lambda item: item.radius_um):
        cumulative += particle.mass_fraction
        if cumulative >= target:
            return 2.0 * particle.radius_um
    return 2.0 * max(item.radius_um for item in scenario.particle_classes)


def write_class_csv(scenarios: tuple[Scenario, ...]) -> None:
    rows = []
    for scenario in scenarios:
        for index, particle in enumerate(scenario.particle_classes, start=1):
            rows.append(
                {
                    "scenario": scenario.name,
                    "class_index": index,
                    "diameter_um": 2.0 * particle.radius_um,
                    "radius_um": particle.radius_um,
                    "mass_fraction": particle.mass_fraction,
                }
            )
    write_csv(CLASSES_CSV, rows)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Measured PSD Simulation Summary",
        "",
        f"- Input PSD file: `data/PSD.csv`",
        f"- PSD classes: {', '.join(PSD_CLASSES)}",
        f"- Surface-area bins were converted to mass fractions using mass proportional to surface area times particle diameter.",
        f"- Coarsened classes per PSD: {COARSENED_CLASS_COUNT}",
        f"- Hydraulic effective diameter: mass D{HYDRAULIC_MASS_PERCENTILE:.0f} from the converted mass-fraction PSD",
        f"- Simulation time limit: {PSD_TOTAL_TIME_S:.0f} s",
        "",
        "| scenario | d_sauter (um) | d_hydraulic (um) | porosity | permeability (m2) | drawdown (s) | cup water (g) | TDS (%) | EY (%) | retained water (g) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {ds:.2f} | {dh:.2f} | {eps:.3f} | {perm:.3e} | {drawdown:.1f} | "
            "{cup:.2f} | {tds:.3f} | {ey:.2f} | {retained:.2f} |".format(
                scenario=row["scenario"],
                ds=float(row["sauter_diameter_um"]),
                dh=float(row["hydraulic_diameter_um"]),
                eps=float(row["porosity"]),
                perm=float(row["permeability_m2"]),
                drawdown=float(row["drawdown_time_s"]),
                cup=float(row["cup_water_mass_g"]),
                tds=float(row["tds_percent"]),
                ey=float(row["extraction_yield_percent"]),
                retained=float(row["retained_water_g"]),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
