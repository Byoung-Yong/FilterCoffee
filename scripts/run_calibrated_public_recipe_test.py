"""Run the calibrated V60 model against the unified public recipe test set.

The in-house matched experiment is the calibration set. The rows in
``data/public_data.csv`` are treated as one held-out recipe-robustness test set.
Recipe-specific dose, water, bloom, finish time, and available composition
values are used as observed inputs/outputs; no recipe-specific coefficient
fitting is performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrated_v60_model import (  # noqa: E402
    CALIBRATED_COEFFICIENTS,
    calibrated_config_for_scenario,
    calibrated_scenario_from_grind_description,
    load_calibrated_psd_scenarios,
)
from v60_physics.parameters import ModelConfig, PourSegment, Recipe, Scenario, load_config  # noqa: E402
from v60_physics.solver import derive_scenario, run_simulation  # noqa: E402


DEFAULT_MAIN_START_S = 30.0
DEFAULT_MAIN_PAUSE_S = 10.0
DEFAULT_BLOOM_DURATION_S = 45.0
DEFAULT_MAIN_FLOW_G_S = 3.0
LOCAL_FINISH_ABS_TOLERANCE_S = 45.0
LOCAL_FINISH_REL_TOLERANCE = 0.25

PUBLIC_DATA_FILE = ROOT / "data" / "public_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "public_recipe_test_calibrated"
RAW_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_raw.csv"
GROUPED_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_grouped.csv"
SCHEDULE_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_schedule_quality.csv"
ANCHOR_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_anchors.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_summary.md"
SKIPPED_OUTPUT = OUTPUT_DIR / "public_recipe_test_calibrated_skipped.csv"


def main() -> None:
    args = parse_args()
    base_config = load_config(ROOT / "configs" / "default_v60.json")
    scenarios = load_calibrated_psd_scenarios()
    base_config = replace(
        base_config,
        geometry=replace(
            base_config.geometry,
            axial_layers=args.axial_layers,
            radial_bins=args.radial_bins,
        ),
        recipe=replace(
            base_config.recipe,
            total_time_s=args.minimum_total_time,
            dt_s=args.dt,
            sample_every_s=args.sample_every,
        ),
        scenarios=scenarios,
    )
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    rows = unified_public_recipe_rows()
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    finish_values = [
        value
        for row in rows
        for value in [_observed_finish_s(row)]
        if value is not None and math.isfinite(value)
    ]
    if not finish_values:
        raise ValueError("No observed finish times found in the unified public recipe test set.")
    public_finish_min_s = min(finish_values)
    public_finish_max_s = max(finish_values)

    results: list[dict[str, float | str | bool]] = []
    skipped: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            config, recipe_notes = config_from_public_recipe(base_config, row)
            scenario_name = calibrated_scenario_from_grind_description(
                row.get("recipe_id", ""),
                row.get("grind_description", ""),
            )
            scenario = scenario_by_name[scenario_name]
            run_config = calibrated_config_for_scenario(config, scenario)
            result = run_simulation(run_config, scenario_name)
            output_row = validation_row(
                row=row,
                config=run_config,
                scenario_name=scenario_name,
                recipe_notes=recipe_notes,
                summary=result.summary,
                public_finish_min_s=public_finish_min_s,
                public_finish_max_s=public_finish_max_s,
            )
            output_row.update(extra_result_fields(row, run_config, scenario))
            results.append(output_row)
            print(
                f"{index:02d}/{len(rows)} {row['recipe_id']} -> {scenario_name}, "
                f"drawdown={output_row['simulated_drawdown_time_s']:.1f} s",
                flush=True,
            )
        except (KeyError, ValueError) as exc:
            skipped.append(
                {
                    "recipe_id": row.get("recipe_id", ""),
                    "reason": str(exc),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RAW_OUTPUT, results)
    write_csv(GROUPED_OUTPUT, grouped_rows(results))
    write_csv(SCHEDULE_OUTPUT, schedule_quality_rows(results))
    write_csv(ANCHOR_OUTPUT, anchor_rows(results))
    write_csv(SKIPPED_OUTPUT, skipped)
    SUMMARY_OUTPUT.write_text(
        summary_text(results, skipped, args, public_finish_min_s, public_finish_max_s),
        encoding="utf-8",
    )
    print(f"Wrote {RAW_OUTPUT}")
    print(f"Wrote {GROUPED_OUTPUT}")
    print(f"Wrote {SCHEDULE_OUTPUT}")
    print(f"Wrote {ANCHOR_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axial-layers", type=int, default=24)
    parser.add_argument("--radial-bins", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sample-every", type=float, default=1.0)
    parser.add_argument("--minimum-total-time", type=float, default=500.0)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def unified_public_recipe_rows() -> list[dict[str, str]]:
    return [
        normalize_public_data_row(row)
        for row in read_csv(PUBLIC_DATA_FILE)
        if row.get("recipe_id", "") and truthy(row.get("has_reported_finish_time", ""))
    ]


def normalize_public_data_row(row: dict[str, str]) -> dict[str, str]:
    finish_s = row.get("reported_finish_time_s", "") or row.get("reported_total_brew_time_s", "")
    return {
        "recipe_id": row.get("recipe_id", ""),
        "source_name": row.get("recipe_name", ""),
        "person_or_brand": row.get("source_person_or_brand", ""),
        "source_type": row.get("source_type", ""),
        "evidence_level": row.get("source_grade", ""),
        "brewer": row.get("brewer", ""),
        "dose_g": row.get("coffee_dose_g", ""),
        "water_g": row.get("total_water_g", ""),
        "brew_ratio_water_per_coffee": row.get("brew_ratio_water_per_coffee", ""),
        "water_temp_C": row.get("water_temperature_C", ""),
        "grind_description": row.get("grind_class_for_model", "") or row.get("grind_description_raw", ""),
        "bloom_water_g": row.get("bloom_water_g", ""),
        "bloom_duration_s": row.get("bloom_time_s", ""),
        "pour_schedule": row.get("pour_schedule_raw", ""),
        "estimated_main_pour_flow_g_s": row.get("main_pour_flow_g_s", ""),
        "total_brew_time_s": finish_s,
        "drawdown_or_finish_s": finish_s,
        "agitation_or_bed_action": row.get("agitation_or_bed_action", ""),
        "pour_style": row.get("method_or_pour_style", ""),
        "physics_notes": row.get("data_quality_notes", ""),
        "source_url": row.get("source_url", ""),
        "reported_tds_percent": row.get("reported_tds_percent", ""),
        "reported_ey_percent": row.get("reported_extraction_yield_percent", ""),
        "reported_beverage_mass_g": row.get("beverage_mass_for_comparison_g", "")
        or row.get("reported_beverage_mass_g", ""),
    }


def config_from_public_recipe(
    base_config: ModelConfig,
    row: dict[str, str],
) -> tuple[ModelConfig, str]:
    """Create a model configuration without using finish time as a schedule input."""

    dose_g = required_float(row, "dose_g")
    water_g = required_float(row, "water_g")
    if dose_g <= 0.0 or water_g <= 0.0:
        raise ValueError("dose_g and water_g must be positive")

    bloom_water_g = float_or_none(row.get("bloom_water_g", ""))
    bloom_duration_s = float_or_none(row.get("bloom_duration_s", ""))
    main_flow_g_s = float_or_none(row.get("estimated_main_pour_flow_g_s", ""))

    if bloom_water_g is None:
        bloom_water_g = min(3.0 * dose_g, 0.20 * water_g)
        schedule_quality = "default_flow_estimated_bloom"
    else:
        schedule_quality = "default_flow_reported_bloom"
    if bloom_duration_s is None:
        bloom_duration_s = DEFAULT_BLOOM_DURATION_S
        schedule_quality += "_default_bloom_duration"

    bloom_water_g = min(max(bloom_water_g, 0.0), water_g)
    main_water_g = water_g - bloom_water_g
    if main_water_g <= 0.0:
        raise ValueError("nonpositive main water after bloom allocation")

    bloom = PourSegment(start_s=0.0, end_s=max(bloom_duration_s, 1e-6), water_g=bloom_water_g)
    main_start_s = max(DEFAULT_MAIN_START_S, bloom.end_s)

    if main_flow_g_s is not None and main_flow_g_s > 0.0:
        total_main_duration_s = main_water_g / main_flow_g_s
        schedule_quality = schedule_quality.replace("default_flow", "direct_schedule")
    else:
        total_main_duration_s = main_water_g / DEFAULT_MAIN_FLOW_G_S

    half_main_water = main_water_g / 2.0
    first_duration_s = max(total_main_duration_s / 2.0, 1e-6)
    second_duration_s = max(total_main_duration_s / 2.0, 1e-6)
    main1 = PourSegment(
        start_s=main_start_s,
        end_s=main_start_s + first_duration_s,
        water_g=half_main_water,
    )
    main2 = PourSegment(
        start_s=main1.end_s + DEFAULT_MAIN_PAUSE_S,
        end_s=main1.end_s + DEFAULT_MAIN_PAUSE_S + second_duration_s,
        water_g=half_main_water,
    )

    total_time_s = max(base_config.recipe.total_time_s, main2.end_s + 240.0)
    recipe = Recipe(
        coffee_mass_g=dose_g,
        total_time_s=total_time_s,
        dt_s=base_config.recipe.dt_s,
        sample_every_s=base_config.recipe.sample_every_s,
        pours=(bloom, main1, main2),
    )
    return replace(base_config, recipe=recipe), schedule_quality


def extra_result_fields(
    input_row: dict[str, str],
    config: ModelConfig,
    scenario: Scenario,
) -> dict[str, float | str | bool]:
    derived = derive_scenario(config, scenario)
    coefficients = CALIBRATED_COEFFICIENTS[scenario.name]
    return {
        "grind_description": input_row.get("grind_description", ""),
        "reported_tds_percent": float_or_nan(input_row.get("reported_tds_percent", "")),
        "reported_ey_percent": float_or_nan(input_row.get("reported_ey_percent", "")),
        "reported_beverage_mass_g": float_or_nan(input_row.get("reported_beverage_mass_g", "")),
        "calibrated_retained_capacity_g_per_g": coefficients[
            "retained_water_capacity_g_per_g_coffee"
        ],
        "calibrated_diffusion_rate_ref_s_inv": coefficients["diffusion_rate_ref_s_inv"],
        "calibrated_surface_rate_ref_s_inv": coefficients["surface_rate_ref_s_inv"],
        "calibrated_hydraulic_correction_multiplier": coefficients[
            "hydraulic_correction_multiplier"
        ],
        "calibrated_permeability_m2": derived.permeability_m2,
        "calibrated_characteristic_diameter_um": derived.characteristic_diameter_m * 1e6,
        "unknown_grind_used_medium": scenario.name == "medium"
        and not input_row.get("grind_description", "").strip(),
    }


def validation_row(
    row: dict[str, str],
    config: ModelConfig,
    scenario_name: str,
    recipe_notes: str,
    summary: dict[str, float | str],
    public_finish_min_s: float,
    public_finish_max_s: float,
) -> dict[str, float | str | bool]:
    observed_finish_s = observed_finish(row)
    observed_finish_available = observed_finish_s is not None and math.isfinite(observed_finish_s)
    simulated_drawdown_s = float_value(summary["drawdown_time_s"])
    finish_error_s = (
        simulated_drawdown_s - observed_finish_s
        if observed_finish_available and math.isfinite(simulated_drawdown_s)
        else float("nan")
    )
    finish_abs_error_s = abs(finish_error_s) if math.isfinite(finish_error_s) else float("nan")
    finish_error_percent = (
        100.0 * finish_error_s / observed_finish_s
        if observed_finish_s is not None and observed_finish_s > 0.0 and math.isfinite(finish_error_s)
        else float("nan")
    )
    local_window_s = (
        max(LOCAL_FINISH_ABS_TOLERANCE_S, LOCAL_FINISH_REL_TOLERANCE * observed_finish_s)
        if observed_finish_s is not None
        else float("nan")
    )
    finite_drawdown = math.isfinite(simulated_drawdown_s)
    inside_public_finish_range = (
        finite_drawdown and public_finish_min_s <= simulated_drawdown_s <= public_finish_max_s
    )
    inside_local_finish_window = (
        observed_finish_available
        and math.isfinite(finish_abs_error_s)
        and finish_abs_error_s <= local_window_s
    )
    total_input_water_g = float_value(summary["total_input_water_g"])
    observed_water_g = required_float(row, "water_g")
    cup_mass_g = float_value(summary["cup_mass_g"])
    non_cup_water_g = observed_water_g - cup_mass_g
    liquid_recovery_percent = 100.0 * cup_mass_g / observed_water_g if observed_water_g > 0.0 else float("nan")

    return {
        "recipe_id": row["recipe_id"],
        "source_name": row["source_name"],
        "person_or_brand": row["person_or_brand"],
        "source_type": row["source_type"],
        "evidence_level": row["evidence_level"],
        "scenario_used": scenario_name,
        "schedule_quality": recipe_notes,
        "observed_dose_g": required_float(row, "dose_g"),
        "observed_water_g": observed_water_g,
        "observed_brew_ratio_water_per_coffee": float_or_nan(row.get("brew_ratio_water_per_coffee", "")),
        "observed_bloom_water_g": float_or_nan(row.get("bloom_water_g", "")),
        "observed_bloom_duration_s": float_or_nan(row.get("bloom_duration_s", "")),
        "observed_main_flow_g_s": float_or_nan(row.get("estimated_main_pour_flow_g_s", "")),
        "observed_finish_s": observed_finish_s if observed_finish_s is not None else float("nan"),
        "observed_finish_available": observed_finish_available,
        "simulated_total_input_water_g": total_input_water_g,
        "input_water_error_g": total_input_water_g - observed_water_g,
        "simulated_cup_mass_g": cup_mass_g,
        "simulated_liquid_recovery_percent": liquid_recovery_percent,
        "simulated_non_cup_water_g": non_cup_water_g,
        "simulated_bed_water_g": float_value(summary["bed_water_g"]),
        "simulated_pool_water_g": float_value(summary["pool_water_g"]),
        "simulated_max_pool_water_g": float_value(summary["max_pool_water_g"]),
        "simulated_tds_percent": float_value(summary["tds_percent"]),
        "simulated_ey_percent": float_value(summary["ey_percent"]),
        "simulated_released_solids_percent": float_value(summary["released_solids_percent"]),
        "simulated_drawdown_time_s": simulated_drawdown_s,
        "finish_error_s": finish_error_s,
        "finish_abs_error_s": finish_abs_error_s,
        "finish_error_percent": finish_error_percent,
        "local_finish_window_s": local_window_s,
        "finite_drawdown": finite_drawdown,
        "inside_public_finish_range": inside_public_finish_range,
        "inside_local_finish_window": inside_local_finish_window,
        "water_balance_residual_g": float_value(summary["max_water_residual_abs_g"]),
        "solids_balance_residual_g": float_value(summary["max_solids_residual_abs_g"]),
        "cup_mass_within_input": 0.0 <= cup_mass_g <= observed_water_g,
        "source_url": row["source_url"],
    }


def grouped_rows(rows: list[dict[str, float | str | bool]]) -> list[dict[str, float | str]]:
    groups: dict[str, list[dict[str, float | str | bool]]] = {}
    for row in rows:
        key = str(row["scenario_used"])
        groups.setdefault(key, []).append(row)

    output = []
    for scenario, group in sorted(groups.items()):
        observed = [row for row in group if bool(row["observed_finish_available"])]
        finish_errors = [
            float(row["finish_abs_error_s"])
            for row in observed
            if math.isfinite(float(row["finish_abs_error_s"]))
        ]
        output.append(
            {
                "scenario_used": scenario,
                "n": len(group),
                "n_with_observed_finish": len(observed),
                "finite_drawdown_percent": percent_true(group, "finite_drawdown"),
                "inside_local_finish_window_percent_of_observed": percent_true_where(
                    group,
                    "inside_local_finish_window",
                    "observed_finish_available",
                ),
                "median_abs_finish_error_s": (
                    statistics.median(finish_errors) if finish_errors else float("nan")
                ),
                "mean_abs_finish_error_s": (
                    statistics.mean(finish_errors) if finish_errors else float("nan")
                ),
                "mean_simulated_tds_percent": statistics.mean(
                    float(row["simulated_tds_percent"]) for row in group
                ),
                "mean_simulated_ey_percent": statistics.mean(
                    float(row["simulated_ey_percent"]) for row in group
                ),
            }
        )
    return output


def schedule_quality_rows(rows: list[dict[str, float | str | bool]]) -> list[dict[str, float | str]]:
    groups: dict[str, list[dict[str, float | str | bool]]] = {}
    for row in rows:
        key = schedule_family(str(row["schedule_quality"]))
        groups.setdefault(key, []).append(row)

    output = []
    for quality, group in sorted(groups.items()):
        observed = [row for row in group if bool(row["observed_finish_available"])]
        finish_errors = [
            float(row["finish_abs_error_s"])
            for row in observed
            if math.isfinite(float(row["finish_abs_error_s"]))
        ]
        output.append(
            {
                "schedule_quality": quality,
                "n": len(group),
                "n_with_observed_finish": len(observed),
                "inside_local_finish_window_percent_of_observed": percent_true_where(
                    group,
                    "inside_local_finish_window",
                    "observed_finish_available",
                ),
                "median_abs_finish_error_s": (
                    statistics.median(finish_errors) if finish_errors else float("nan")
                ),
                "mean_abs_finish_error_s": (
                    statistics.mean(finish_errors) if finish_errors else float("nan")
                ),
            }
        )
    return output


def anchor_rows(rows: list[dict[str, float | str | bool]]) -> list[dict[str, float | str]]:
    output = []
    for row in rows:
        reported_tds = float_or_nan(row.get("reported_tds_percent", ""))
        reported_ey = float_or_nan(row.get("reported_ey_percent", ""))
        reported_beverage = float_or_nan(row.get("reported_beverage_mass_g", ""))
        if not any(math.isfinite(value) for value in (reported_tds, reported_ey, reported_beverage)):
            continue
        output.append(
            {
                "recipe_id": row["recipe_id"],
                "source_name": row["source_name"],
                "scenario_used": row["scenario_used"],
                "observed_dose_g": row["observed_dose_g"],
                "observed_water_g": row["observed_water_g"],
                "reported_tds_percent": reported_tds,
                "simulated_tds_percent": row["simulated_tds_percent"],
                "tds_error_percent_points": diff_or_nan(row["simulated_tds_percent"], reported_tds),
                "reported_ey_percent": reported_ey,
                "simulated_ey_percent": row["simulated_ey_percent"],
                "ey_error_percent_points": diff_or_nan(row["simulated_ey_percent"], reported_ey),
                "reported_beverage_mass_g": reported_beverage,
                "simulated_cup_mass_g": row["simulated_cup_mass_g"],
                "cup_mass_error_g": diff_or_nan(row["simulated_cup_mass_g"], reported_beverage),
                "source_url": row["source_url"],
            }
        )
    return output


def summary_text(
    rows: list[dict[str, float | str | bool]],
    skipped: list[dict[str, str]],
    args: argparse.Namespace,
    public_finish_min_s: float,
    public_finish_max_s: float,
) -> str:
    observed = [row for row in rows if bool(row["observed_finish_available"])]
    local_pass = [row for row in observed if bool(row["inside_local_finish_window"])]
    anchors = anchor_rows(rows)
    finish_errors = [
        float(row["finish_abs_error_s"])
        for row in observed
        if math.isfinite(float(row["finish_abs_error_s"]))
    ]
    max_water = max(float(row["water_balance_residual_g"]) for row in rows) if rows else float("nan")
    max_solids = max(float(row["solids_balance_residual_g"]) for row in rows) if rows else float("nan")
    unknown_grind = sum(1 for row in rows if bool(row["unknown_grind_used_medium"]))
    scenario_counts = {
        scenario: sum(1 for row in rows if row["scenario_used"] == scenario)
        for scenario in sorted({str(row["scenario_used"]) for row in rows})
    }
    schedule_groups = schedule_quality_rows(rows)

    lines = [
        "# Calibrated public recipe test",
        "",
        "The matched in-house pour-over experiment is the calibration set. The public recipe CSV is used as one held-out public recipe test set.",
        "Each public recipe is simulated with its reported dose, water, bloom, and available schedule-derived flow. Reported finish time is used only as a test observable.",
        "No recipe-specific coefficient fitting is performed. If particle size is not specified, the medium measured PSD class is used.",
        "",
        "## Calculation",
        "",
        f"- Grid: {args.axial_layers} axial layers x {args.radial_bins} radial bins",
        f"- Time step: {args.dt:.4g} s",
        f"- Sample interval: {args.sample_every:.4g} s",
        f"- Minimum simulated time horizon: {args.minimum_total_time:.1f} s",
        "- Calibrated coefficients: retained-water capacity, permeability correction, diffusion-like release rate, and surface-like release rate.",
        "",
        "## Dataset coverage",
        "",
        f"- Simulated public recipe rows: {len(rows)}",
        f"- Skipped rows: {len(skipped)}",
        f"- Rows with observed finish time: {len(observed)}",
        f"- Rows with reported TDS/EY/beverage anchor values: {len(anchors)}",
        f"- Observed finish range in combined public test set: {public_finish_min_s:.1f}-{public_finish_max_s:.1f} s",
        f"- Rows with unspecified grind mapped to medium: {unknown_grind}",
        "",
        "## Scenario assignment",
        "",
    ]
    for scenario, count in scenario_counts.items():
        lines.append(f"- {scenario}: {count}")

    lines.extend(["", "## Schedule reconstruction", ""])
    for row in schedule_groups:
        lines.append(
            "- "
            f"{row['schedule_quality']}: n={row['n']}, "
            f"inside local window={row['inside_local_finish_window_percent_of_observed']:.1f}%, "
            f"median absolute error={row['median_abs_finish_error_s']:.1f} s, "
            f"mean absolute error={row['mean_abs_finish_error_s']:.1f} s"
        )

    lines.extend(
        [
            "",
            "## Finish-time comparison",
            "",
            f"- Finite drawdown reported: {percent_true(rows, 'finite_drawdown'):.1f}%",
            f"- Inside local recipe finish-time window among rows with observed finish: {100.0 * len(local_pass) / len(observed):.1f}%",
            f"- Cup mass within input water: {percent_true(rows, 'cup_mass_within_input'):.1f}%",
            f"- Maximum water balance residual: {max_water:.3e} g",
            f"- Maximum dissolved-solids balance residual: {max_solids:.3e} g",
        ]
    )
    if finish_errors:
        lines.extend(
            [
                f"- Median absolute finish-time error: {statistics.median(finish_errors):.1f} s",
                f"- Mean absolute finish-time error: {statistics.mean(finish_errors):.1f} s",
                f"- Maximum absolute finish-time error: {max(finish_errors):.1f} s",
            ]
        )

    lines.extend(["", "## Beverage-strength anchors", ""])
    if anchors:
        lines.extend(
            [
                "| recipe_id | reported TDS (%) | simulated TDS (%) | reported EY (%) | simulated EY (%) | reported beverage (g) | simulated cup (g) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in anchors:
            lines.append(
                "| "
                f"{row['recipe_id']} | "
                f"{format_optional(row['reported_tds_percent'])} | "
                f"{format_optional(row['simulated_tds_percent'])} | "
                f"{format_optional(row['reported_ey_percent'])} | "
                f"{format_optional(row['simulated_ey_percent'])} | "
                f"{format_optional(row['reported_beverage_mass_g'])} | "
                f"{format_optional(row['simulated_cup_mass_g'])} |"
            )
    else:
        lines.append("No public recipe test rows reported beverage-strength anchor values.")

    failures = [
        row
        for row in observed
        if not bool(row["inside_local_finish_window"])
    ]
    lines.extend(["", "## Rows outside the local finish-time window", ""])
    if failures:
        lines.extend(
            [
                "| recipe_id | scenario | observed finish (s) | simulated drawdown (s) | error (s) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in failures:
            lines.append(
                "| "
                f"{row['recipe_id']} | "
                f"{row['scenario_used']} | "
                f"{float(row['observed_finish_s']):.1f} | "
                f"{float(row['simulated_drawdown_time_s']):.1f} | "
                f"{float(row['finish_error_s']):.1f} |"
            )
    else:
        lines.append("No rows with observed finish time fell outside the local finish-time window.")

    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- Raw rows: `{RAW_OUTPUT.relative_to(ROOT)}`",
            f"- Grouped rows: `{GROUPED_OUTPUT.relative_to(ROOT)}`",
            f"- Beverage-strength anchors: `{ANCHOR_OUTPUT.relative_to(ROOT)}`",
            f"- Schedule-quality rows: `{SCHEDULE_OUTPUT.relative_to(ROOT)}`",
            f"- Skipped rows: `{SKIPPED_OUTPUT.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _observed_finish_s(row: dict[str, str]) -> float | None:
    return float_or_none(row.get("drawdown_or_finish_s", "")) or float_or_none(
        row.get("total_brew_time_s", "")
    )


def observed_finish(row: dict[str, str]) -> float | None:
    return _observed_finish_s(row)


def float_value(value: object) -> float:
    numeric = float_or_nan(value)
    if not math.isfinite(numeric):
        return float("nan")
    return numeric


def float_or_none(value: str | object | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    return float(text)


def required_float(row: dict[str, str], key: str) -> float:
    value = float_or_none(row.get(key, ""))
    if value is None or not math.isfinite(value):
        raise ValueError(f"missing required numeric value: {key}")
    return value


def truthy(value: str | object | None) -> bool:
    return str(value).strip().upper() == "TRUE"


def schedule_family(value: str) -> str:
    if value.startswith("direct_schedule"):
        return "direct_schedule"
    if value.startswith("default_flow"):
        return "default_flow"
    return value


def float_or_nan(value: str | object | None) -> float:
    try:
        parsed = float_or_none(value)
    except ValueError:
        return float("nan")
    return parsed if parsed is not None else float("nan")


def diff_or_nan(left: object, right: object, reverse: bool = False) -> float:
    left_value = float_or_nan(left)
    right_value = float_or_nan(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return right_value - left_value if reverse else left_value - right_value


def percent_true(rows: Iterable[dict[str, object]], key: str) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for row in rows if bool(row[key])) / len(rows)


def percent_true_where(
    rows: Iterable[dict[str, object]],
    key: str,
    availability_key: str,
) -> float:
    available = [row for row in rows if bool(row[availability_key])]
    if not available:
        return float("nan")
    return 100.0 * sum(1 for row in available if bool(row[key])) / len(available)


def format_optional(value: object) -> str:
    numeric = float_or_nan(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.3f}"


if __name__ == "__main__":
    main()
