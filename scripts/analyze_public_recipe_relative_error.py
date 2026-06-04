"""Analyze public recipe reconstruction with relative finish-time error only."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = (
    ROOT / "outputs" / "jfe_revision" / "d90_closure" / "d90_closure_public_recipe_reconstruction.csv",
    ROOT / "outputs" / "jfe_revision" / "public_recipe_reconstruction_diagnostics.csv",
)
OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "public_recipe_relative_error"

DIAGNOSTICS_CSV = OUTPUT_DIR / "public_recipe_reconstruction_relative_error_diagnostics.csv"
THRESHOLD_CSV = OUTPUT_DIR / "finish_time_relative_threshold_summary.csv"
THRESHOLD_MD = OUTPUT_DIR / "finish_time_relative_threshold_summary.md"
CDF_CSV = OUTPUT_DIR / "finish_time_relative_error_cdf.csv"
SUMMARY_MD = OUTPUT_DIR / "public_recipe_relative_error_summary.md"
MAIN_FIG = OUTPUT_DIR / "figure_public_recipe_relative_error.png"
CDF_FIG = OUTPUT_DIR / "figure_finish_time_relative_error_cdf.png"

THRESHOLDS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
KEY_THRESHOLDS = (10, 15, 20, 25, 30, 40)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_file = select_input_file()
    source = pd.read_csv(source_file)
    diagnostics = build_diagnostics(source)
    threshold = threshold_summary(diagnostics)
    cdf = cdf_table(diagnostics)

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    threshold.to_csv(THRESHOLD_CSV, index=False)
    cdf.to_csv(CDF_CSV, index=False)
    THRESHOLD_MD.write_text(threshold_markdown(threshold), encoding="utf-8")
    SUMMARY_MD.write_text(summary_markdown(diagnostics, threshold, source_file), encoding="utf-8")
    make_main_figure(diagnostics, threshold)
    make_cdf_figure(diagnostics, cdf)

    for path in (
        DIAGNOSTICS_CSV,
        THRESHOLD_CSV,
        THRESHOLD_MD,
        CDF_CSV,
        SUMMARY_MD,
        MAIN_FIG,
        CDF_FIG,
    ):
        print(path)


def select_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No existing public recipe reconstruction diagnostics CSV found.")


def build_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    reported = pd.to_numeric(source["observed_finish_s"], errors="coerce")
    simulated = pd.to_numeric(source["simulated_drawdown_time_s"], errors="coerce")
    valid = source[(reported > 0) & np.isfinite(reported) & np.isfinite(simulated)].copy()
    reported = pd.to_numeric(valid["observed_finish_s"], errors="coerce")
    simulated = pd.to_numeric(valid["simulated_drawdown_time_s"], errors="coerce")
    error = simulated - reported
    abs_error = error.abs()
    relative_abs = 100.0 * abs_error / reported
    relative_signed = 100.0 * error / reported
    cup_water = pd.to_numeric(valid["simulated_cup_mass_g"], errors="coerce")
    tds = pd.to_numeric(valid["simulated_tds_percent"], errors="coerce")
    dose = pd.to_numeric(valid["observed_dose_g"], errors="coerce")
    retained = pd.to_numeric(valid["simulated_bed_water_g"], errors="coerce")

    diagnostics = pd.DataFrame(
        {
            "recipe_id": valid["recipe_id"],
            "source": valid.get("source_name", ""),
            "dose_g": dose,
            "total_water_g": pd.to_numeric(valid["observed_water_g"], errors="coerce"),
            "brew_ratio_g_g": pd.to_numeric(
                valid.get("observed_brew_ratio_water_per_coffee", np.nan),
                errors="coerce",
            ),
            "grind_description": valid.get("grind_description", ""),
            "mapped_PSD_class": valid["scenario_used"],
            "schedule_quality": valid["schedule_quality"],
            "reported_finish_time_s": reported,
            "simulated_drawdown_time_s": simulated,
            "finish_time_error_s": error,
            "absolute_finish_time_error_s": abs_error,
            "relative_abs_finish_error_percent": relative_abs,
            "relative_signed_finish_error_percent": relative_signed,
            "cup_water_mass_g": cup_water,
            "retained_water_g": retained,
            "retained_water_per_coffee_g_g": retained / dose,
            "cup_dissolved_solids_g": cup_water * tds / 100.0,
            "TDS_percent": tds,
            "extraction_yield_percent": pd.to_numeric(valid["simulated_ey_percent"], errors="coerce"),
            "max_water_balance_residual_g": pd.to_numeric(
                valid.get("water_balance_residual_g", valid.get("max_water_balance_residual_g", np.nan)),
                errors="coerce",
            ),
            "max_dissolved_solids_balance_residual_g": pd.to_numeric(
                valid.get(
                    "solids_balance_residual_g",
                    valid.get("max_dissolved_solids_balance_residual_g", np.nan),
                ),
                errors="coerce",
            ),
            "solver_status": np.where(pd.to_numeric(valid["simulated_drawdown_time_s"], errors="coerce").notna(), "ok", "no_drawdown"),
        }
    )
    return diagnostics


def schedule_family(schedule_quality: str) -> str:
    text = str(schedule_quality)
    if text.startswith("direct_schedule"):
        return "direct schedule"
    if text.startswith("default_flow"):
        return "default flow"
    return "other schedule"


def threshold_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all observations", diagnostics)]
    for name, group in diagnostics.groupby("schedule_quality", dropna=False):
        groups.append(("schedule_quality", str(name), group))
    for name, group in diagnostics.groupby("mapped_PSD_class", dropna=False):
        groups.append(("mapped_PSD_class", str(name), group))
    family = diagnostics.assign(schedule_family=diagnostics["schedule_quality"].map(schedule_family))
    for name, group in family.groupby("schedule_family", dropna=False):
        groups.append(("schedule_family", str(name), group))

    for group_type, group_label, group in groups:
        values = pd.to_numeric(group["relative_abs_finish_error_percent"], errors="coerce").dropna()
        n_total = len(values)
        for threshold in THRESHOLDS:
            n_within = int((values <= threshold).sum())
            rows.append(
                {
                    "group_type": group_type,
                    "group_label": group_label,
                    "threshold_percent": threshold,
                    "n_total": n_total,
                    "n_within_threshold": n_within,
                    "fraction_within_threshold_percent": 100.0 * n_within / n_total if n_total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cdf_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    sorted_rows = diagnostics.sort_values("relative_abs_finish_error_percent").reset_index(drop=True)
    n = len(sorted_rows)
    return pd.DataFrame(
        {
            "relative_abs_finish_error_percent_sorted": sorted_rows[
                "relative_abs_finish_error_percent"
            ],
            "cumulative_fraction_percent": 100.0 * (np.arange(n) + 1) / n,
            "recipe_id": sorted_rows["recipe_id"],
            "source": sorted_rows["source"],
            "mapped_PSD_class": sorted_rows["mapped_PSD_class"],
            "schedule_quality": sorted_rows["schedule_quality"],
        }
    )


def metric_summary(group: pd.DataFrame) -> dict[str, float]:
    abs_error = pd.to_numeric(group["absolute_finish_time_error_s"], errors="coerce")
    rel_abs = pd.to_numeric(group["relative_abs_finish_error_percent"], errors="coerce")
    rel_signed = pd.to_numeric(group["relative_signed_finish_error_percent"], errors="coerce")
    return {
        "n": float(len(group)),
        "median_abs_error_s": float(abs_error.median()),
        "mean_abs_error_s": float(abs_error.mean()),
        "max_abs_error_s": float(abs_error.max()),
        "median_relative_abs_error_percent": float(rel_abs.median()),
        "mean_relative_abs_error_percent": float(rel_abs.mean()),
        "median_signed_relative_error_percent": float(rel_signed.median()),
        "mean_signed_relative_error_percent": float(rel_signed.mean()),
    }


def summary_markdown(
    diagnostics: pd.DataFrame,
    threshold: pd.DataFrame,
    source_file: Path,
) -> str:
    overall = metric_summary(diagnostics)
    max_water = diagnostics["max_water_balance_residual_g"].max()
    max_solids = diagnostics["max_dissolved_solids_balance_residual_g"].max()
    lines = [
        "# Public recipe relative finish-time error analysis",
        "",
        f"- Input reconstruction file: `{source_file.relative_to(ROOT)}`",
        "- Public recipes were not used for coefficient fitting.",
        "- Reported finish time was not used to reconstruct the pour schedule.",
        "- Continuous finish-time error is the primary metric.",
        "- Threshold fractions are scale-agreement summaries, not validation pass/fail criteria.",
        "",
        "## Overall metrics",
        "",
        f"- Public observations with positive reported finish time: {int(overall['n'])}",
        f"- Median absolute finish-time error: {overall['median_abs_error_s']:.1f} s",
        f"- Mean absolute finish-time error: {overall['mean_abs_error_s']:.1f} s",
        f"- Maximum absolute finish-time error: {overall['max_abs_error_s']:.1f} s",
        f"- Median relative absolute finish-time error: {overall['median_relative_abs_error_percent']:.1f}%",
        f"- Mean relative absolute finish-time error: {overall['mean_relative_abs_error_percent']:.1f}%",
        f"- Median signed relative finish-time error: {overall['median_signed_relative_error_percent']:.1f}%",
        f"- Mean signed relative finish-time error: {overall['mean_signed_relative_error_percent']:.1f}%",
        "",
        "## Fraction within relative-error thresholds",
        "",
    ]
    all_threshold = threshold[threshold["group_type"] == "all"]
    for value in KEY_THRESHOLDS:
        row = all_threshold[all_threshold["threshold_percent"] == value].iloc[0]
        lines.append(
            f"- Within {value}% relative absolute error: {int(row['n_within_threshold'])}/{int(row['n_total'])} ({row['fraction_within_threshold_percent']:.1f}%)"
        )

    lines.extend(
        [
            "",
            "## Metrics by mapped PSD class",
            "",
            "| mapped PSD class | n | median abs error (s) | mean abs error (s) | median relative abs error (%) | mean signed relative error (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, group in diagnostics.groupby("mapped_PSD_class"):
        metrics = metric_summary(group)
        lines.append(
            "| {name} | {n:.0f} | {med_abs:.1f} | {mean_abs:.1f} | {med_rel:.1f} | {mean_signed:.1f} |".format(
                name=name,
                n=metrics["n"],
                med_abs=metrics["median_abs_error_s"],
                mean_abs=metrics["mean_abs_error_s"],
                med_rel=metrics["median_relative_abs_error_percent"],
                mean_signed=metrics["mean_signed_relative_error_percent"],
            )
        )

    lines.extend(
        [
            "",
            "## Metrics by schedule quality",
            "",
            "| schedule quality | n | median abs error (s) | mean abs error (s) | median relative abs error (%) | mean signed relative error (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, group in diagnostics.groupby("schedule_quality"):
        metrics = metric_summary(group)
        lines.append(
            "| {name} | {n:.0f} | {med_abs:.1f} | {mean_abs:.1f} | {med_rel:.1f} | {mean_signed:.1f} |".format(
                name=name,
                n=metrics["n"],
                med_abs=metrics["median_abs_error_s"],
                mean_abs=metrics["mean_abs_error_s"],
                med_rel=metrics["median_relative_abs_error_percent"],
                mean_signed=metrics["mean_signed_relative_error_percent"],
            )
        )

    lines.extend(
        [
            "",
            "## Balance residuals",
            "",
            f"- Maximum water balance residual: {max_water:.3e} g",
            f"- Maximum dissolved-solids balance residual: {max_solids:.3e} g",
        ]
    )
    return "\n".join(lines) + "\n"


def threshold_markdown(threshold: pd.DataFrame) -> str:
    lines = [
        "# Finish-time relative threshold summary",
        "",
        "Fractions are scale-agreement summaries based on relative absolute finish-time error. They are not validation pass/fail criteria.",
        "",
        "| group type | group | threshold (%) | n total | n within threshold | fraction within threshold (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in threshold.iterrows():
        lines.append(
            "| {gt} | {gl} | {th:.0f} | {n:.0f} | {nw:.0f} | {frac:.1f} |".format(
                gt=row["group_type"],
                gl=row["group_label"],
                th=row["threshold_percent"],
                n=row["n_total"],
                nw=row["n_within_threshold"],
                frac=row["fraction_within_threshold_percent"],
            )
        )
    return "\n".join(lines) + "\n"


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
        }
    )


def make_main_figure(diagnostics: pd.DataFrame, threshold: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
    ax_a, ax_b, ax_c, ax_d, ax_e = axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]
    axes[1, 2].axis("off")

    within25 = diagnostics["relative_abs_finish_error_percent"] <= 25.0
    reported = diagnostics["reported_finish_time_s"]
    simulated = diagnostics["simulated_drawdown_time_s"]
    low, high = min(reported.min(), simulated.min()), max(reported.max(), simulated.max())
    line = np.linspace(low, high, 100)
    ax_a.scatter(
        reported[within25],
        simulated[within25],
        color="#0072B2",
        edgecolor="black",
        linewidth=0.4,
        s=24,
        label="within 25% band",
    )
    ax_a.scatter(
        reported[~within25],
        simulated[~within25],
        color="#D55E00",
        edgecolor="black",
        linewidth=0.4,
        s=24,
        label="outside 25% band",
    )
    ax_a.plot(line, line, color="black", linewidth=1.0)
    ax_a.plot(line, 1.25 * line, color="0.45", linestyle="--", linewidth=0.9)
    ax_a.plot(line, 0.75 * line, color="0.45", linestyle="--", linewidth=0.9)
    ax_a.set_xlabel("Reported finish time (s)")
    ax_a.set_ylabel("Simulated drawdown time (s)")
    ax_a.legend(frameon=False, loc="upper left")
    ax_a.text(0.02, 0.95, "A", transform=ax_a.transAxes, fontweight="bold", va="top")

    class_order = ["coarse", "medium", "fine"]
    boxplot_by_group(ax_b, diagnostics, "mapped_PSD_class", class_order, "absolute_finish_time_error_s")
    ax_b.set_ylabel("Absolute finish-time error (s)")
    ax_b.text(0.02, 0.95, "B", transform=ax_b.transAxes, fontweight="bold", va="top")

    boxplot_by_group(ax_c, diagnostics, "mapped_PSD_class", class_order, "relative_abs_finish_error_percent")
    ax_c.set_ylabel("Relative absolute error (%)")
    ax_c.text(0.02, 0.95, "C", transform=ax_c.transAxes, fontweight="bold", va="top")

    ax_d.scatter(
        diagnostics["reported_finish_time_s"],
        diagnostics["relative_signed_finish_error_percent"],
        c=[class_color(value) for value in diagnostics["mapped_PSD_class"]],
        edgecolor="black",
        linewidth=0.4,
        s=22,
        alpha=0.85,
    )
    ax_d.axhline(0.0, color="black", linewidth=0.8)
    ax_d.set_xlabel("Reported finish time (s)")
    ax_d.set_ylabel("Signed relative error (%)")
    ax_d.text(0.02, 0.95, "D", transform=ax_d.transAxes, fontweight="bold", va="top")

    plot_threshold_curves(ax_e, threshold)
    ax_e.axvline(25, color="0.4", linestyle="--", linewidth=0.9)
    ax_e.set_xlabel("Relative absolute error threshold (%)")
    ax_e.set_ylabel("Fraction within threshold (%)")
    ax_e.text(0.02, 0.95, "E", transform=ax_e.transAxes, fontweight="bold", va="top")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        clean_axis(ax)
    fig.tight_layout()
    fig.savefig(MAIN_FIG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_cdf_figure(diagnostics: pd.DataFrame, cdf: pd.DataFrame) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(
        cdf["relative_abs_finish_error_percent_sorted"],
        cdf["cumulative_fraction_percent"],
        color="black",
        linewidth=1.6,
        label="All observations",
    )
    for family, group in diagnostics.assign(schedule_family=diagnostics["schedule_quality"].map(schedule_family)).groupby("schedule_family"):
        sorted_group = group.sort_values("relative_abs_finish_error_percent")
        n = len(sorted_group)
        ax.plot(
            sorted_group["relative_abs_finish_error_percent"],
            100.0 * (np.arange(n) + 1) / n,
            linewidth=1.0,
            label=family,
            alpha=0.8,
        )
    ax.axvline(25.0, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Relative absolute finish-time error (%)")
    ax.set_ylabel("Cumulative fraction within relative error (%)")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(CDF_FIG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def boxplot_by_group(ax, data: pd.DataFrame, group_key: str, order: list[str], value_key: str) -> None:
    values = [data.loc[data[group_key] == label, value_key].dropna().to_numpy() for label in order]
    ax.boxplot(
        values,
        tick_labels=order,
        patch_artist=True,
        boxprops={"facecolor": "#D0D0D0", "edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    for idx, label in enumerate(order, start=1):
        subset = data.loc[data[group_key] == label, value_key].dropna()
        jitter = np.linspace(-0.08, 0.08, len(subset)) if len(subset) else []
        ax.scatter(
            np.full(len(subset), idx) + jitter,
            subset,
            s=13,
            color=class_color(label),
            edgecolor="black",
            linewidth=0.3,
            alpha=0.75,
        )


def plot_threshold_curves(ax, threshold: pd.DataFrame) -> None:
    all_rows = threshold[threshold["group_type"] == "all"]
    ax.plot(
        all_rows["threshold_percent"],
        all_rows["fraction_within_threshold_percent"],
        color="black",
        marker="o",
        label="All observations",
    )
    for group_name, group in threshold[threshold["group_type"] == "schedule_family"].groupby("group_label"):
        ax.plot(
            group["threshold_percent"],
            group["fraction_within_threshold_percent"],
            marker="o",
            linewidth=0.9,
            label=group_name,
            alpha=0.8,
        )
    ax.legend(frameon=False, loc="lower right")


def class_color(label: str) -> str:
    return {"coarse": "#0072B2", "medium": "#E69F00", "fine": "#009E73"}.get(str(label), "0.5")


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


if __name__ == "__main__":
    sys.exit(main())
