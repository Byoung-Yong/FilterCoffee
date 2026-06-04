"""Interpolate measured PSD shapes to target mass-fraction D90 values.

The input PSD file contains surface-area fractions on a common size grid.
This script linearly interpolates complete surface-area PSD shapes between the
nearest measured PSD anchors and solves for the interpolation weight that gives
the requested mass-fraction D90 after surface-to-mass conversion.

The generated PSDs are interpolated scenarios, not new measured data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PSD_FILE = ROOT / "data" / "PSD.csv"
OUTPUT_DIR = ROOT / "outputs" / "jfe_revision" / "interpolated_psd"
SURFACE_CSV = OUTPUT_DIR / "interpolated_psd_surface_fraction.csv"
MASS_CSV = OUTPUT_DIR / "interpolated_psd_mass_fraction.csv"
SUMMARY_MD = OUTPUT_DIR / "interpolated_psd_summary.md"
FIGURE_PNG = OUTPUT_DIR / "figure_interpolated_psd_d90.png"
FIGURE_PDF = OUTPUT_DIR / "figure_interpolated_psd_d90.pdf"

TARGET_D90 = (900.0, 1300.0)
PSD_ORDER = ("fine", "medium", "coarse")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    size_um, surface = load_surface_psd()
    anchors = {name: d90_from_surface(size_um, surface[name]) for name in PSD_ORDER}

    interpolated_surface = {}
    interpolation_info = []
    for target in TARGET_D90:
        lower, upper = bracketing_anchors(target, anchors)
        weight = solve_weight_for_target_d90(size_um, surface[lower], surface[upper], target)
        psd = normalize((1.0 - weight) * surface[lower] + weight * surface[upper])
        actual_d90 = d90_from_surface(size_um, psd)
        name = f"D90_{int(target)}um"
        interpolated_surface[name] = psd
        interpolation_info.append(
            {
                "target_D90_um": target,
                "actual_D90_um": actual_d90,
                "lower_anchor": lower,
                "upper_anchor": upper,
                "upper_anchor_weight": weight,
            }
        )

    surface_out = pd.DataFrame({"size_um": size_um})
    mass_out = pd.DataFrame({"size_um": size_um})
    for name in PSD_ORDER:
        surface_out[name] = surface[name]
        mass_out[name] = mass_fraction_from_surface(size_um, surface[name])
    for name, psd in interpolated_surface.items():
        surface_out[name] = psd
        mass_out[name] = mass_fraction_from_surface(size_um, psd)

    surface_out.to_csv(SURFACE_CSV, index=False)
    mass_out.to_csv(MASS_CSV, index=False)
    SUMMARY_MD.write_text(summary_markdown(anchors, interpolation_info), encoding="utf-8")
    make_figure(size_um, surface, interpolated_surface, interpolation_info)

    print("Generated files:")
    for path in (SURFACE_CSV, MASS_CSV, SUMMARY_MD, FIGURE_PNG, FIGURE_PDF):
        print(path)


def load_surface_psd() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(PSD_FILE)
    size_um = df["size"].astype(float).to_numpy()
    surface = {}
    for name in PSD_ORDER:
        values = df[name].astype(float).clip(lower=0).to_numpy()
        surface[name] = normalize(values)
    return size_um, surface


def normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("PSD has no positive weight.")
    return values / total


def mass_fraction_from_surface(size_um: np.ndarray, surface_fraction: np.ndarray) -> np.ndarray:
    mass_weight = surface_fraction * size_um
    return normalize(mass_weight)


def d90_from_surface(size_um: np.ndarray, surface_fraction: np.ndarray) -> float:
    mass_fraction = mass_fraction_from_surface(size_um, surface_fraction)
    cumulative = np.cumsum(mass_fraction)
    return float(np.interp(0.9, cumulative, size_um))


def bracketing_anchors(target_d90: float, anchors: dict[str, float]) -> tuple[str, str]:
    ordered = sorted(anchors.items(), key=lambda item: item[1])
    for (lower_name, lower_d90), (upper_name, upper_d90) in zip(ordered[:-1], ordered[1:]):
        if lower_d90 <= target_d90 <= upper_d90:
            return lower_name, upper_name
    raise ValueError(
        f"Target D90 {target_d90:.1f} um is outside measured anchor range "
        f"{ordered[0][1]:.1f}-{ordered[-1][1]:.1f} um."
    )


def solve_weight_for_target_d90(
    size_um: np.ndarray,
    lower_surface: np.ndarray,
    upper_surface: np.ndarray,
    target_d90: float,
) -> float:
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        candidate = normalize((1.0 - mid) * lower_surface + mid * upper_surface)
        d90 = d90_from_surface(size_um, candidate)
        if d90 < target_d90:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_figure(
    size_um: np.ndarray,
    surface: dict[str, np.ndarray],
    interpolated_surface: dict[str, np.ndarray],
    interpolation_info: list[dict[str, object]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)

    colors = {
        "fine": "#0072B2",
        "medium": "#009E73",
        "coarse": "#D55E00",
        "D90_900um": "#CC79A7",
        "D90_1300um": "#E69F00",
    }
    linestyles = {
        "fine": "-",
        "medium": "-",
        "coarse": "-",
        "D90_900um": "--",
        "D90_1300um": "--",
    }

    for name in PSD_ORDER:
        axes[0].plot(
            size_um,
            surface[name],
            color=colors[name],
            linestyle=linestyles[name],
            label=name,
        )
        axes[1].plot(
            size_um,
            mass_fraction_from_surface(size_um, surface[name]),
            color=colors[name],
            linestyle=linestyles[name],
            label=name,
        )
    for name, psd in interpolated_surface.items():
        label = name.replace("_", " ")
        axes[0].plot(size_um, psd, color=colors[name], linestyle=linestyles[name], label=label)
        axes[1].plot(
            size_um,
            mass_fraction_from_surface(size_um, psd),
            color=colors[name],
            linestyle=linestyles[name],
            label=label,
        )

    axes[0].set_ylabel("Surface-area fraction")
    axes[1].set_ylabel("Mass fraction")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Particle diameter (um)")
        ax.tick_params(direction="out", length=3.5, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].text(-0.16, 1.08, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=12)
    axes[1].text(-0.16, 1.08, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=12)
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(FIGURE_PNG, dpi=300)
    fig.savefig(FIGURE_PDF)
    plt.close(fig)


def summary_markdown(
    anchors: dict[str, float],
    interpolation_info: list[dict[str, object]],
) -> str:
    lines = [
        "# D90-interpolated PSD scenarios",
        "",
        "These PSDs are interpolated scenarios, not measured PSDs.",
        "The interpolation is performed on the complete surface-area PSD shape on the common size grid in `data/PSD.csv`.",
        "After interpolation, the surface-area PSD is converted to mass fraction using mass weight proportional to surface-area fraction times particle diameter.",
        "",
        "## Measured anchor D90 values",
        "",
        "| PSD anchor | mass-fraction D90 (um) |",
        "|---|---:|",
    ]
    for name in PSD_ORDER:
        lines.append(f"| {name} | {anchors[name]:.2f} |")
    lines.extend(
        [
            "",
            "## Interpolated PSDs",
            "",
            "| target D90 (um) | actual D90 (um) | interpolation interval | upper-anchor weight |",
            "|---:|---:|---|---:|",
        ]
    )
    for item in interpolation_info:
        lines.append(
            "| {target_D90_um:.1f} | {actual_D90_um:.2f} | {lower_anchor}-{upper_anchor} | {upper_anchor_weight:.4f} |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
