"""
make_fig4_calibrated_partitioning_revised.py

Purpose:
    Generate revised Figure 4 for the JFE manuscript.

Outputs:
    outputs/jfe_revision/calibrated_partitioning/revised_figure4/
        figure4_calibrated_partitioning_revised.png
        figure4_calibrated_partitioning_revised.pdf
        figure4_plotting_data.csv
        figure4_caption.md

Notes:
    This script is fully standalone for figure generation. The endpoint
    calibrated partitioning data are embedded from the existing calibrated
    partitioning CSV. It does not rerun calibration or the simulator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def set_publication_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stixsans",
    })


def load_embedded_data():
    """Return endpoint partitioning data used for revised Figure 4."""
    return pd.DataFrame({
        "PSD_class": ["coarse", "medium", "fine"],
        "D90_um": [1587.1150455829554, 1169.8973149533497, 823.9724741833481],
        "cup_water_g": [262.34330529483634, 257.13875471442293, 249.8052209420701],
        "retained_water_g": [34.8331536333583, 40.35904932402941, 47.80084237910121],
        "mobile_bed_water_g": [2.8235410717916922, 2.5021959615243894, 2.3939366787658893],
        "pooled_water_g": [0.0, 0.0, 0.0],
        "input_water_g": [299.99999999998397, 299.99999999998397, 299.99999999998397],
        "cup_dissolved_solids_g": [2.9400529991780875, 3.933604561609387, 4.100251222308918],
        "bed_liquid_dissolved_solids_g": [0.2894401561294256, 0.25432919448910896, 0.25764807478646273],
        "remaining_extractable_solids_g": [2.7705068446924597, 1.8120662439014787, 1.6421007029045913],
        "initial_extractable_solids_g": [6.000000000000003, 6.000000000000003, 6.000000000000003],
        "drawdown_time_s": [164.0, 221.0, 247.0],
        "TDS_percent": [1.1206891656236049, 1.5297595129050185, 1.641379314189661],
        "extraction_yield_percent": [14.70026499589044, 19.668022808046935, 20.50125611154459],
        "water_balance_residual_g": [-2.3305801732931286e-12, 7.275957614183426e-12, 4.6782133722444996e-11],
        "dissolved_solids_balance_residual_g": [2.930988785010413e-14, 2.7533531010703882e-14, 3.019806626980426e-14],
    })


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_axis(ax):
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def stacked_bar(ax, data, components, colors, labels, hatches, ylabel, ylim=None):
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for component, color, label, hatch in zip(components, colors, labels, hatches):
        values = data[component].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            hatch=hatch,
            label=label,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(data["PSD_class"].str.capitalize())
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    finish_axis(ax)


def make_output_table(ax, data):
    ax.axis("off")
    add_panel_label(ax, "C")
    table_data = []
    for _, row in data.iterrows():
        table_data.append([
            row["PSD_class"].capitalize(),
            f"{row['drawdown_time_s']:.0f}",
            f"{row['TDS_percent']:.2f}",
            f"{row['extraction_yield_percent']:.2f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["PSD input", "Drawdown\n(s)", "TDS\n(%)", "Extraction\nyield (%)"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.03, 0.10, 0.94, 0.72],
        colWidths=[0.24, 0.27, 0.18, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.18)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#666666")
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#F7F7F7")

    ax.text(
        0.03,
        0.92,
        "Brewing output summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def make_residual_panel(ax, data):
    add_panel_label(ax, "D")
    x = np.arange(len(data))
    width = 0.34
    water_res = np.abs(data["water_balance_residual_g"].to_numpy(dtype=float))
    solids_res = np.abs(data["dissolved_solids_balance_residual_g"].to_numpy(dtype=float))

    plot_floor = 1e-16
    water_plot = np.maximum(water_res, plot_floor)
    solids_plot = np.maximum(solids_res, plot_floor)

    ax.bar(
        x - width / 2,
        water_plot,
        width=width,
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.5,
        label="Water",
    )
    ax.bar(
        x + width / 2,
        solids_plot,
        width=width,
        color="#D8B365",
        edgecolor="black",
        linewidth=0.5,
        label="Extractable solids",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3e-10)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(data["PSD_class"].str.capitalize())
    ax.set_ylabel("Absolute balance residual (g)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    finish_axis(ax)


def make_figure(data):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 6.2),
        gridspec_kw={"height_ratios": [1.06, 0.94], "wspace": 0.34, "hspace": 0.46},
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    add_panel_label(ax_a, "A")
    water_components = [
        "cup_water_g",
        "retained_water_g",
        "mobile_bed_water_g",
        "pooled_water_g",
    ]
    water_colors = ["#0072B2", "#56B4E9", "#009E73", "#8DD3C7"]
    water_labels = ["Cup water", "Retained water", "Mobile bed water", "Pooled water"]
    water_hatches = ["", "//", "\\\\", ".."]
    stacked_bar(
        ax_a,
        data,
        water_components,
        water_colors,
        water_labels,
        water_hatches,
        "Water inventory (g)",
        ylim=(0, 320),
    )
    ax_a.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.52, 1.03),
        columnspacing=0.9,
        handlelength=1.5,
    )

    add_panel_label(ax_b, "B")
    solids_components = [
        "cup_dissolved_solids_g",
        "bed_liquid_dissolved_solids_g",
        "remaining_extractable_solids_g",
    ]
    solids_colors = ["#8C510A", "#D8B365", "#BDBDBD"]
    solids_labels = [
        "Cup dissolved solids",
        "Bed-liquid dissolved solids",
        "Remaining extractable solids",
    ]
    solids_hatches = ["", "//", ".."]
    stacked_bar(
        ax_b,
        data,
        solids_components,
        solids_colors,
        solids_labels,
        solids_hatches,
        "Extractable-solids inventory (g)",
        ylim=(0, 6.8),
    )
    ax_b.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.52, 1.03),
        handlelength=1.5,
    )

    make_output_table(ax_c, data)
    make_residual_panel(ax_d, data)

    return fig


def write_caption(output_dir):
    caption = (
        "Figure 4. Calibrated reconstruction of water and extractable-solids "
        "inventories. (A) Endpoint water partitioning among cup water, retained "
        "water, mobile bed water, and pooled water for the measured coarse, "
        "medium, and fine PSD inputs. (B) Endpoint extractable-solids partitioning "
        "among cup dissolved solids, bed-liquid dissolved solids, and remaining "
        "extractable solids. (C) Corresponding drawdown time, TDS, and extraction "
        "yield. (D) Absolute water and extractable-solids balance residuals. "
        "These quantities are reconstructed model inventories rather than "
        "independently measured internal states. Cup water and retained water are "
        "parts of the same conserved water balance, while cup dissolved solids, "
        "bed-liquid dissolved solids, and remaining extractable solids are parts "
        "of the same extractable-solids balance. Fine PSD increased retained "
        "water and cup dissolved-solids delivery relative to coarse PSD, while "
        "preserving numerical mass balance. Water and extractable-solids residuals "
        "remained near numerical precision."
    )
    (output_dir / "figure4_caption.md").write_text(caption + "\n", encoding="utf-8")


def validate_balances(data):
    water_total = (
        data["cup_water_g"]
        + data["retained_water_g"]
        + data["mobile_bed_water_g"]
        + data["pooled_water_g"]
    )
    solids_total = (
        data["cup_dissolved_solids_g"]
        + data["bed_liquid_dissolved_solids_g"]
        + data["remaining_extractable_solids_g"]
    )
    water_check = data["input_water_g"] - water_total
    solids_check = data["initial_extractable_solids_g"] - solids_total
    max_water_residual = float(np.max(np.abs(water_check.to_numpy(dtype=float))))
    max_solids_residual = float(np.max(np.abs(solids_check.to_numpy(dtype=float))))
    return max_water_residual, max_solids_residual


def main():
    set_publication_style()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "jfe_revision" / "calibrated_partitioning" / "revised_figure4"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_embedded_data()
    cleaned_csv = output_dir / "figure4_plotting_data.csv"
    data.to_csv(cleaned_csv, index=False)

    max_water_residual, max_solids_residual = validate_balances(data)

    fig = make_figure(data)
    png_path = output_dir / "figure4_calibrated_partitioning_revised.png"
    pdf_path = output_dir / "figure4_calibrated_partitioning_revised.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    write_caption(output_dir)
    caption_path = output_dir / "figure4_caption.md"

    print(f"Maximum water balance residual from plotted stacks: {max_water_residual:.3e} g")
    print(f"Maximum extractable-solids balance residual from plotted stacks: {max_solids_residual:.3e} g")
    print("Generated files:")
    for path in [png_path, pdf_path, cleaned_csv, caption_path]:
        print(path)


if __name__ == "__main__":
    main()
