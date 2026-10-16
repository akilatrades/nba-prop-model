"""Calibration chart: does a 30% prediction happen about 30% of the time?"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def plot_calibration(table, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.55, "perfect calibration", color=TEXT_2, fontsize=9, rotation=41)
    ax.plot(table.predicted, table.actual, color=SERIES, linewidth=2, zorder=2)
    ax.scatter(
        table.predicted,
        table.actual,
        s=64,
        color=SERIES,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [f"{t:.0%}" for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Model's predicted chance", color=TEXT_2, fontsize=10)
    ax.set_ylabel("How often it actually happened", color=TEXT_2, fontsize=10)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, metadata={"Date": None})
    plt.close(fig)
