from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _figure_utils import ANALYSIS_DIR, COLORS, FIG_DIR, percent, read_csv_dicts, save_figure, set_paper_style


HIGHLIGHTS = {
    "A": ("Qwen3Guard-Gen-0.6B|gpt-4o-mini|wildguard", COLORS["blue"]),
    "B": ("shieldgemma-2b|phi-4|Llama-Guard-3-8B", COLORS["green"]),
    "C": ("Qwen3Guard-Gen-0.6B|Ministral-8B-Instruct-2410|wildguard", COLORS["orange"]),
}


def main() -> None:
    set_paper_style()
    rows = read_csv_dicts(ANALYSIS_DIR / "audit_configuration_asr.csv")
    rows = sorted(rows, key=lambda r: float(r["asr"]))
    asrs = [float(r["asr"]) for r in rows]
    xs = list(range(1, len(asrs) + 1))

    fig, ax = plt.subplots(figsize=(6.15, 2.28))
    ax.fill_between(xs, asrs, color=COLORS["blue"], alpha=0.08, linewidth=0)
    ax.plot(xs, asrs, color=COLORS["blue_dark"], lw=1.2, solid_capstyle="round")
    ax.scatter(xs[::6], asrs[::6], s=4, color=COLORS["blue_dark"], alpha=0.45, linewidths=0)

    median = sorted(asrs)[len(asrs) // 2]
    ax.axhline(median, color=COLORS["muted"], lw=0.65, ls=(0, (2, 2)), zorder=0)

    by_name = {r["config_name"]: idx for idx, r in enumerate(rows)}
    legend_handles = [
        Line2D([0], [0], color=COLORS["blue_dark"], lw=1.2, label="ASR"),
        Line2D([0], [0], color=COLORS["muted"], lw=0.65, ls=(0, (2, 2)), label=f"Median {percent(median)}"),
    ]
    for label, (name, color) in HIGHLIGHTS.items():
        idx = by_name[name]
        v = float(rows[idx]["asr"])
        ax.scatter(idx + 1, v, s=28, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=5.5,
                label=f"Config {label} {percent(v)}",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )

    ax.set_xlim(1, len(asrs))
    ax.set_ylim(0.16, 0.48)
    ax.set_xlabel("Defense configurations, sorted by ASR")
    ax.set_ylabel("Attack success rate")
    ax.set_yticks([0.2, 0.3, 0.4])
    ax.set_yticklabels(["20%", "30%", "40%"])
    ax.set_xticks([1, 75, 150, 225, 275])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, FIG_DIR / "fig2_asr_distribution.pdf")


if __name__ == "__main__":
    main()
