from __future__ import annotations

import matplotlib.pyplot as plt

from _figure_utils import COLORS, FIG_DIR, load_verified_numbers, percent, save_figure, set_paper_style


def main() -> None:
    set_paper_style()
    n = load_verified_numbers()
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(6.3, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.28], "wspace": 0.34},
    )

    stages = ["IG", "LLM", "OG"]
    vals = [n["bottleneck_ig"], n["bottleneck_llm"], n["bottleneck_og"]]
    ci = [
        n["bottleneck_ig_bootstrap95"],
        n["bottleneck_llm_bootstrap95"],
        n["bottleneck_og_bootstrap95"],
    ]
    lows = [center - bounds[0] for bounds, center in zip(ci, vals)]
    highs = [bounds[1] - center for bounds, center in zip(ci, vals)]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    bars = ax1.bar(stages, vals, color=colors, width=0.58, edgecolor="white", linewidth=0.6)
    ax1.errorbar(stages, vals, yerr=[lows, highs], fmt="none", ecolor=COLORS["ink"], elinewidth=0.75, capsize=2)
    for bar, v in zip(bars, vals):
        ax1.text(
            bar.get_x() + bar.get_width() + 0.055,
            max(v, 0.055),
            percent(v),
            ha="left",
            va="center",
            fontsize=7,
            color=COLORS["ink"],
        )
    ax1.set_xlim(-0.55, 2.72)
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("Share of configurations")
    ax1.set_yticks([0, 0.25, 0.5, 0.75])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%"])
    ax1.grid(axis="y", color=COLORS["grid"], lw=0.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(-0.12, 1.02, "(a)", transform=ax1.transAxes, ha="left", va="bottom", fontweight="bold")

    remaining = [
        ("Submitted", 1.0, COLORS["muted"]),
        ("After IG", n["survival_after_ig"], COLORS["blue"]),
        ("After LLM", n["survival_after_llm"], COLORS["green"]),
        ("After OG", n["survival_after_og"], COLORS["orange"]),
        ("ASR", n["asr_mean"], COLORS["red"]),
    ]
    labels = [item[0] for item in remaining]
    values = [item[1] for item in remaining]
    stage_colors = [item[2] for item in remaining]
    y_pos = list(range(len(values)))
    ax2.barh(y_pos, values, color=stage_colors, height=0.46, edgecolor="white", linewidth=0.6)
    ax2.invert_yaxis()
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Share of submitted attempts")
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.grid(axis="x", color=COLORS["grid"], lw=0.55)
    for y, v in zip(y_pos, values):
        ha = "right" if v > 0.92 else "left"
        x = v - 0.025 if v > 0.92 else v + 0.025
        color = "white" if v > 0.92 else COLORS["ink"]
        ax2.text(x, y, percent(v), va="center", ha=ha, fontsize=7, color=color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(-0.06, 1.02, "(b)", transform=ax2.transAxes, ha="left", va="bottom", fontweight="bold")

    save_figure(fig, FIG_DIR / "fig3_bottleneck_funnel.pdf")


if __name__ == "__main__":
    main()
