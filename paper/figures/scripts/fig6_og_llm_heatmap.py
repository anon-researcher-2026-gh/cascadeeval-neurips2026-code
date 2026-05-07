from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _figure_utils import ANALYSIS_DIR, COLORS, FIG_DIR, read_csv_dicts, save_figure, set_paper_style, short_model


def main() -> None:
    set_paper_style()
    rows = read_csv_dicts(ANALYSIS_DIR / "audit_configuration_asr.csv")
    targets = sorted({r["target_llm"] for r in rows}, key=short_model)
    ogs = sorted({r["output_guard"] for r in rows}, key=short_model)

    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        vals[(r["target_llm"], r["output_guard"])].append(float(r["og_block_rate"]))
    means = {
        key: sum(values) / len(values)
        for key, values in vals.items()
    }
    matrix = [[means[(target, og)] for og in ogs] for target in targets]

    cmap = LinearSegmentedColormap.from_list("cascade_blues", ["#f5f8fb", "#9eb9d8", "#1f4f86"])
    fig, ax = plt.subplots(figsize=(6.15, 3.65))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.04, vmax=0.34, aspect="auto")

    ax.set_xticks(range(len(ogs)))
    ax.set_xticklabels([short_model(og) for og in ogs], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([short_model(target) for target in targets])
    ax.set_xlabel("Output Guard")
    ax.set_ylabel("Target LLM")

    for i, target in enumerate(targets):
        for j, og in enumerate(ogs):
            v = means[(target, og)]
            color = "white" if v >= 0.245 else COLORS["ink"]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.2, color=color)

    ax.set_xticks([x - 0.5 for x in range(1, len(ogs))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(targets))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("OG block rate")
    cbar.set_ticks([0.1, 0.2, 0.3])
    cbar.set_ticklabels(["10%", "20%", "30%"])
    save_figure(fig, FIG_DIR / "fig6_og_llm_heatmap.pdf")


if __name__ == "__main__":
    main()
