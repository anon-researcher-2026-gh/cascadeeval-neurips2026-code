from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _figure_utils import ANALYSIS_DIR, COLORS, FIG_DIR, read_csv_dicts, save_figure, set_paper_style


DATA_PATH = ANALYSIS_DIR / "technique_stage_pass_rates.csv"
TECHNIQUES = ["LEX", "MSYN", "SEM", "PRAG", "ORTH"]


def load_stage_technique_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    rows = read_csv_dicts(path)
    stages = [row["stage"] for row in rows]
    matrix = [[float(row[tech]) for tech in TECHNIQUES] for row in rows]
    return stages, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the stage x technique pass-rate heatmap.")
    parser.add_argument(
        "--input",
        default=str(DATA_PATH),
        help="Derived stage x technique pass-rate CSV.",
    )
    parser.add_argument("--out", default=str(FIG_DIR / "fig5_technique_heatmap.pdf"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")

    stages, matrix = load_stage_technique_matrix(path)

    set_paper_style()
    cmap = LinearSegmentedColormap.from_list("cascade_reds", ["#f7f8fa", "#e1afa0", "#9f3b36"])
    fig, ax = plt.subplots(figsize=(5.9, 2.25))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TECHNIQUES)))
    ax.set_xticklabels(TECHNIQUES)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("Technique label")
    ax.set_ylabel("Cascade stage")
    for i, _stage in enumerate(stages):
        for j, _tech in enumerate(TECHNIQUES):
            v = matrix[i][j]
            ax.text(
                j,
                i,
                f"{100 * v:.0f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if v > 0.58 else COLORS["ink"],
            )
    ax.set_xticks([x - 0.5 for x in range(1, len(TECHNIQUES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(stages))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Pass rate")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0%", "50%", "100%"])

    save_figure(fig, Path(args.out))


if __name__ == "__main__":
    main()
