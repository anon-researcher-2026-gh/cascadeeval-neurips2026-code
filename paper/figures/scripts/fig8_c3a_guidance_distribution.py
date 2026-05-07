from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _figure_utils import ANALYSIS_DIR, COLORS, FIG_DIR, read_csv_dicts, save_figure, set_paper_style


DATA_PATH = ANALYSIS_DIR / "c3a_guidance_distribution.csv"
CONFIG_ORDER = ["Config A", "Config B", "Config C"]
TECHNIQUE_ORDER = ["LEX", "MSYN", "SEM", "PRAG", "ORTH", "FALLBACK"]
TECHNIQUE_LABELS = {
    "LEX": "LEX",
    "MSYN": "MSYN",
    "SEM": "SEM",
    "PRAG": "PRAG",
    "ORTH": "ORTH",
    "FALLBACK": "Fallback",
}
TECHNIQUE_COLORS = {
    "LEX": COLORS["blue"],
    "MSYN": COLORS["purple"],
    "SEM": COLORS["orange"],
    "PRAG": COLORS["green"],
    "ORTH": COLORS["red"],
    "FALLBACK": "#aab4c0",
}


def load_guidance_shares(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv_dicts(path)
    shares: dict[str, dict[str, float]] = {config: {} for config in CONFIG_ORDER}
    for row in rows:
        if row["method"] != "c3a":
            continue
        config = row["config_label"]
        if config not in shares:
            continue
        shares[config][row["category"]] = float(row["share"])
    return shares


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Appendix Figure 8 C3A guidance distribution.")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--out", default=str(FIG_DIR / "fig8_c3a_guidance_distribution.pdf"))
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise SystemExit(f"Missing input CSV: {data_path}")

    shares = load_guidance_shares(data_path)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 1.85))
    y_positions = list(range(len(CONFIG_ORDER)))

    for y, config in zip(y_positions, CONFIG_ORDER, strict=True):
        left = 0.0
        for technique in TECHNIQUE_ORDER:
            value = shares[config].get(technique, 0.0)
            ax.barh(
                y,
                value,
                left=left,
                height=0.46,
                color=TECHNIQUE_COLORS[technique],
                edgecolor="white",
                linewidth=0.7,
                label=TECHNIQUE_LABELS[technique] if y == 0 else None,
            )
            if value >= 0.075:
                text_color = COLORS["ink"] if technique == "FALLBACK" else "white"
                ax.text(
                    left + value / 2,
                    y,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=text_color,
                )
            left += value

    ax.set_xlim(0, 1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(CONFIG_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Share of C3A full-loop guidance choices")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=0.85,
        handlelength=1.0,
    )

    save_figure(fig, Path(args.out))


if __name__ == "__main__":
    main()
