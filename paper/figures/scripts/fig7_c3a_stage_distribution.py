from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _figure_utils import COLORS, FIG_DIR, ROOT, read_csv_dicts, save_figure, set_paper_style


DATA_PATH = ROOT / "paper" / "data" / "gpu_results_20260505" / "c3a_stage_distribution.csv"
CONFIG_ORDER = ["Config A", "Config B", "Config C"]
FULL_C3A_METHOD = {
    "Config A": "c3a-full-kb",
    "Config B": "c3a-full-kb",
    "Config C": "c3a-full-kb",
}
STAGE_ORDER = ["IG", "LLM", "OG", "PASSED_GUARDS"]
STAGE_LABELS = {
    "IG": "Input Guard",
    "LLM": "Target LLM",
    "OG": "Output Guard",
    "PASSED_GUARDS": "Passed",
}
STAGE_COLORS = {
    "IG": COLORS["blue"],
    "LLM": COLORS["orange"],
    "OG": COLORS["green"],
    "PASSED_GUARDS": "#aab4c0",
}


def load_stage_shares(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv_dicts(path)
    shares: dict[str, dict[str, float]] = {config: {} for config in CONFIG_ORDER}
    for row in rows:
        config = row["config_label"]
        if config not in FULL_C3A_METHOD:
            continue
        if row["method"] != FULL_C3A_METHOD[config]:
            continue
        shares[config][row["stage"]] = float(row["share"])
    missing = [
        f"{config}:{stage}"
        for config in CONFIG_ORDER
        for stage in STAGE_ORDER
        if stage not in shares[config]
    ]
    if missing:
        raise SystemExit(f"Missing stage rows: {', '.join(missing)}")
    return shares


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Appendix Figure 7 C3A stage distribution.")
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--out", default=str(FIG_DIR / "fig7_c3a_stage_distribution.pdf"))
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise SystemExit(f"Missing input CSV: {data_path}")

    shares = load_stage_shares(data_path)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 1.85))
    y_positions = list(range(len(CONFIG_ORDER)))

    for y, config in zip(y_positions, CONFIG_ORDER, strict=True):
        left = 0.0
        for stage in STAGE_ORDER:
            value = shares[config][stage]
            ax.barh(
                y,
                value,
                left=left,
                height=0.46,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                linewidth=0.7,
                label=STAGE_LABELS[stage] if y == 0 else None,
            )
            if value >= 0.075:
                text_color = "white" if stage in {"IG", "LLM", "OG"} else COLORS["ink"]
                ax.text(
                    left + value / 2,
                    y,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=text_color,
                )
            left += value

    ax.set_xlim(0, 1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(CONFIG_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Share of C3A full-loop iterations")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
    )

    save_figure(fig, Path(args.out))


if __name__ == "__main__":
    main()
