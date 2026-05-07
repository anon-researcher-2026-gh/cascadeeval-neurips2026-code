from __future__ import annotations

from collections.abc import Iterable

from matplotlib import patches
import matplotlib.pyplot as plt

from _figure_utils import COLORS, FIG_DIR, save_figure, set_paper_style


INK = COLORS["ink"]
MUTED = COLORS["muted"]
GRID = "#cfd7e2"
LIGHT = "#e7ebf1"
PANEL = "#fbfcfd"
CASCADE = COLORS["blue_dark"]
JUDGE = "#7a3f34"


def line(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = GRID,
    lw: float = 0.65,
    ls: str = "-",
    zorder: int = 1,
) -> None:
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=zorder)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.72,
    ls: str = "-",
    scale: float = 8.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "linestyle": ls,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": scale,
        },
        zorder=2,
    )


def node(ax: plt.Axes, x: float, y: float, *, edge: str, radius: float = 0.015, lw: float = 1.35) -> None:
    size = 32 if radius <= 0.011 else 68
    ax.scatter([x], [y], s=size, facecolors="white", edgecolors=edge, linewidths=lw, zorder=3)


def rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = GRID,
    face: str = "white",
    lw: float = 0.6,
) -> None:
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            zorder=0,
        )
    )


def bracket(ax: plt.Axes, x0: float, x1: float, y: float, label: str, *, color: str) -> None:
    line(ax, x0, y, x1, y, color=color, lw=0.9)
    line(ax, x0, y, x0, y - 0.024, color=color, lw=0.9)
    line(ax, x1, y, x1, y - 0.024, color=color, lw=0.9)
    ax.text((x0 + x1) / 2, y + 0.028, label, ha="center", va="center", fontsize=7.7, color=color)


def table_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    value: str,
    *,
    accent: str = GRID,
    title_color: str = MUTED,
    title_size: float = 5.65,
    value_size: float = 5.45,
) -> None:
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_size, color=title_color, linespacing=1.0)
    ax.text(x + w / 2, y + h * 0.26, value, ha="center", va="center", fontsize=value_size, color=INK, linespacing=1.0)
    line(ax, x + 0.012, y + 0.012, x + w - 0.012, y + 0.012, color=accent, lw=0.95)


def small_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    rect(ax, x, y, w, h, edge=GRID, face="white", lw=0.6)
    del subtitle
    title_size = 6.95 if len(title) <= 12 else 5.65
    ax.text(x + w / 2, y + h * 0.50, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK)


def ensure_no_text_overlap(fig: plt.Figure, texts: Iterable[plt.Text]) -> None:
    """Fail fast if figure text boxes visibly collide in display coordinates."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in texts:
        if not text.get_text():
            continue
        box = text.get_window_extent(renderer=renderer).expanded(1.01, 1.02)
        boxes.append((text.get_text(), box))
    collisions: list[tuple[str, str]] = []
    for i, (label_a, box_a) in enumerate(boxes):
        for label_b, box_b in boxes[i + 1 :]:
            if box_a.overlaps(box_b):
                collisions.append((label_a, label_b))
    if collisions:
        pairs = "; ".join(f"{a!r} vs {b!r}" for a, b in collisions[:8])
        raise RuntimeError(f"Figure 1 text overlaps detected: {pairs}")


def main() -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.05, 3.02))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    texts: list[plt.Text] = []

    def add_text(*args, **kwargs) -> None:
        texts.append(ax.text(*args, **kwargs))

    # Stage positions.
    xs = {
        "prompt": 0.115,
        "ig": 0.295,
        "llm": 0.465,
        "og": 0.635,
        "y": 0.785,
        "judge": 0.895,
    }
    spine_y = 0.735

    # Top brackets make the evaluation boundary explicit.
    bracket(
        ax,
        xs["ig"] - 0.105,
        xs["og"] + 0.105,
        0.925,
        r"deployed defense  $\mathcal{D}=(g_{\mathrm{in}}, m, g_{\mathrm{out}})$",
        color=CASCADE,
    )
    bracket(ax, xs["judge"] - 0.073, xs["judge"] + 0.073, 0.925, "calibrated measurement", color=JUDGE)

    # Pipeline spine.
    node(ax, xs["prompt"], spine_y, edge=INK, radius=0.010, lw=1.05)
    node(ax, xs["ig"], spine_y, edge=CASCADE)
    node(ax, xs["llm"], spine_y, edge=CASCADE)
    node(ax, xs["og"], spine_y, edge=CASCADE)
    node(ax, xs["y"], spine_y, edge=MUTED, radius=0.010, lw=1.05)
    node(ax, xs["judge"], spine_y, edge=JUDGE)
    arrow(ax, (xs["prompt"] + 0.010, spine_y), (xs["ig"] - 0.016, spine_y), color=INK, lw=0.72, scale=8.5)
    arrow(ax, (xs["ig"] + 0.016, spine_y), (xs["llm"] - 0.016, spine_y), color=INK, lw=0.72, scale=8.5)
    arrow(ax, (xs["llm"] + 0.016, spine_y), (xs["og"] - 0.016, spine_y), color=INK, lw=0.72, scale=8.5)
    arrow(ax, (xs["og"] + 0.016, spine_y), (xs["y"] - 0.010, spine_y), color=INK, lw=0.72, scale=8.5)
    arrow(ax, (xs["y"] + 0.010, spine_y), (xs["judge"] - 0.016, spine_y), color=MUTED, lw=0.65, ls=(0, (5, 4)), scale=8.0)

    stage_labels = [
        ("prompt", "prompt", r"$x$", INK),
        ("ig", "input guard", r"$g_{\mathrm{in}}$", CASCADE),
        ("llm", "target LLM", r"$m$", CASCADE),
        ("og", "output guard", r"$g_{\mathrm{out}}$", CASCADE),
        ("y", "", r"$y$", MUTED),
        ("judge", "judge", r"$J_{\tau}(y)$", JUDGE),
    ]
    for key, title, symbol, color in stage_labels:
        if title:
            add_text(xs[key], 0.865, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
        add_text(xs[key], 0.79, symbol, ha="center", va="center", fontsize=7.0, color=INK if key == "y" else color)

    outcomes = [
        ("ig", "input\nblock", CASCADE),
        ("llm", "model\nrefusal", CASCADE),
        ("og", "output\nblock", CASCADE),
        ("judge", "harm score\n(calibrated)", JUDGE),
    ]
    for key, label, color in outcomes:
        line(ax, xs[key], spine_y - 0.022, xs[key], 0.647, color=color, lw=0.56, ls=(0, (3, 3)))
        add_text(xs[key], 0.612, label, ha="center", va="center", fontsize=5.7, color=color, linespacing=1.08)

    # Stage-level record table.
    table_x, table_y = 0.16, 0.39
    table_w, table_h = 0.78, 0.15
    rect(ax, table_x, table_y, table_w, table_h, edge=GRID, face=PANEL, lw=0.62)
    add_text(0.025, table_y + table_h * 0.62, "per-attempt\ntrace record $r$", ha="left", va="center", fontsize=6.0, fontweight="bold", linespacing=1.06)
    line(ax, 0.03, table_y + table_h + 0.035, 0.96, table_y + table_h + 0.035, color=GRID, lw=0.55)

    cols = [
        (0.09, "prompt", r"$x_i$", INK),
        (0.12, "IG label", "pass / block", CASCADE),
        (0.20, "LLM output", r"$y_i$ + refusal flag", CASCADE),
        (0.12, "OG label", "pass / block", CASCADE),
        (0.14, "judge", r"$s_i$ + harm label", JUDGE),
        (0.21, "blocking stage", "in / llm / out / none", MUTED),
        (0.12, "success", "0 / 1", MUTED),
    ]
    x = table_x
    centers: dict[str, float] = {}
    for idx, (frac, title, value, accent) in enumerate(cols):
        w = table_w * frac
        if idx:
            line(ax, x, table_y + 0.018, x, table_y + table_h - 0.018, color=GRID, lw=0.5)
        table_cell(
            ax,
            x,
            table_y,
            w,
            table_h,
            title,
            value,
            accent=accent,
            title_color=accent if idx in {1, 2, 3, 4} else MUTED,
            value_size=5.15 if idx == 5 else 5.45,
        )
        centers[str(idx)] = x + w / 2
        x += w

    # Dashed verticals align observed outcomes to their record fields.
    table_top = table_y + table_h
    for x0, cx, color in [
        (xs["ig"], centers["1"], CASCADE),
        (xs["llm"], centers["2"], CASCADE),
        (xs["og"], centers["3"], CASCADE),
        (xs["judge"], centers["4"], JUDGE),
    ]:
        line(ax, x0, 0.575, x0, table_top, color=color, lw=0.52, ls=(0, (3, 3)))
        if abs(x0 - cx) > 0.004:
            line(ax, min(x0, cx), table_top, max(x0, cx), table_top, color=GRID, lw=0.5)

    output_boxes = [
        (0.12, "ASR", ""),
        (0.33, "bottleneck attribution", ""),
        (0.55, "KB profiles", ""),
        (0.76, "C3A feedback", ""),
    ]
    box_w, box_h, box_y = 0.17, 0.072, 0.122
    box_top = box_y + box_h
    output_centers = [x + box_w / 2 for x, _, _ in output_boxes]

    # Aggregation products use one clean rail with centered drops into each box.
    rail_y = 0.287
    stem_x = table_x + table_w / 2
    line(ax, stem_x, table_y - 0.006, stem_x, rail_y, color=MUTED, lw=0.62)
    line(ax, output_centers[0], rail_y, output_centers[-1], rail_y, color=MUTED, lw=0.62)
    add_text(0.025, rail_y + 0.006, "aggregate over\nmany traces $R$", ha="left", va="center", fontsize=5.8, fontweight="bold", linespacing=1.06)

    for x, title, subtitle in output_boxes:
        small_box(ax, x, box_y, box_w, box_h, title, subtitle)
    for cx in output_centers:
        arrow(ax, (cx, rail_y), (cx, box_top + 0.004), color=MUTED, lw=0.52, scale=6.6)

    ensure_no_text_overlap(fig, texts)
    save_figure(fig, FIG_DIR / "fig1_pipeline.pdf")


if __name__ == "__main__":
    main()
