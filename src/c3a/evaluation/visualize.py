"""可視化モジュール.

Part 1 分析結果の可視化を提供する。
- 構成別 ASR ヒートマップ
- ステージ別ブロック率棒グラフ
- 技法別有効性グラフ
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.c3a.evaluation.analysis import AnalysisResult
from src.c3a.types import INPUT_GUARDS, OUTPUT_GUARDS, Stage, TARGET_LLMS, TECHNIQUES


def setup_matplotlib() -> None:
    """matplotlib の設定."""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def plot_asr_heatmap(
    asr_matrix: NDArray[np.float64],
    save_path: str | Path | None = None,
    title: str = "Attack Success Rate by Defense Configuration",
) -> plt.Figure:
    """構成別 ASR ヒートマップを描画する.

    2D ヒートマップとして表示（IG×LLM で、OG は平均または個別パネル）。

    Args:
        asr_matrix: (n_ig, n_llm, n_og) の ASR マトリックス
        save_path: 保存先パス（オプション）
        title: グラフタイトル

    Returns:
        Figure オブジェクト
    """
    setup_matplotlib()

    n_og = len(OUTPUT_GUARDS)
    fig, axes = plt.subplots(1, n_og, figsize=(4 * n_og + 2, 5))

    if n_og == 1:
        axes = [axes]

    for og_idx, (ax, og_name) in enumerate(zip(axes, OUTPUT_GUARDS)):
        # IG × LLM の 2D スライス
        data = asr_matrix[:, :, og_idx]

        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        # 軸ラベル
        ax.set_xticks(range(len(TARGET_LLMS)))
        ax.set_xticklabels(TARGET_LLMS, rotation=45, ha="right")
        ax.set_yticks(range(len(INPUT_GUARDS)))
        ax.set_yticklabels(INPUT_GUARDS)

        ax.set_xlabel("Target LLM")
        ax.set_ylabel("Input Guard")
        ax.set_title(f"Output Guard: {og_name}")

        # 値を表示
        for i in range(len(INPUT_GUARDS)):
            for j in range(len(TARGET_LLMS)):
                val = data[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color)

    # カラーバー
    fig.colorbar(im, ax=axes, label="ASR", shrink=0.8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_stage_block_rates(
    block_rates: NDArray[np.float64],
    save_path: str | Path | None = None,
    title: str = "Stage-wise Block Rate",
) -> plt.Figure:
    """ステージ別ブロック率の棒グラフを描画する.

    Args:
        block_rates: (3,) のブロック率配列 [IG, LLM, OG]
        save_path: 保存先パス（オプション）
        title: グラフタイトル

    Returns:
        Figure オブジェクト
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    stages = ["Input Guard", "Target LLM", "Output Guard"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.bar(stages, block_rates, color=colors, edgecolor="black", linewidth=1.5)

    # 値を表示
    for bar, rate in zip(bars, block_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Block Rate")
    ax.set_title(title)
    ax.set_ylim(0, max(block_rates) * 1.2)

    # グリッド
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_technique_effectiveness(
    effectiveness: NDArray[np.float64],
    save_path: str | Path | None = None,
    title: str = "Technique Effectiveness by Stage",
) -> plt.Figure:
    """技法別有効性のグラフを描画する.

    Args:
        effectiveness: (n_techniques, 3) の有効性マトリックス
        save_path: 保存先パス（オプション）
        title: グラフタイトル

    Returns:
        Figure オブジェクト
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TECHNIQUES))
    width = 0.25

    stages = ["Input Guard", "Target LLM", "Output Guard"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (stage, color) in enumerate(zip(stages, colors)):
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            effectiveness[:, i],
            width,
            label=stage,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Attack Technique")
    ax.set_ylabel("Pass Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(TECHNIQUES)
    ax.legend()
    ax.set_ylim(0, 1)

    # グリッド
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_bottleneck_distribution(
    distribution: dict[Stage, int],
    save_path: str | Path | None = None,
    title: str = "Bottleneck Stage Distribution",
) -> plt.Figure:
    """ボトルネックステージ分布の円グラフを描画する.

    Args:
        distribution: {stage: count} の辞書
        save_path: 保存先パス（オプション）
        title: グラフタイトル

    Returns:
        Figure オブジェクト
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = []
    sizes = []
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for stage in [Stage.INPUT_GUARD, Stage.TARGET_LLM, Stage.OUTPUT_GUARD]:
        if stage in distribution and distribution[stage] > 0:
            labels.append(stage.value)
            sizes.append(distribution[stage])

    if not sizes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors[: len(sizes)],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05] * len(sizes),
    )

    # スタイル調整
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_config_ranking(
    rankings: list[tuple[str, float]],
    top_n: int = 10,
    save_path: str | Path | None = None,
    title: str = "Top Vulnerable Configurations",
) -> plt.Figure:
    """構成ランキングの棒グラフを描画する.

    Args:
        rankings: [(config_name, asr), ...] のリスト（ASR降順）
        top_n: 表示する構成数
        save_path: 保存先パス（オプション）
        title: グラフタイトル

    Returns:
        Figure オブジェクト
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    top_configs = rankings[:top_n]
    names = [name for name, _ in top_configs]
    asrs = [asr for _, asr in top_configs]

    # カラーマップ（高 ASR = 赤、低 ASR = 緑）
    colors = plt.cm.RdYlGn_r(np.array(asrs))

    bars = ax.barh(range(len(names)), asrs, color=colors, edgecolor="black")

    # ラベル
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # 上から高い順

    # 値を表示
    for bar, asr in zip(bars, asrs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{asr:.1%}",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Attack Success Rate (ASR)")
    ax.set_title(title)
    ax.set_xlim(0, max(asrs) * 1.15)

    # グリッド
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def create_all_figures(
    result: AnalysisResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """全ての図を生成して保存する.

    Args:
        result: 分析結果
        output_dir: 出力ディレクトリ

    Returns:
        {figure_name: path} の辞書
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Path] = {}

    # ASR ヒートマップ
    path = output_dir / "asr_heatmap.png"
    plot_asr_heatmap(result.asr_matrix, save_path=path)
    figures["asr_heatmap"] = path
    plt.close()

    # ステージ別ブロック率
    path = output_dir / "stage_block_rates.png"
    plot_stage_block_rates(result.stage_block_rates, save_path=path)
    figures["stage_block_rates"] = path
    plt.close()

    # 技法別有効性
    path = output_dir / "technique_effectiveness.png"
    plot_technique_effectiveness(result.technique_effectiveness, save_path=path)
    figures["technique_effectiveness"] = path
    plt.close()

    # ボトルネック分布
    path = output_dir / "bottleneck_distribution.png"
    plot_bottleneck_distribution(result.bottleneck_distribution, save_path=path)
    figures["bottleneck_distribution"] = path
    plt.close()

    # 構成ランキング
    path = output_dir / "config_ranking.png"
    plot_config_ranking(result.config_rankings, save_path=path)
    figures["config_ranking"] = path
    plt.close()

    return figures
