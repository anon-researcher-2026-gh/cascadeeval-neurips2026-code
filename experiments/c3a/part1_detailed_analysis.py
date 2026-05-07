"""C3A Part 1: 詳細分析と可視化.

既存ラベルデータを詳細に分析し、論文品質のグラフを生成する。

Usage:
    uv run python experiments/c3a/part1_detailed_analysis.py

出力:
    experiments/c3a/results/analysis/
    ├── figures/          # PNG グラフ
    ├── report.md         # 詳細レポート
    └── statistics.json   # 統計データ
"""

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# =============================================================================
# 設定
# =============================================================================

DATA_PATH = Path("data/processed/en_v2_results.csv")
KB_PATH = Path("experiments/c3a/results/knowledge_base.json")
OUTPUT_DIR = Path("experiments/c3a/results/analysis")

# 論文品質のスタイル設定
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# カラーパレット
STAGE_COLORS = {"IG": "#E74C3C", "LLM": "#3498DB", "OG": "#2ECC71"}
TECHNIQUE_COLORS = {
    "LEX": "#9B59B6",
    "MSYN": "#E67E22",
    "SEM": "#1ABC9C",
    "PRAG": "#E91E63",
    "ORTH": "#34495E",
}

# Guard 表示名マッピング
GUARD_DISPLAY_NAMES = {
    "wildguard": "WildGuard",
    "Llama-Guard-3-8B": "LlamaGuard-3",
    "Llama-Guard-4-12B": "LlamaGuard-4",
    "Qwen3Guard-Gen-0.6B": "Qwen3Guard",
    "shieldgemma-2b": "ShieldGemma",
}

# LLM 表示名マッピング
LLM_DISPLAY_NAMES = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.0-flash": "Gemini-2.0",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen3-8B": "Qwen3-8B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "gemma-3-12b-it": "Gemma-3-12B",
    "GPT-OSS-20B": "GPT-OSS-20B",
    "phi-4": "Phi-4",
}


def setup_style():
    """matplotlib スタイルを設定."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("husl")


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class DetailedStats:
    """詳細統計データ."""

    n_attacks: int
    n_configs: int

    # グローバル統計
    global_ig_block_rate: float
    global_llm_block_rate: float
    global_og_block_rate: float
    bottleneck_stage: str

    # Guard 別統計
    ig_stats: dict[str, dict]
    llm_stats: dict[str, dict]
    og_stats: dict[str, dict]

    # 技法別統計
    technique_stats: dict[str, dict]

    # 構成別統計
    config_stats: dict[str, dict]

    # ASR Distribution Statistics
    asr_distribution: dict[str, float]


# =============================================================================
# ヘルパー関数
# =============================================================================


def col_equals(series: pd.Series, value: int | float) -> pd.Series:
    """データ型に関わらず値を比較する."""
    if series.dtype == "object":
        return series == str(int(value))
    else:
        return series == value


def get_display_name(name: str, name_map: dict) -> str:
    """表示名を取得."""
    return name_map.get(name, name.split("_")[-1] if "_" in name else name)


def save_figure(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    """図を保存."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    return path


# =============================================================================
# データ読み込み
# =============================================================================


def load_data() -> tuple[pd.DataFrame, dict]:
    """データを読み込む."""
    df = pd.read_csv(DATA_PATH)
    cols = df.columns.tolist()

    # コンポーネント抽出
    ig_cols = [c for c in cols if c.startswith("input_guard_") and c.endswith("_is_safe")]
    input_guards = [c.replace("input_guard_", "").replace("_is_safe", "") for c in ig_cols]

    llm_cols = [c for c in cols if c.startswith("llm_") and c.endswith("_is_refusal")]
    target_llms = [c.replace("llm_", "").replace("_is_refusal", "") for c in llm_cols]

    first_llm = target_llms[0]
    og_cols = [c for c in cols if c.startswith("output_guard_") and f"_on_{first_llm}_is_safe" in c]
    output_guards = [c.replace("output_guard_", "").replace(f"_on_{first_llm}_is_safe", "") for c in og_cols]

    tech_cols = [c for c in cols if c.startswith("taxonomy_technique_")]
    techniques = [c.replace("taxonomy_technique_", "") for c in tech_cols]

    components = {
        "input_guards": input_guards,
        "target_llms": target_llms,
        "output_guards": output_guards,
        "techniques": techniques,
    }

    return df, components


def load_knowledge_base() -> dict:
    """Knowledge Base を読み込む."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 統計計算
# =============================================================================


def compute_ig_stats(df: pd.DataFrame, input_guards: list[str], techniques: list[str]) -> dict:
    """Input Guard 別統計を計算."""
    stats = {}
    for ig in input_guards:
        col = f"input_guard_{ig}_is_safe"
        block_rate = float(col_equals(df[col], 0).mean())

        # 技法別通過率
        tech_rates = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() > 0:
                tech_rates[tech] = float(col_equals(df.loc[tech_mask, col], 1).mean())

        best_tech = max(tech_rates, key=tech_rates.get) if tech_rates else "N/A"

        stats[ig] = {
            "block_rate": block_rate,
            "pass_rate": 1 - block_rate,
            "technique_pass_rates": tech_rates,
            "best_technique": best_tech,
            "best_technique_rate": tech_rates.get(best_tech, 0),
        }

    return stats


def compute_llm_stats(df: pd.DataFrame, target_llms: list[str], techniques: list[str]) -> dict:
    """Target LLM 別統計を計算."""
    stats = {}
    for llm in target_llms:
        col = f"llm_{llm}_is_refusal"
        refusal_rate = float(col_equals(df[col], 1).mean())

        # 技法別非拒否率
        tech_rates = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() > 0:
                tech_rates[tech] = float(col_equals(df.loc[tech_mask, col], 0).mean())

        best_tech = max(tech_rates, key=tech_rates.get) if tech_rates else "N/A"

        stats[llm] = {
            "refusal_rate": refusal_rate,
            "pass_rate": 1 - refusal_rate,
            "technique_pass_rates": tech_rates,
            "best_technique": best_tech,
            "best_technique_rate": tech_rates.get(best_tech, 0),
        }

    return stats


def compute_og_stats(
    df: pd.DataFrame,
    output_guards: list[str],
    target_llms: list[str],
    techniques: list[str],
) -> dict:
    """Output Guard 別統計を計算."""
    stats = {}
    for og in output_guards:
        rates = []
        for llm in target_llms:
            col = f"output_guard_{og}_on_{llm}_is_safe"
            if col in df.columns:
                rates.append(float(col_equals(df[col], 0).mean()))

        block_rate = np.mean(rates) if rates else 0.0

        # 技法別通過率（全LLM平均）
        tech_rates = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() > 0:
                tech_df = df[tech_mask]
                llm_rates = []
                for llm in target_llms:
                    col = f"output_guard_{og}_on_{llm}_is_safe"
                    if col in df.columns:
                        llm_rates.append(float(col_equals(tech_df[col], 1).mean()))
                if llm_rates:
                    tech_rates[tech] = np.mean(llm_rates)

        best_tech = max(tech_rates, key=tech_rates.get) if tech_rates else "N/A"

        stats[og] = {
            "block_rate": block_rate,
            "pass_rate": 1 - block_rate,
            "technique_pass_rates": tech_rates,
            "best_technique": best_tech,
            "best_technique_rate": tech_rates.get(best_tech, 0),
        }

    return stats


def compute_technique_stats(
    df: pd.DataFrame, components: dict
) -> dict:
    """技法別統計を計算."""
    techniques = components["techniques"]
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]

    stats = {}
    for tech in techniques:
        tech_col = f"taxonomy_technique_{tech}"
        tech_mask = df[tech_col] == 1
        count = int(tech_mask.sum())

        if count == 0:
            continue

        tech_df = df[tech_mask]

        # IG 通過率
        ig_rates = []
        for ig in input_guards:
            col = f"input_guard_{ig}_is_safe"
            ig_rates.append(float(col_equals(tech_df[col], 1).mean()))

        # LLM 非拒否率
        llm_rates = []
        for llm in target_llms:
            col = f"llm_{llm}_is_refusal"
            llm_rates.append(float(col_equals(tech_df[col], 0).mean()))

        # OG 通過率
        og_rates = []
        for og in output_guards:
            rates = []
            for llm in target_llms:
                col = f"output_guard_{og}_on_{llm}_is_safe"
                if col in df.columns:
                    rates.append(float(col_equals(tech_df[col], 1).mean()))
            if rates:
                og_rates.append(np.mean(rates))

        stats[tech] = {
            "count": count,
            "percentage": count / len(df) * 100,
            "ig_pass_rate": np.mean(ig_rates),
            "llm_pass_rate": np.mean(llm_rates),
            "og_pass_rate": np.mean(og_rates) if og_rates else 0.0,
            "overall_effectiveness": np.mean([np.mean(ig_rates), np.mean(llm_rates), np.mean(og_rates) if og_rates else 0.0]),
        }

    return stats


def compute_config_stats(df: pd.DataFrame, components: dict) -> dict:
    """Compute per-config statistics including ASR."""
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]

    stats = {}

    for ig in input_guards:
        for llm in target_llms:
            for og in output_guards:
                ig_col = f"input_guard_{ig}_is_safe"
                refusal_col = f"llm_{llm}_is_refusal"
                og_col = f"output_guard_{og}_on_{llm}_is_safe"
                harmful_col = f"multijudge_on_{llm}_is_harmful"

                if og_col not in df.columns:
                    continue
                if harmful_col not in df.columns:
                    continue

                ig_pass = col_equals(df[ig_col], 1)
                llm_pass = col_equals(df[refusal_col], 0)
                og_pass = col_equals(df[og_col], 1)
                is_harmful = col_equals(df[harmful_col], 1)

                ig_rate = float(ig_pass.mean())
                llm_rate = float(llm_pass.mean())
                og_rate = float(og_pass.mean())

                # Pass Rate: all 3 stages passed
                all_pass = ig_pass & llm_pass & og_pass
                pass_rate = float(all_pass.mean())

                # ASR: all 3 stages passed AND judge says harmful
                asr = float((all_pass & is_harmful).mean())

                # Bottleneck
                block_rates = {"IG": 1 - ig_rate, "LLM": 1 - llm_rate, "OG": 1 - og_rate}
                bottleneck = max(block_rates, key=block_rates.get)

                config_name = f"{get_display_name(ig, GUARD_DISPLAY_NAMES)}|{get_display_name(llm, LLM_DISPLAY_NAMES)}|{get_display_name(og, GUARD_DISPLAY_NAMES)}"

                stats[config_name] = {
                    "ig": ig,
                    "llm": llm,
                    "og": og,
                    "asr": asr,
                    "pass_rate": pass_rate,
                    "ig_pass_rate": ig_rate,
                    "llm_pass_rate": llm_rate,
                    "og_pass_rate": og_rate,
                    "bottleneck": bottleneck,
                }

    return stats


# =============================================================================
# 可視化関数
# =============================================================================


def plot_global_block_rates(ig_rate: float, llm_rate: float, og_rate: float) -> plt.Figure:
    """Global block rate bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = ["Input Guard\n(IG)", "Target LLM\n(LLM)", "Output Guard\n(OG)"]
    rates = [ig_rate, llm_rate, og_rate]
    colors = [STAGE_COLORS["IG"], STAGE_COLORS["LLM"], STAGE_COLORS["OG"]]

    bars = ax.bar(stages, rates, color=colors, edgecolor="black", linewidth=1.5, width=0.6)

    # Highlight max (bottleneck)
    max_idx = np.argmax(rates)
    bars[max_idx].set_hatch("//")
    bars[max_idx].set_edgecolor("black")
    bars[max_idx].set_linewidth(2)

    # Display values
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Block Rate", fontsize=12)
    bottleneck_name = ["IG", "LLM", "OG"][max_idx]
    ax.set_title(f"Global Stage-wise Block Rate\n(Bottleneck: {bottleneck_name} Stage)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.annotate(
        "Bottleneck Stage",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    return fig


def plot_technique_effectiveness_heatmap(technique_stats: dict) -> plt.Figure:
    """Technique x Stage effectiveness heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))

    techniques = list(technique_stats.keys())
    stages = ["IG", "LLM", "OG"]

    data = np.zeros((len(techniques), 3))
    for i, tech in enumerate(techniques):
        data[i, 0] = technique_stats[tech]["ig_pass_rate"]
        data[i, 1] = technique_stats[tech]["llm_pass_rate"]
        data[i, 2] = technique_stats[tech]["og_pass_rate"]

    # Custom colormap (low=red, high=green)
    cmap = LinearSegmentedColormap.from_list("custom", ["#E74C3C", "#F7DC6F", "#2ECC71"])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(["Input Guard", "Target LLM", "Output Guard"])
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques)

    # Display values
    for i in range(len(techniques)):
        for j in range(len(stages)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontweight="bold")

    ax.set_title("Technique Effectiveness by Stage\n(Pass Rate)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Pass Rate", shrink=0.8)

    fig.tight_layout()
    return fig


def plot_guard_comparison(ig_stats: dict, og_stats: dict) -> plt.Figure:
    """Guard comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IG comparison
    ax = axes[0]
    guards = list(ig_stats.keys())
    display_names = [get_display_name(g, GUARD_DISPLAY_NAMES) for g in guards]
    pass_rates = [ig_stats[g]["pass_rate"] for g in guards]

    colors = plt.cm.RdYlGn(np.array(pass_rates))
    bars = ax.barh(display_names, pass_rates, color=colors, edgecolor="black")

    for bar, rate in zip(bars, pass_rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center", fontweight="bold")

    ax.set_xlabel("Pass Rate (Attack Bypass Rate)", fontsize=11)
    ax.set_title("Input Guard Comparison\n(Higher = More Vulnerable)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()

    # OG comparison
    ax = axes[1]
    guards = list(og_stats.keys())
    display_names = [get_display_name(g, GUARD_DISPLAY_NAMES) for g in guards]
    pass_rates = [og_stats[g]["pass_rate"] for g in guards]

    colors = plt.cm.RdYlGn(np.array(pass_rates))
    bars = ax.barh(display_names, pass_rates, color=colors, edgecolor="black")

    for bar, rate in zip(bars, pass_rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center", fontweight="bold")

    ax.set_xlabel("Pass Rate (Attack Bypass Rate)", fontsize=11)
    ax.set_title("Output Guard Comparison\n(Higher = More Vulnerable)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_llm_vulnerability(llm_stats: dict) -> plt.Figure:
    """LLM vulnerability profile."""
    fig, ax = plt.subplots(figsize=(10, 6))

    llms = list(llm_stats.keys())
    display_names = [get_display_name(llm, LLM_DISPLAY_NAMES) for llm in llms]
    pass_rates = [llm_stats[llm]["pass_rate"] for llm in llms]

    # Sort by vulnerability (descending)
    sorted_idx = np.argsort(pass_rates)[::-1]
    display_names = [display_names[i] for i in sorted_idx]
    pass_rates = [pass_rates[i] for i in sorted_idx]

    colors = plt.cm.RdYlGn(np.array(pass_rates))
    bars = ax.barh(display_names, pass_rates, color=colors, edgecolor="black", height=0.7)

    for bar, rate in zip(bars, pass_rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center", fontweight="bold")

    ax.set_xlabel("Pass Rate (Non-Refusal Rate)", fontsize=11)
    ax.set_title("Target LLM Vulnerability Profile\n(Higher = More Vulnerable, Does Not Refuse Attacks)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_asr_distribution(config_stats: dict) -> plt.Figure:
    """ASR distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))

    asr_values = [cfg["asr"] * 100 for cfg in config_stats.values()]

    ax.hist(asr_values, bins=20, color="#3498DB", edgecolor="black", alpha=0.8)

    # Statistics
    mean_asr = np.mean(asr_values)
    median_asr = np.median(asr_values)
    std_asr = np.std(asr_values)

    ax.axvline(mean_asr, color="#E74C3C", linestyle="--", linewidth=2, label=f"Mean: {mean_asr:.1f}%")
    ax.axvline(median_asr, color="#2ECC71", linestyle="-.", linewidth=2, label=f"Median: {median_asr:.1f}%")

    ax.set_xlabel("Attack Success Rate (%)", fontsize=11)
    ax.set_ylabel("Number of Configurations", fontsize=11)
    ax.set_title(f"Distribution of Configuration ASR\n(n={len(asr_values)}, std={std_asr:.1f}%)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")

    # Stats box
    stats_text = f"Min: {min(asr_values):.1f}%\nMax: {max(asr_values):.1f}%\nRange: {max(asr_values) - min(asr_values):.1f}%"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    return fig


def plot_bottleneck_distribution(config_stats: dict) -> plt.Figure:
    """Bottleneck distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))

    bottleneck_counts = {"IG": 0, "LLM": 0, "OG": 0}
    for cfg in config_stats.values():
        bottleneck_counts[cfg["bottleneck"]] += 1

    labels = ["Input Guard", "Target LLM", "Output Guard"]
    sizes = [bottleneck_counts["IG"], bottleneck_counts["LLM"], bottleneck_counts["OG"]]
    colors = [STAGE_COLORS["IG"], STAGE_COLORS["LLM"], STAGE_COLORS["OG"]]

    # Find max to emphasize
    max_idx = np.argmax(sizes)
    explode = [0.05 if i == max_idx else 0.02 for i in range(3)]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100 * sum(sizes))})",
        startangle=90,
        explode=explode,
        textprops={"fontsize": 11},
    )

    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title("Bottleneck Stage Distribution\n(Which stage blocks most attacks per config)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_config_ranking_top_bottom(config_stats: dict, top_n: int = 10) -> plt.Figure:
    """Configuration ranking by ASR (top vulnerable and most robust)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sorted_configs = sorted(config_stats.items(), key=lambda x: x[1]["asr"], reverse=True)

    # Top vulnerable (highest ASR)
    ax = axes[0]
    top_configs = sorted_configs[:top_n]
    names = [name for name, _ in top_configs]
    asrs = [cfg["asr"] for _, cfg in top_configs]

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(asrs)))
    bars = ax.barh(names, [r * 100 for r in asrs], color=colors, edgecolor="black")

    for bar, asr in zip(bars, asrs):
        ax.text(asr * 100 + 0.5, bar.get_y() + bar.get_height() / 2, f"{asr:.1%}", va="center", fontweight="bold")

    ax.set_xlabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Most Vulnerable Configurations", fontsize=13, fontweight="bold", color="#E74C3C")
    ax.set_xlim(0, max(asrs) * 100 * 1.15)
    ax.invert_yaxis()

    # Most robust (lowest ASR)
    ax = axes[1]
    bottom_configs = sorted_configs[-top_n:][::-1]
    names = [name for name, _ in bottom_configs]
    asrs = [cfg["asr"] for _, cfg in bottom_configs]

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(asrs)))
    bars = ax.barh(names, [r * 100 for r in asrs], color=colors, edgecolor="black")

    for bar, asr in zip(bars, asrs):
        ax.text(asr * 100 + 0.5, bar.get_y() + bar.get_height() / 2, f"{asr:.1%}", va="center", fontweight="bold")

    ax.set_xlabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Most Robust Configurations", fontsize=13, fontweight="bold", color="#2ECC71")
    ax.set_xlim(0, max(asrs) * 100 * 1.15)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_technique_by_guard_heatmap(ig_stats: dict, stage: str = "IG") -> plt.Figure:
    """Guard x Technique effectiveness heatmap."""
    fig, ax = plt.subplots(figsize=(10, 5))

    guards = list(ig_stats.keys())
    techniques = list(next(iter(ig_stats.values()))["technique_pass_rates"].keys())

    data = np.zeros((len(guards), len(techniques)))
    for i, guard in enumerate(guards):
        for j, tech in enumerate(techniques):
            data[i, j] = ig_stats[guard]["technique_pass_rates"].get(tech, 0)

    display_names = [get_display_name(g, GUARD_DISPLAY_NAMES) for g in guards]

    cmap = LinearSegmentedColormap.from_list("custom", ["#E74C3C", "#F7DC6F", "#2ECC71"])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels(techniques)
    ax.set_yticks(range(len(guards)))
    ax.set_yticklabels(display_names)

    for i in range(len(guards)):
        for j in range(len(techniques)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_title(f"{stage} Pass Rate: Guard x Technique", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Pass Rate", shrink=0.8)

    fig.tight_layout()
    return fig


def plot_technique_count_and_effectiveness(technique_stats: dict) -> plt.Figure:
    """技法の使用数と有効性の関係."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    techniques = list(technique_stats.keys())
    counts = [technique_stats[t]["count"] for t in techniques]
    effectiveness = [technique_stats[t]["overall_effectiveness"] for t in techniques]

    x = np.arange(len(techniques))
    width = 0.35

    ax1.bar(x - width / 2, counts, width, label="Count", color="#3498DB", edgecolor="black")
    ax1.set_ylabel("Count (Number of Attacks)", color="#3498DB", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#3498DB")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, effectiveness, width, label="Effectiveness", color="#2ECC71", edgecolor="black")
    ax2.set_ylabel("Overall Effectiveness", color="#2ECC71", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2ECC71")
    ax2.set_ylim(0, 1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(techniques)
    ax1.set_xlabel("Technique", fontsize=11)

    # 凡例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Technique Usage Count vs. Overall Effectiveness", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


# =============================================================================
# レポート生成
# =============================================================================


def generate_report(
    stats: DetailedStats,
    config_stats: dict,
    ig_stats: dict,
    llm_stats: dict,
    og_stats: dict,
    technique_stats: dict,
    output_dir: Path,
) -> str:
    """Markdown レポートを生成."""
    report = []
    report.append("# C3A Part 1: 詳細分析レポート\n")
    report.append(f"**生成日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # サマリー
    report.append("\n## 1. データサマリー\n")
    report.append(f"- **攻撃数**: {stats.n_attacks:,}\n")
    report.append(f"- **構成数**: {stats.n_configs}\n")
    report.append(f"- **Input Guards**: {len(ig_stats)}\n")
    report.append(f"- **Target LLMs**: {len(llm_stats)}\n")
    report.append(f"- **Output Guards**: {len(og_stats)}\n")
    report.append(f"- **技法**: {len(technique_stats)}\n")

    # グローバル統計
    report.append("\n## 2. グローバル統計\n")
    report.append("\n### 2.1 ステージ別ブロック率\n")
    report.append("| Stage | Block Rate |\n")
    report.append("|-------|------------|\n")
    report.append(f"| Input Guard (IG) | {stats.global_ig_block_rate:.1%} |\n")
    report.append(f"| **Target LLM (LLM)** | **{stats.global_llm_block_rate:.1%}** |\n")
    report.append(f"| Output Guard (OG) | {stats.global_og_block_rate:.1%} |\n")
    report.append(f"\n**ボトルネックステージ**: {stats.bottleneck_stage}\n")

    # Guard 比較
    report.append("\n## 3. Guard 比較\n")
    report.append("\n### 3.1 Input Guard\n")
    report.append("| Guard | Pass Rate | Best Technique |\n")
    report.append("|-------|-----------|----------------|\n")
    sorted_ig = sorted(ig_stats.items(), key=lambda x: x[1]["pass_rate"], reverse=True)
    for guard, s in sorted_ig:
        display_name = get_display_name(guard, GUARD_DISPLAY_NAMES)
        report.append(f"| {display_name} | {s['pass_rate']:.1%} | {s['best_technique']} ({s['best_technique_rate']:.1%}) |\n")

    report.append("\n### 3.2 Output Guard\n")
    report.append("| Guard | Pass Rate | Best Technique |\n")
    report.append("|-------|-----------|----------------|\n")
    sorted_og = sorted(og_stats.items(), key=lambda x: x[1]["pass_rate"], reverse=True)
    for guard, s in sorted_og:
        display_name = get_display_name(guard, GUARD_DISPLAY_NAMES)
        report.append(f"| {display_name} | {s['pass_rate']:.1%} | {s['best_technique']} ({s['best_technique_rate']:.1%}) |\n")

    # LLM 比較
    report.append("\n## 4. Target LLM 比較\n")
    report.append("| LLM | Pass Rate (非拒否率) | Best Technique |\n")
    report.append("|-----|---------------------|----------------|\n")
    sorted_llm = sorted(llm_stats.items(), key=lambda x: x[1]["pass_rate"], reverse=True)
    for llm, s in sorted_llm:
        display_name = get_display_name(llm, LLM_DISPLAY_NAMES)
        report.append(f"| {display_name} | {s['pass_rate']:.1%} | {s['best_technique']} ({s['best_technique_rate']:.1%}) |\n")

    # 技法分析
    report.append("\n## 5. 技法分析\n")
    report.append("| Technique | Count | IG Pass | LLM Pass | OG Pass | Overall |\n")
    report.append("|-----------|-------|---------|----------|---------|----------|\n")
    sorted_tech = sorted(technique_stats.items(), key=lambda x: x[1]["overall_effectiveness"], reverse=True)
    for tech, s in sorted_tech:
        report.append(f"| {tech} | {s['count']:,} ({s['percentage']:.1f}%) | {s['ig_pass_rate']:.1%} | {s['llm_pass_rate']:.1%} | {s['og_pass_rate']:.1%} | {s['overall_effectiveness']:.1%} |\n")

    # Configuration Ranking by ASR
    report.append("\n## 6. Configuration Ranking (by ASR)\n")
    sorted_configs = sorted(config_stats.items(), key=lambda x: x[1]["asr"], reverse=True)

    report.append("\n### 6.1 Most Vulnerable Configurations (Top 10)\n")
    report.append("| Rank | Configuration | ASR | Bottleneck |\n")
    report.append("|------|---------------|-----|------------|\n")
    for i, (name, cfg) in enumerate(sorted_configs[:10], 1):
        report.append(f"| {i} | {name} | {cfg['asr']:.1%} | {cfg['bottleneck']} |\n")

    report.append("\n### 6.2 Most Robust Configurations (Top 10)\n")
    report.append("| Rank | Configuration | ASR | Bottleneck |\n")
    report.append("|------|---------------|-----|------------|\n")
    for i, (name, cfg) in enumerate(sorted_configs[-10:][::-1], 1):
        report.append(f"| {i} | {name} | {cfg['asr']:.1%} | {cfg['bottleneck']} |\n")

    # ASR Distribution
    asr_values = [cfg["asr"] * 100 for cfg in config_stats.values()]
    report.append("\n## 7. ASR Distribution Statistics\n")
    report.append(f"- **Mean**: {np.mean(asr_values):.1f}%\n")
    report.append(f"- **Median**: {np.median(asr_values):.1f}%\n")
    report.append(f"- **Std Dev**: {np.std(asr_values):.1f}%\n")
    report.append(f"- **Min**: {np.min(asr_values):.1f}%\n")
    report.append(f"- **Max**: {np.max(asr_values):.1f}%\n")
    report.append(f"- **Range**: {np.max(asr_values) - np.min(asr_values):.1f}%\n")

    # ボトルネック分布
    bn_counts = {"IG": 0, "LLM": 0, "OG": 0}
    for cfg in config_stats.values():
        bn_counts[cfg["bottleneck"]] += 1

    report.append("\n## 8. ボトルネック分布\n")
    report.append("| Stage | Count | Percentage |\n")
    report.append("|-------|-------|------------|\n")
    total = sum(bn_counts.values())
    for stage, count in sorted(bn_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"| {stage} | {count} | {count/total:.1%} |\n")

    # 生成したグラフ
    report.append("\n## 9. 生成されたグラフ\n")
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for f in sorted(figures_dir.glob("*.png")):
            report.append(f"- `{f.name}`\n")

    return "".join(report)


# =============================================================================
# メイン
# =============================================================================


def main():
    print("=" * 60)
    print("C3A Part 1: 詳細分析")
    print("=" * 60)

    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("\n[1/6] Loading data...")
    df, components = load_data()
    print(f"  Attacks: {len(df):,}")
    print(f"  Input Guards: {len(components['input_guards'])}")
    print(f"  Target LLMs: {len(components['target_llms'])}")
    print(f"  Output Guards: {len(components['output_guards'])}")
    print(f"  Techniques: {len(components['techniques'])}")

    # 統計計算
    print("\n[2/6] Computing statistics...")
    ig_stats = compute_ig_stats(df, components["input_guards"], components["techniques"])
    llm_stats = compute_llm_stats(df, components["target_llms"], components["techniques"])
    og_stats = compute_og_stats(df, components["output_guards"], components["target_llms"], components["techniques"])
    technique_stats = compute_technique_stats(df, components)
    config_stats = compute_config_stats(df, components)

    # グローバル統計
    ig_block = np.mean([s["block_rate"] for s in ig_stats.values()])
    llm_block = np.mean([s["refusal_rate"] for s in llm_stats.values()])
    og_block = np.mean([s["block_rate"] for s in og_stats.values()])
    bottleneck = max({"IG": ig_block, "LLM": llm_block, "OG": og_block}, key=lambda k: {"IG": ig_block, "LLM": llm_block, "OG": og_block}[k])

    print(f"  Global IG block rate: {ig_block:.1%}")
    print(f"  Global LLM block rate: {llm_block:.1%}")
    print(f"  Global OG block rate: {og_block:.1%}")
    print(f"  Bottleneck: {bottleneck}")

    # 可視化
    print("\n[3/6] Generating visualizations...")
    figures = {}

    # グローバルブロック率
    fig = plot_global_block_rates(ig_block, llm_block, og_block)
    figures["01_global_block_rates"] = save_figure(fig, "01_global_block_rates", OUTPUT_DIR)
    print("  - Global block rates")

    # 技法×ステージ ヒートマップ
    fig = plot_technique_effectiveness_heatmap(technique_stats)
    figures["02_technique_effectiveness_heatmap"] = save_figure(fig, "02_technique_effectiveness_heatmap", OUTPUT_DIR)
    print("  - Technique effectiveness heatmap")

    # Guard 比較
    fig = plot_guard_comparison(ig_stats, og_stats)
    figures["03_guard_comparison"] = save_figure(fig, "03_guard_comparison", OUTPUT_DIR)
    print("  - Guard comparison")

    # LLM 脆弱性
    fig = plot_llm_vulnerability(llm_stats)
    figures["04_llm_vulnerability"] = save_figure(fig, "04_llm_vulnerability", OUTPUT_DIR)
    print("  - LLM vulnerability")

    # Pass Rate 分布
    fig = plot_asr_distribution(config_stats)
    figures["05_asr_distribution"] = save_figure(fig, "05_asr_distribution", OUTPUT_DIR)
    print("  - ASR distribution")

    # ボトルネック分布
    fig = plot_bottleneck_distribution(config_stats)
    figures["06_bottleneck_distribution"] = save_figure(fig, "06_bottleneck_distribution", OUTPUT_DIR)
    print("  - Bottleneck distribution")

    # 構成ランキング
    fig = plot_config_ranking_top_bottom(config_stats)
    figures["07_config_ranking"] = save_figure(fig, "07_config_ranking", OUTPUT_DIR)
    print("  - Config ranking")

    # Guard×技法ヒートマップ (IG)
    fig = plot_technique_by_guard_heatmap(ig_stats, "IG")
    figures["08_ig_technique_heatmap"] = save_figure(fig, "08_ig_technique_heatmap", OUTPUT_DIR)
    print("  - IG × Technique heatmap")

    # Guard×技法ヒートマップ (OG)
    fig = plot_technique_by_guard_heatmap(og_stats, "OG")
    figures["09_og_technique_heatmap"] = save_figure(fig, "09_og_technique_heatmap", OUTPUT_DIR)
    print("  - OG × Technique heatmap")

    # 技法使用数と有効性
    fig = plot_technique_count_and_effectiveness(technique_stats)
    figures["10_technique_count_effectiveness"] = save_figure(fig, "10_technique_count_effectiveness", OUTPUT_DIR)
    print("  - Technique count vs effectiveness")

    # Save statistics
    print("\n[4/6] Saving statistics...")
    asr_values = [cfg["asr"] * 100 for cfg in config_stats.values()]

    detailed_stats = DetailedStats(
        n_attacks=len(df),
        n_configs=len(config_stats),
        global_ig_block_rate=ig_block,
        global_llm_block_rate=llm_block,
        global_og_block_rate=og_block,
        bottleneck_stage=bottleneck,
        ig_stats=ig_stats,
        llm_stats=llm_stats,
        og_stats=og_stats,
        technique_stats=technique_stats,
        config_stats=config_stats,
        asr_distribution={
            "mean": float(np.mean(asr_values)),
            "median": float(np.median(asr_values)),
            "std": float(np.std(asr_values)),
            "min": float(np.min(asr_values)),
            "max": float(np.max(asr_values)),
        },
    )

    stats_path = OUTPUT_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(asdict(detailed_stats), f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {stats_path}")

    # レポート生成
    print("\n[5/6] Generating report...")
    report = generate_report(
        detailed_stats, config_stats, ig_stats, llm_stats, og_stats, technique_stats, OUTPUT_DIR
    )
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved to: {report_path}")

    # サマリー
    print("\n[6/6] Summary")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures generated: {len(figures)}")
    print(f"Report: {report_path}")
    print(f"Statistics: {stats_path}")
    print("=" * 60)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
