"""C3A Part 1: Knowledge Base 構築.

既存ラベルデータを分析し、Part 2 で使用する Knowledge Base を構築する。

Usage:
    uv run python experiments/c3a/part1_analysis.py

出力:
    experiments/c3a/results/knowledge_base.json
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict


# =============================================================================
# ヘルパー関数
# =============================================================================

def col_equals(series: pd.Series, value: int | float) -> pd.Series:
    """データ型に関わらず値を比較する.

    一部の列は文字列 ('1', '0', 'controversial')、他は数値 (1.0, 0.0) を含む。
    この関数は両方のケースを正しく処理する。

    Args:
        series: 比較する pandas Series
        value: 比較する値 (0 または 1)

    Returns:
        比較結果の boolean Series
    """
    if series.dtype == 'object':
        # 文字列列: '1' や '0' と比較
        return series == str(int(value))
    else:
        # 数値列: 直接比較
        return series == value


# =============================================================================
# 設定
# =============================================================================

DATA_PATH = Path("data/processed/en_v2_results.csv")
OUTPUT_PATH = Path("experiments/c3a/results/knowledge_base.json")

# 除外する LLM リスト（空の場合は全 LLM を使用）
EXCLUDED_LLMS: list[str] = []

# =============================================================================
# データ構造
# =============================================================================

@dataclass
class StageWeakness:
    """ステージ別脆弱性."""
    pass_rate: float
    block_rate: float
    best_technique: str
    best_technique_pass_rate: float


@dataclass
class ConfigProfile:
    """構成プロファイル."""
    config_name: str
    ig: str
    llm: str
    og: str
    pass_rate: float  # 全ステージ通過率
    ig_pass_rate: float
    llm_pass_rate: float
    og_pass_rate: float
    bottleneck: str  # "IG", "LLM", or "OG"
    stage_weakness: dict[str, StageWeakness] = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    """C3A 用 Knowledge Base."""
    # メタ情報
    n_attacks: int
    n_configs: int
    input_guards: list[str]
    target_llms: list[str]
    output_guards: list[str]
    techniques: list[str]

    # グローバル統計
    global_ig_block_rate: float
    global_llm_block_rate: float
    global_og_block_rate: float
    bottleneck_stage: str

    # ステージ別・技法別有効性 (EffectivenessMap)
    # effectiveness[stage][technique] = pass_rate
    effectiveness: dict[str, dict[str, float]] = field(default_factory=dict)

    # ステージ別・技法別・Guard別有効性（詳細版）
    # effectiveness_by_guard[stage][guard][technique] = pass_rate
    effectiveness_by_guard: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # 構成別プロファイル (ConfigWeaknessProfile)
    # config_profiles[config_name] = ConfigProfile
    config_profiles: dict[str, dict] = field(default_factory=dict)

    # 成功事例インデックス (SuccessExampleIndex)
    # stage_pass_indices[config_name][stage] = list of attack indices
    stage_pass_indices: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    # プロンプトテキスト（Few-shot用）
    prompts: list[str] = field(default_factory=list)


# =============================================================================
# KB 構築関数
# =============================================================================

def load_data() -> tuple[pd.DataFrame, dict]:
    """データを読み込み、コンポーネントを抽出する."""
    df = pd.read_csv(DATA_PATH)

    cols = df.columns.tolist()

    # Input Guards
    ig_cols = [c for c in cols if c.startswith("input_guard_") and c.endswith("_is_safe")]
    input_guards = [c.replace("input_guard_", "").replace("_is_safe", "") for c in ig_cols]

    # Target LLMs（gemini除外）
    llm_cols = [c for c in cols if c.startswith("llm_") and c.endswith("_is_refusal")]
    target_llms = [c.replace("llm_", "").replace("_is_refusal", "") for c in llm_cols]
    target_llms = [llm for llm in target_llms if llm not in EXCLUDED_LLMS]

    # Output Guards
    first_llm = target_llms[0]
    og_cols = [c for c in cols if c.startswith("output_guard_") and f"_on_{first_llm}_is_safe" in c]
    output_guards = [c.replace("output_guard_", "").replace(f"_on_{first_llm}_is_safe", "") for c in og_cols]

    # Techniques
    tech_cols = [c for c in cols if c.startswith("taxonomy_technique_")]
    techniques = [c.replace("taxonomy_technique_", "") for c in tech_cols]

    components = {
        "input_guards": input_guards,
        "target_llms": target_llms,
        "output_guards": output_guards,
        "techniques": techniques,
    }

    return df, components


def compute_global_stats(df: pd.DataFrame, components: dict) -> dict:
    """グローバル統計を計算."""
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]

    # IG ブロック率
    ig_rates = []
    for ig in input_guards:
        col = f"input_guard_{ig}_is_safe"
        ig_rates.append(col_equals(df[col], 0).mean())
    avg_ig_block = np.mean(ig_rates)

    # LLM 拒否率
    llm_rates = []
    for llm in target_llms:
        col = f"llm_{llm}_is_refusal"
        llm_rates.append(col_equals(df[col], 1).mean())
    avg_llm_block = np.mean(llm_rates)

    # OG ブロック率
    og_rates = []
    for og in output_guards:
        rates = []
        for llm in target_llms:
            col = f"output_guard_{og}_on_{llm}_is_safe"
            if col in df.columns:
                rates.append(col_equals(df[col], 0).mean())
        if rates:
            og_rates.append(np.mean(rates))
    avg_og_block = np.mean(og_rates)

    # ボトルネック
    stages = {"IG": avg_ig_block, "LLM": avg_llm_block, "OG": avg_og_block}
    bottleneck = max(stages, key=stages.get)

    return {
        "global_ig_block_rate": avg_ig_block,
        "global_llm_block_rate": avg_llm_block,
        "global_og_block_rate": avg_og_block,
        "bottleneck_stage": bottleneck,
    }


def compute_effectiveness_map(df: pd.DataFrame, components: dict) -> dict:
    """ステージ別・技法別有効性マップを計算."""
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]
    techniques = components["techniques"]

    effectiveness = {"IG": {}, "LLM": {}, "OG": {}}

    for tech in techniques:
        tech_col = f"taxonomy_technique_{tech}"
        tech_mask = df[tech_col] == 1

        if tech_mask.sum() == 0:
            continue

        tech_df = df[tech_mask]

        # IG 通過率
        ig_rates = []
        for ig in input_guards:
            col = f"input_guard_{ig}_is_safe"
            ig_rates.append(col_equals(tech_df[col], 1).mean())
        effectiveness["IG"][tech] = float(np.mean(ig_rates))

        # LLM 非拒否率
        llm_rates = []
        for llm in target_llms:
            col = f"llm_{llm}_is_refusal"
            llm_rates.append(col_equals(tech_df[col], 0).mean())
        effectiveness["LLM"][tech] = float(np.mean(llm_rates))

        # OG 通過率
        og_rates = []
        for og in output_guards:
            rates = []
            for llm in target_llms:
                col = f"output_guard_{og}_on_{llm}_is_safe"
                if col in df.columns:
                    rates.append(col_equals(tech_df[col], 1).mean())
            if rates:
                og_rates.append(np.mean(rates))
        effectiveness["OG"][tech] = float(np.mean(og_rates))

    return effectiveness


def compute_effectiveness_by_guard(df: pd.DataFrame, components: dict) -> dict:
    """Guard/LLM 別の技法有効性を計算."""
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]
    techniques = components["techniques"]

    result = {
        "IG": {},
        "LLM": {},
        "OG": {},
    }

    # IG × Tech
    for ig in input_guards:
        result["IG"][ig] = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() == 0:
                continue
            tech_df = df[tech_mask]
            col = f"input_guard_{ig}_is_safe"
            result["IG"][ig][tech] = float(col_equals(tech_df[col], 1).mean())

    # LLM × Tech
    for llm in target_llms:
        result["LLM"][llm] = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() == 0:
                continue
            tech_df = df[tech_mask]
            col = f"llm_{llm}_is_refusal"
            result["LLM"][llm][tech] = float(col_equals(tech_df[col], 0).mean())

    # OG × Tech (LLM平均)
    for og in output_guards:
        result["OG"][og] = {}
        for tech in techniques:
            tech_col = f"taxonomy_technique_{tech}"
            tech_mask = df[tech_col] == 1
            if tech_mask.sum() == 0:
                continue
            tech_df = df[tech_mask]
            rates = []
            for llm in target_llms:
                col = f"output_guard_{og}_on_{llm}_is_safe"
                if col in df.columns:
                    rates.append(col_equals(tech_df[col], 1).mean())
            if rates:
                result["OG"][og][tech] = float(np.mean(rates))

    return result


def compute_config_profiles(df: pd.DataFrame, components: dict) -> dict:
    """構成別プロファイルを計算."""
    input_guards = components["input_guards"]
    target_llms = components["target_llms"]
    output_guards = components["output_guards"]
    techniques = components["techniques"]

    profiles = {}

    for ig in input_guards:
        for llm in target_llms:
            for og in output_guards:
                ig_col = f"input_guard_{ig}_is_safe"
                refusal_col = f"llm_{llm}_is_refusal"
                og_col = f"output_guard_{og}_on_{llm}_is_safe"

                if og_col not in df.columns:
                    continue

                # 各ステージ通過
                ig_pass = col_equals(df[ig_col], 1)
                llm_pass = col_equals(df[refusal_col], 0)
                og_pass = col_equals(df[og_col], 1)

                # 通過率
                ig_pass_rate = float(ig_pass.mean())
                llm_pass_rate = float(llm_pass.mean())
                og_pass_rate = float(og_pass.mean())

                # 全ステージ通過率
                all_pass = ig_pass & llm_pass & og_pass
                pass_rate = float(all_pass.mean())

                # ボトルネック（ブロック率が最も高いステージ）
                block_rates = {
                    "IG": 1 - ig_pass_rate,
                    "LLM": 1 - llm_pass_rate,
                    "OG": 1 - og_pass_rate,
                }
                bottleneck = max(block_rates, key=block_rates.get)

                # ステージ別・技法別有効性
                stage_weakness = {}
                for stage, labels, stage_pass_rate in [
                    ("IG", ig_pass, ig_pass_rate),
                    ("LLM", llm_pass, llm_pass_rate),
                    ("OG", og_pass, og_pass_rate),
                ]:
                    best_tech = techniques[0]
                    best_rate = 0.0
                    for tech in techniques:
                        tech_col = f"taxonomy_technique_{tech}"
                        tech_mask = df[tech_col] == 1
                        if tech_mask.sum() > 0:
                            rate = float(labels[tech_mask].mean())
                            if rate > best_rate:
                                best_rate = rate
                                best_tech = tech

                    stage_weakness[stage] = {
                        "pass_rate": stage_pass_rate,
                        "block_rate": 1 - stage_pass_rate,
                        "best_technique": best_tech,
                        "best_technique_pass_rate": best_rate,
                    }

                # 短縮名
                ig_short = ig.split("_")[-1]
                llm_short = llm.split("_")[-1] if "_" in llm else llm
                og_short = og.split("_")[-1]
                config_name = f"{ig_short}|{llm_short}|{og_short}"

                profiles[config_name] = {
                    "config_name": config_name,
                    "ig": ig,
                    "llm": llm,
                    "og": og,
                    "pass_rate": pass_rate,
                    "ig_pass_rate": ig_pass_rate,
                    "llm_pass_rate": llm_pass_rate,
                    "og_pass_rate": og_pass_rate,
                    "bottleneck": bottleneck,
                    "stage_weakness": stage_weakness,
                }

    return profiles


def compute_stage_pass_indices(df: pd.DataFrame, components: dict, config_profiles: dict) -> dict:
    """ステージ通過インデックスを計算."""
    result = {}

    for config_name, profile in config_profiles.items():
        ig = profile["ig"]
        llm = profile["llm"]
        og = profile["og"]

        ig_col = f"input_guard_{ig}_is_safe"
        refusal_col = f"llm_{llm}_is_refusal"
        og_col = f"output_guard_{og}_on_{llm}_is_safe"

        ig_pass = col_equals(df[ig_col], 1)
        llm_pass = col_equals(df[refusal_col], 0)
        og_pass = col_equals(df[og_col], 1)

        result[config_name] = {
            "IG": np.where(ig_pass)[0].tolist(),
            "LLM": np.where(ig_pass & llm_pass)[0].tolist(),
            "OG": np.where(ig_pass & llm_pass & og_pass)[0].tolist(),
        }

    return result


def build_knowledge_base() -> KnowledgeBase:
    """KB を構築."""
    print("Loading data...")
    df, components = load_data()

    print(f"Data: {len(df):,} attacks")
    print(f"Input Guards: {len(components['input_guards'])}")
    print(f"Target LLMs: {len(components['target_llms'])}")
    print(f"Output Guards: {len(components['output_guards'])}")
    print(f"Techniques: {len(components['techniques'])}")

    print("\nComputing global stats...")
    global_stats = compute_global_stats(df, components)

    print("Computing effectiveness map...")
    effectiveness = compute_effectiveness_map(df, components)

    print("Computing effectiveness by guard...")
    effectiveness_by_guard = compute_effectiveness_by_guard(df, components)

    print("Computing config profiles...")
    config_profiles = compute_config_profiles(df, components)

    print("Computing stage pass indices...")
    stage_pass_indices = compute_stage_pass_indices(df, components, config_profiles)

    # プロンプトテキスト
    prompts = df["prompt_text_en"].fillna("").tolist()

    kb = KnowledgeBase(
        n_attacks=len(df),
        n_configs=len(config_profiles),
        input_guards=components["input_guards"],
        target_llms=components["target_llms"],
        output_guards=components["output_guards"],
        techniques=components["techniques"],
        global_ig_block_rate=global_stats["global_ig_block_rate"],
        global_llm_block_rate=global_stats["global_llm_block_rate"],
        global_og_block_rate=global_stats["global_og_block_rate"],
        bottleneck_stage=global_stats["bottleneck_stage"],
        effectiveness=effectiveness,
        effectiveness_by_guard=effectiveness_by_guard,
        config_profiles=config_profiles,
        stage_pass_indices=stage_pass_indices,
        prompts=prompts,
    )

    return kb


def save_knowledge_base(kb: KnowledgeBase, path: Path) -> None:
    """KB を JSON に保存."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(kb)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nKnowledge Base saved to: {path}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


def compute_and_save_embeddings(
    prompts: list[str],
    output_path: Path,
    batch_size: int = 64,
) -> None:
    """プロンプトの埋め込みを計算して保存.

    Args:
        prompts: プロンプトリスト
        output_path: 出力パス（.npy）
        batch_size: バッチサイズ
    """
    from src.c3a.knowledge.embeddings import (
        compute_embeddings_batch,
        save_embeddings,
    )

    print(f"\nComputing embeddings for {len(prompts):,} prompts...")
    embeddings = compute_embeddings_batch(
        prompts,
        batch_size=batch_size,
        show_progress=True,
    )
    save_embeddings(embeddings, output_path)
    print(f"Embeddings saved to: {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    print("="*60)
    print("C3A Knowledge Base Builder")
    print("="*60)

    kb = build_knowledge_base()

    print("\n" + "="*60)
    print("Knowledge Base Summary")
    print("="*60)
    print(f"Attacks: {kb.n_attacks:,}")
    print(f"Configs: {kb.n_configs}")
    print(f"\nGlobal Bottleneck: {kb.bottleneck_stage}")
    print(f"  IG block rate:  {kb.global_ig_block_rate:.2%}")
    print(f"  LLM block rate: {kb.global_llm_block_rate:.2%}")
    print(f"  OG block rate:  {kb.global_og_block_rate:.2%}")

    print("\nEffectiveness Map (stage × technique):")
    for stage in ["IG", "LLM", "OG"]:
        print(f"  {stage}:", end=" ")
        for tech, rate in kb.effectiveness[stage].items():
            print(f"{tech}={rate:.1%}", end=" ")
        print()

    # 最脆弱/最堅牢構成
    sorted_configs = sorted(
        kb.config_profiles.values(),
        key=lambda x: x["pass_rate"],
        reverse=True
    )

    print("\nMost Vulnerable Configs (Top 5):")
    for profile in sorted_configs[:5]:
        print(f"  {profile['pass_rate']:.2%} | {profile['config_name']}")

    print("\nMost Robust Configs (Top 5):")
    for profile in sorted_configs[-5:]:
        print(f"  {profile['pass_rate']:.2%} | {profile['config_name']}")

    save_knowledge_base(kb, OUTPUT_PATH)

    # 埋め込み計算（類似度検索用）
    if kb.prompts:
        embeddings_path = OUTPUT_PATH.parent / "prompt_embeddings.npy"
        compute_and_save_embeddings(kb.prompts, embeddings_path)


if __name__ == "__main__":
    main()
