"""データローダー.

en_results.csv を読み込み、AttackDataset を構築する。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.c3a.data import columns
from src.c3a.types import OUTPUT_GUARDS, TARGET_LLMS


@dataclass
class AttackDataset:
    """攻撃データセット（Part 1 分析用）.

    Attributes:
        uids: 攻撃ID配列 (N,)
        prompts: プロンプトテキスト配列 (N,)
        techniques: テクニックフラグ配列 (N, 5) - [LEX, MSYN, SEM, PRAG, ORTH]
        intents: インテント配列 (N,)
        ig_labels: Input Guard 結果配列 (N, 5) - 1=safe(通過), 0=unsafe(ブロック)
        llm_labels: Target LLM 結果配列 (N, 9) - 1=通過, 0=拒否/ブロック
        og_labels: Output Guard 結果配列 (N, 9, 5) - [LLM, OG] の組み合わせ
        harmful_labels: is_harmful ラベル (N, 9) - 1=有害, 0=無害, NaN=未評価
    """

    uids: NDArray[np.str_]
    prompts: NDArray[np.str_]
    techniques: NDArray[np.int8]
    intents: NDArray[np.str_]
    ig_labels: NDArray[np.int8]
    llm_labels: NDArray[np.int8]
    og_labels: NDArray[np.int8]
    harmful_labels: NDArray[np.float64]  # NaN を保持するため float

    @property
    def n_attacks(self) -> int:
        """攻撃数を返す."""
        return len(self.uids)

    @property
    def n_input_guards(self) -> int:
        """Input Guard 数を返す."""
        return self.ig_labels.shape[1]

    @property
    def n_target_llms(self) -> int:
        """Target LLM 数を返す."""
        return self.llm_labels.shape[1]

    @property
    def n_output_guards(self) -> int:
        """Output Guard 数を返す."""
        return self.og_labels.shape[2]


def load_dataset(path: str | Path) -> AttackDataset:
    """CSV を読み込み AttackDataset を構築する.

    Args:
        path: CSV ファイルのパス

    Returns:
        AttackDataset インスタンス

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 必要なカラムが存在しない場合
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # 基本情報
    uids = df[columns.UID].values.astype(np.str_)
    prompts = df[columns.PROMPT_TEXT_EN].values.astype(np.str_)
    intents = df[columns.INTENT].values.astype(np.str_)

    # テクニックフラグ (N, 5)
    techniques = np.column_stack(
        [df[col].values.astype(np.int8) for col in columns.TECHNIQUE_COLUMNS]
    )

    # Input Guard ラベル (N, 5)
    ig_labels = np.column_stack(
        [df[col].values.astype(np.int8) for col in columns.INPUT_GUARD_COLUMNS]
    )

    # Target LLM ラベル (N, 9)
    # responded=1 かつ is_refusal=0 を「通過」とする
    n_attacks = len(df)
    n_llms = len(TARGET_LLMS)
    llm_labels = np.zeros((n_attacks, n_llms), dtype=np.int8)

    for llm_idx, llm_name in enumerate(TARGET_LLMS):
        responded_col = columns.get_llm_responded_column(llm_name)
        refusal_col = columns.get_llm_refusal_column(llm_name)

        if responded_col in df.columns and refusal_col in df.columns:
            responded = df[responded_col].fillna(0).values.astype(np.int8)
            refusal = df[refusal_col].fillna(1).values.astype(np.int8)
            # 通過 = responded かつ 非拒否
            llm_labels[:, llm_idx] = ((responded == 1) & (refusal == 0)).astype(np.int8)
        elif responded_col in df.columns:
            # refusal カラムがなければ responded のみで判定
            llm_labels[:, llm_idx] = df[responded_col].fillna(0).values.astype(np.int8)

    # Output Guard ラベル (N, 9, 5)
    n_ogs = len(OUTPUT_GUARDS)
    og_labels = np.zeros((n_attacks, n_llms, n_ogs), dtype=np.int8)

    for llm_idx, llm_name in enumerate(TARGET_LLMS):
        for og_idx, og_name in enumerate(OUTPUT_GUARDS):
            col_name = columns.get_output_guard_column(og_name, llm_name)
            if col_name in df.columns:
                og_labels[:, llm_idx, og_idx] = df[col_name].fillna(0).values.astype(np.int8)

    # is_harmful ラベル (N, 9)
    harmful_labels = np.full((n_attacks, n_llms), np.nan, dtype=np.float64)

    for llm_idx, llm_name in enumerate(TARGET_LLMS):
        harmful_col = columns.get_multijudge_harmful_column(llm_name)
        if harmful_col in df.columns:
            harmful_labels[:, llm_idx] = df[harmful_col].values.astype(np.float64)

    return AttackDataset(
        uids=uids,
        prompts=prompts,
        techniques=techniques,
        intents=intents,
        ig_labels=ig_labels,
        llm_labels=llm_labels,
        og_labels=og_labels,
        harmful_labels=harmful_labels,
    )


def get_stage_result_from_labels(
    ig_labels: NDArray[np.int8],
    llm_labels: NDArray[np.int8],
    og_labels: NDArray[np.int8],
    ig_idx: int,
    llm_idx: int,
    og_idx: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """ラベルデータから各ステージの結果を取得する.

    Args:
        ig_labels: Input Guard ラベル (N, 5)
        llm_labels: Target LLM ラベル (N, 9)
        og_labels: Output Guard ラベル (N, 9, 5)
        ig_idx: Input Guard インデックス
        llm_idx: Target LLM インデックス
        og_idx: Output Guard インデックス

    Returns:
        (ig_passed, llm_passed, og_passed) のタプル配列
    """
    ig_passed = ig_labels[:, ig_idx].astype(bool)
    llm_passed = llm_labels[:, llm_idx].astype(bool)
    og_passed = og_labels[:, llm_idx, og_idx].astype(bool)
    return ig_passed, llm_passed, og_passed


def compute_pass_rate(
    dataset: AttackDataset,
    ig_idx: int,
    llm_idx: int,
    og_idx: int,
) -> float:
    """特定の構成に対する全ステージ通過率を計算する.

    Args:
        dataset: AttackDataset
        ig_idx: Input Guard インデックス
        llm_idx: Target LLM インデックス
        og_idx: Output Guard インデックス

    Returns:
        全ステージ通過率 (0.0 - 1.0)
    """
    ig_passed, llm_passed, og_passed = get_stage_result_from_labels(
        dataset.ig_labels,
        dataset.llm_labels,
        dataset.og_labels,
        ig_idx,
        llm_idx,
        og_idx,
    )

    # 全ステージ通過
    all_passed = ig_passed & llm_passed & og_passed
    return float(all_passed.mean())


def compute_stage_block_rates(
    dataset: AttackDataset,
    ig_idx: int,
    llm_idx: int,
    og_idx: int,
) -> tuple[float, float, float]:
    """特定の構成に対する各ステージのブロック率を計算する.

    Args:
        dataset: AttackDataset
        ig_idx: Input Guard インデックス
        llm_idx: Target LLM インデックス
        og_idx: Output Guard インデックス

    Returns:
        (ig_block_rate, llm_block_rate, og_block_rate)
    """
    ig_passed, llm_passed, og_passed = get_stage_result_from_labels(
        dataset.ig_labels,
        dataset.llm_labels,
        dataset.og_labels,
        ig_idx,
        llm_idx,
        og_idx,
    )

    # IG ブロック率
    ig_block = (~ig_passed).mean()

    # LLM ブロック率（IG 通過者のうち）
    ig_passed_count = ig_passed.sum()
    if ig_passed_count > 0:
        llm_block = (~llm_passed[ig_passed]).mean()
    else:
        llm_block = 0.0

    # OG ブロック率（LLM 通過者のうち）
    llm_passed_full = ig_passed & llm_passed
    llm_passed_count = llm_passed_full.sum()
    if llm_passed_count > 0:
        og_block = (~og_passed[llm_passed_full]).mean()
    else:
        og_block = 0.0

    return float(ig_block), float(llm_block), float(og_block)
