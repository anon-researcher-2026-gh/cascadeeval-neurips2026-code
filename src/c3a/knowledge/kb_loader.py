"""Part 1 JSON 形式の Knowledge Base ローダー.

build_knowledge_base.py で構築した JSON を読み込み、
Part 2 の攻撃生成ガイダンスを提供する。
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.c3a.types import (
    DefenseConfig,
    Stage,
    TechniqueName,
    TECHNIQUES,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class StrategyGuidance:
    """攻撃生成のためのガイダンス情報.

    Attributes:
        best_technique: 推奨技法
        technique_effectiveness: その技法の有効性
        stage_pass_rate: 対象ステージの通過率
        stage_block_rate: 対象ステージのブロック率
        example_prompts: Few-shot 用のプロンプト例
        bottleneck_stage: 構成のボトルネックステージ
    """

    best_technique: TechniqueName
    technique_effectiveness: float
    stage_pass_rate: float
    stage_block_rate: float
    example_prompts: list[str]
    bottleneck_stage: Stage


@dataclass
class Part1KnowledgeBase:
    """Part 1 分析結果から構築された Knowledge Base.

    build_knowledge_base.py で生成された JSON を読み込み、
    攻撃生成ガイダンスを提供する。

    Attributes:
        n_attacks: 分析に使用した攻撃数
        n_configs: 構成数
        input_guards: Input Guard リスト
        target_llms: Target LLM リスト
        output_guards: Output Guard リスト
        techniques: 技法リスト
        bottleneck_stage: グローバルボトルネックステージ
        effectiveness: ステージ別・技法別有効性
        effectiveness_by_guard: Guard 別・技法別有効性
        config_profiles: 構成別プロファイル
        stage_pass_indices: ステージ通過インデックス
        prompts: プロンプトテキストリスト
    """

    n_attacks: int
    n_configs: int
    input_guards: list[str]
    target_llms: list[str]
    output_guards: list[str]
    techniques: list[str]
    bottleneck_stage: str
    effectiveness: dict[str, dict[str, float]]
    effectiveness_by_guard: dict[str, dict[str, dict[str, float]]]
    config_profiles: dict[str, dict]
    stage_pass_indices: dict[str, dict[str, list[int]]]
    prompts: list[str] = field(default_factory=list)
    _embeddings: "NDArray[np.float32] | None" = field(default=None, repr=False)
    _goal_embedding_cache: dict[str, "NDArray[np.float32]"] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def load(cls, path: str | Path) -> "Part1KnowledgeBase":
        """JSON ファイルから KB を読み込む.

        Args:
            path: JSON ファイルパス

        Returns:
            Part1KnowledgeBase インスタンス
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge Base not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            n_attacks=data["n_attacks"],
            n_configs=data["n_configs"],
            input_guards=data["input_guards"],
            target_llms=data["target_llms"],
            output_guards=data["output_guards"],
            techniques=data["techniques"],
            bottleneck_stage=data["bottleneck_stage"],
            effectiveness=data["effectiveness"],
            effectiveness_by_guard=data["effectiveness_by_guard"],
            config_profiles=data["config_profiles"],
            stage_pass_indices=data["stage_pass_indices"],
            prompts=data.get("prompts", []),
        )

    @classmethod
    def load_with_embeddings(
        cls,
        kb_path: str | Path,
        embeddings_path: str | Path | None = None,
    ) -> "Part1KnowledgeBase":
        """KB と埋め込みを読み込む.

        Args:
            kb_path: Knowledge Base JSON パス
            embeddings_path: 埋め込み .npy パス（None の場合は kb_path から推測）

        Returns:
            Part1KnowledgeBase インスタンス（埋め込み付き）
        """
        kb = cls.load(kb_path)

        # 埋め込みパスを推測
        if embeddings_path is None:
            kb_path = Path(kb_path)
            embeddings_path = kb_path.parent / "prompt_embeddings.npy"

        embeddings_path = Path(embeddings_path)
        if embeddings_path.exists():
            kb._embeddings = np.load(embeddings_path)
            logger.info(
                f"Loaded embeddings: {kb._embeddings.shape} from {embeddings_path}"
            )
        else:
            logger.warning(f"Embeddings not found: {embeddings_path}")

        return kb

    def has_embeddings(self) -> bool:
        """埋め込みが読み込まれているか."""
        return self._embeddings is not None

    def _get_goal_embedding(self, goal: str) -> "NDArray[np.float32]":
        """Goal の埋め込みを取得（キャッシュ付き）."""
        if goal in self._goal_embedding_cache:
            return self._goal_embedding_cache[goal]

        from src.c3a.knowledge.embeddings import compute_embedding

        embedding = compute_embedding(goal)
        self._goal_embedding_cache[goal] = embedding
        return embedding

    def _normalize_stage_key(self, stage: Stage) -> str:
        """Stage を stage_pass_indices のキーに変換する.

        PASSED_GUARDS ステージは KB に存在しないため OG を使用。
        （OG 通過 = 全ステージ通過）
        """
        return "OG" if stage == Stage.PASSED_GUARDS else stage.value

    def get_similar_example_prompts(
        self,
        config: DefenseConfig,
        stage: Stage,
        goal: str,
        k: int = 3,
    ) -> list[str]:
        """Goal と類似したプロンプト例を取得.

        埋め込みベースの類似度検索を行う。
        埋め込みがない場合はランダム選択にフォールバック。

        Args:
            config: 防御構成
            stage: 対象ステージ
            goal: 攻撃目標（類似度計算に使用）
            k: 取得数

        Returns:
            類似度の高い順にソートされたプロンプトリスト
        """
        if not self.prompts:
            return []

        # 埋め込みがない場合はランダム選択
        if self._embeddings is None:
            return self.get_example_prompts(config, stage, k)

        config_key = self._get_config_key(config)
        if config_key is None:
            return []

        stage_key = self._normalize_stage_key(stage)
        if config_key not in self.stage_pass_indices:
            return []

        indices = self.stage_pass_indices[config_key].get(stage_key, [])
        if not indices:
            return []

        # フィルタされたインデックスの埋め込みを取得
        indices_arr = np.array(indices)
        valid_mask = indices_arr < len(self._embeddings)
        indices_arr = indices_arr[valid_mask]

        if len(indices_arr) == 0:
            return []

        candidate_embeddings = self._embeddings[indices_arr]

        # Goal の埋め込みを計算（キャッシュ）
        goal_embedding = self._get_goal_embedding(goal)

        # 類似度計算（正規化済みなら内積 = コサイン類似度）
        similarities = candidate_embeddings @ goal_embedding

        # 類似度の高い順にソート
        sorted_local_indices = np.argsort(similarities)[::-1]

        # 重複を排除しながら k 件取得
        result: list[str] = []
        seen_texts: set[str] = set()

        for local_idx in sorted_local_indices:
            if len(result) >= k:
                break
            global_idx = indices_arr[local_idx]
            if 0 <= global_idx < len(self.prompts):
                prompt_text = self.prompts[global_idx]
                if prompt_text not in seen_texts:
                    seen_texts.add(prompt_text)
                    result.append(prompt_text)

        return result

    def _get_config_key(self, config: DefenseConfig) -> str | None:
        """DefenseConfig から config_profiles のキーを取得する.

        構成名形式: "wildguard|gpt-4o|wildguard" または
        "allenai_wildguard|gpt-4o|allenai_wildguard"
        """
        # 短縮名で試す
        short_key = config.name
        if short_key in self.config_profiles:
            return short_key

        # フルネームで試す
        full_key = config.full_name
        if full_key in self.config_profiles:
            return full_key

        # 部分一致で検索（最長一致を優先）
        def match_score(key: str) -> int:
            """マッチの品質スコア（高いほど良い）."""
            parts = key.split("|")
            if len(parts) != 3:
                return -1

            ig_match = parts[0] in config.input_guard or config.input_guard in parts[0]
            llm_match = parts[1] == config.target_llm  # LLM は完全一致を優先
            og_match = parts[2] in config.output_guard or config.output_guard in parts[2]

            if not (ig_match and og_match):
                return -1

            # LLM 完全一致なら高スコア
            if llm_match:
                return 1000 + len(parts[1])

            # LLM 部分一致（長い方を優先）
            if parts[1] in config.target_llm or config.target_llm in parts[1]:
                return len(parts[1])

            return -1

        best_key = None
        best_score = -1

        for key in self.config_profiles:
            score = match_score(key)
            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _get_guard_key(self, guard_name: str, guard_dict: dict) -> str | None:
        """Guard 名から effectiveness_by_guard のキーを取得する."""
        if guard_name in guard_dict:
            return guard_name

        # 部分一致で検索
        for key in guard_dict:
            if guard_name in key or key in guard_name:
                return key

        return None

    def get_best_technique(
        self,
        stage: Stage,
        config: DefenseConfig | None = None,
    ) -> TechniqueName:
        """指定ステージで最も有効な技法を返す.

        Args:
            stage: 対象ステージ
            config: 構成（None の場合はグローバル有効性を使用）

        Returns:
            最も有効な技法名
        """
        stage_key = self._normalize_stage_key(stage)

        if stage == Stage.PASSED_GUARDS:
            # 全ステージ平均で最良のものを返す
            avg_effectiveness = {}
            for tech in TECHNIQUES:
                total = 0.0
                count = 0
                for s in ["IG", "LLM", "OG"]:
                    if s in self.effectiveness and tech in self.effectiveness[s]:
                        total += self.effectiveness[s][tech]
                        count += 1
                avg_effectiveness[tech] = total / count if count > 0 else 0.0
            return max(avg_effectiveness, key=avg_effectiveness.get)  # type: ignore

        # 構成指定時は構成別の Guard 有効性を使用
        if config is not None and stage_key in self.effectiveness_by_guard:
            guard_dict = self.effectiveness_by_guard[stage_key]

            # 対応する Guard を特定
            if stage == Stage.INPUT_GUARD:
                guard_name = config.input_guard
            elif stage == Stage.TARGET_LLM:
                guard_name = config.target_llm
            else:  # OUTPUT_GUARD
                guard_name = config.output_guard

            guard_key = self._get_guard_key(guard_name, guard_dict)
            if guard_key:
                tech_eff = guard_dict[guard_key]
                return max(tech_eff, key=tech_eff.get)  # type: ignore

        # フォールバック: グローバル有効性を使用
        if stage_key in self.effectiveness:
            tech_eff = self.effectiveness[stage_key]
            return max(tech_eff, key=tech_eff.get)  # type: ignore

        return "SEM"  # デフォルト

    def get_effectiveness(
        self,
        stage: Stage,
        technique: TechniqueName,
        config: DefenseConfig | None = None,
    ) -> float:
        """指定ステージ・技法の有効性を返す.

        Args:
            stage: 対象ステージ
            technique: 技法名
            config: 構成（None の場合はグローバル有効性を使用）

        Returns:
            有効性（0.0 - 1.0）
        """
        if stage == Stage.PASSED_GUARDS:
            return 0.5  # デフォルト

        stage_key = self._normalize_stage_key(stage)

        # 構成指定時は Guard 別有効性を使用
        if config is not None and stage_key in self.effectiveness_by_guard:
            guard_dict = self.effectiveness_by_guard[stage_key]

            if stage == Stage.INPUT_GUARD:
                guard_name = config.input_guard
            elif stage == Stage.TARGET_LLM:
                guard_name = config.target_llm
            else:
                guard_name = config.output_guard

            guard_key = self._get_guard_key(guard_name, guard_dict)
            if guard_key and technique in guard_dict[guard_key]:
                return guard_dict[guard_key][technique]

        # フォールバック
        if stage_key in self.effectiveness and technique in self.effectiveness[stage_key]:
            return self.effectiveness[stage_key][technique]

        return 0.5  # デフォルト

    def get_stage_pass_rate(
        self,
        config: DefenseConfig,
        stage: Stage,
    ) -> float:
        """構成・ステージの通過率を返す.

        Args:
            config: 防御構成
            stage: 対象ステージ

        Returns:
            通過率（0.0 - 1.0）
        """
        config_key = self._get_config_key(config)
        if config_key is None:
            return 0.5

        profile = self.config_profiles.get(config_key, {})
        stage_weakness = profile.get("stage_weakness", {})
        stage_key = self._normalize_stage_key(stage)

        if stage_key in stage_weakness:
            return stage_weakness[stage_key].get("pass_rate", 0.5)

        return 0.5

    def get_bottleneck_stage(self, config: DefenseConfig) -> Stage:
        """構成のボトルネックステージを返す.

        Args:
            config: 防御構成

        Returns:
            ボトルネックステージ
        """
        config_key = self._get_config_key(config)
        if config_key is None:
            return Stage.TARGET_LLM  # グローバルデフォルト

        profile = self.config_profiles.get(config_key, {})
        bottleneck = profile.get("bottleneck_stage", "LLM")

        stage_map = {"IG": Stage.INPUT_GUARD, "LLM": Stage.TARGET_LLM, "OG": Stage.OUTPUT_GUARD}
        return stage_map.get(bottleneck, Stage.TARGET_LLM)

    def get_example_prompts(
        self,
        config: DefenseConfig,
        stage: Stage,
        k: int = 3,
        seed: int | None = None,
    ) -> list[str]:
        """指定ステージを通過した攻撃プロンプト例を取得する.

        Args:
            config: 防御構成
            stage: 対象ステージ
            k: 取得する例の数
            seed: ランダムシード

        Returns:
            プロンプトテキストのリスト
        """
        if not self.prompts:
            return []

        config_key = self._get_config_key(config)
        if config_key is None:
            return []

        stage_key = self._normalize_stage_key(stage)
        if config_key not in self.stage_pass_indices:
            return []

        indices = self.stage_pass_indices[config_key].get(stage_key, [])
        if not indices:
            return []

        # ランダムに k 件選択（重複排除）
        if seed is not None:
            random.seed(seed)

        # シャッフルして重複排除しながら取得
        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)

        result: list[str] = []
        seen_texts: set[str] = set()

        for idx in shuffled_indices:
            if len(result) >= k:
                break
            if 0 <= idx < len(self.prompts):
                prompt_text = self.prompts[idx]
                if prompt_text not in seen_texts:
                    seen_texts.add(prompt_text)
                    result.append(prompt_text)

        return result

    def get_strategy_guidance(
        self,
        config: DefenseConfig,
        failed_stage: Stage,
        n_examples: int = 3,
        seed: int | None = None,
        goal: str | None = None,
    ) -> StrategyGuidance:
        """失敗ステージに基づく戦略ガイダンスを生成する.

        Args:
            config: 防御構成
            failed_stage: 失敗したステージ
            n_examples: Few-shot 例の数
            seed: ランダムシード（類似度検索時は無視）
            goal: 攻撃目標（指定時は類似度ベースで例を取得）

        Returns:
            StrategyGuidance インスタンス
        """
        # 最適技法の取得
        best_technique = self.get_best_technique(failed_stage, config)

        # 技法の有効性
        technique_effectiveness = self.get_effectiveness(
            failed_stage, best_technique, config
        )

        # ステージの通過率/ブロック率
        stage_pass_rate = self.get_stage_pass_rate(config, failed_stage)
        stage_block_rate = 1.0 - stage_pass_rate

        # ボトルネックステージ
        bottleneck_stage = self.get_bottleneck_stage(config)

        # Few-shot 例の取得（goal 指定時は類似度ベース）
        if goal is not None and self.has_embeddings():
            example_prompts = self.get_similar_example_prompts(
                config, failed_stage, goal, n_examples
            )
        else:
            example_prompts = self.get_example_prompts(
                config, failed_stage, n_examples, seed
            )

        return StrategyGuidance(
            best_technique=best_technique,
            technique_effectiveness=technique_effectiveness,
            stage_pass_rate=stage_pass_rate,
            stage_block_rate=stage_block_rate,
            example_prompts=example_prompts,
            bottleneck_stage=bottleneck_stage,
        )

    def get_config_summary(self, config: DefenseConfig) -> dict:
        """構成のサマリー情報を返す.

        Args:
            config: 防御構成

        Returns:
            サマリー辞書
        """
        config_key = self._get_config_key(config)
        if config_key is None:
            return {
                "config_name": config.name,
                "asr": 0.0,
                "bottleneck_stage": "LLM",
                "found": False,
            }

        profile = self.config_profiles.get(config_key, {})

        return {
            "config_name": config.name,
            "asr": profile.get("asr", 0.0),
            "bottleneck_stage": profile.get("bottleneck_stage", "LLM"),
            "found": True,
        }


def load_knowledge_base(path: str | Path) -> Part1KnowledgeBase:
    """Knowledge Base を読み込む便利関数.

    Args:
        path: JSON ファイルパス

    Returns:
        Part1KnowledgeBase インスタンス
    """
    return Part1KnowledgeBase.load(path)
