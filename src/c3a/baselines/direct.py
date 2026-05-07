"""Direct Request ベースライン.

Goal テキストをそのまま防御システムに送信する最も単純なベースライン。
攻撃生成や適応は一切行わない。

JailbreakBench (Chao et al., NeurIPS 2024) で標準的に使用されるベースライン。
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from src.c3a.types import AttackResult, DefenseConfig, StageResult

logger = logging.getLogger(__name__)


class OnlineEvaluator(Protocol):
    """オンライン評価器のプロトコル."""

    def evaluate(self, attack: str, goal: str) -> tuple[StageResult, bool]: ...


@dataclass
class DirectConfig:
    """Direct Request 設定.

    Attributes:
        verbose: 詳細ログを出力するか
    """

    verbose: bool = False


@dataclass
class DirectBaseline:
    """Direct Request ベースライン.

    Goal テキストをそのまま1回送信する。
    攻撃生成・適応・反復は一切行わない。

    Attributes:
        evaluator: オンライン評価器
        config: 設定
    """

    evaluator: OnlineEvaluator
    config: DirectConfig = field(default_factory=DirectConfig)

    def attack(
        self,
        goal: str,
        defense_config: DefenseConfig,
    ) -> tuple[bool, list[AttackResult]]:
        """Goal テキストをそのまま送信して評価する.

        Args:
            goal: 攻撃目標（そのまま送信）
            defense_config: 対象の防御構成

        Returns:
            (成功したか, 試行履歴) のタプル
        """
        if self.config.verbose:
            logger.info(f"Direct: Sending goal as-is to {defense_config.name}")
            logger.info(f"Goal: {goal[:100]}...")

        stage_result, is_jailbreak = self.evaluator.evaluate(goal, goal)

        history = [
            AttackResult(
                attack_text=goal,
                iteration=0,
                stage_result=stage_result,
                is_jailbreak=is_jailbreak,
            )
        ]

        if self.config.verbose:
            logger.info(
                f"  Stages: IG={'✓' if stage_result.ig_passed else '✗'} "
                f"LLM={'✓' if stage_result.llm_passed else '✗'} "
                f"OG={'✓' if stage_result.og_passed else '✗'}"
            )
            logger.info(f"  Jailbreak: {is_jailbreak}")

        return is_jailbreak, history

    def get_stats(self, history: list[AttackResult]) -> dict:
        """攻撃履歴の統計を返す."""
        if not history:
            return {"n_iterations": 0, "success": False}

        return {
            "n_iterations": 1,
            "success": history[0].is_jailbreak,
            "first_success_iteration": 1 if history[0].is_jailbreak else None,
        }


def create_direct_baseline(
    evaluator: OnlineEvaluator,
    verbose: bool = False,
    **_kwargs: object,
) -> DirectBaseline:
    """Direct Request ベースラインを作成する便利関数.

    Args:
        evaluator: オンライン評価器
        verbose: 詳細ログを出力するか
        **_kwargs: 互換性のため無視するキーワード引数

    Returns:
        DirectBaseline インスタンス
    """
    config = DirectConfig(verbose=verbose)
    return DirectBaseline(evaluator=evaluator, config=config)
