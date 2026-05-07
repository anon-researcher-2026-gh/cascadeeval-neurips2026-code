"""C3AAgent: Cascade-Aware Adaptive Agent.

ステージ認識型の適応的攻撃エージェント。
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from src.c3a.agent.generator import AttackGenerator, FailedAttempt
from src.c3a.knowledge.kb_loader import Part1KnowledgeBase
from src.c3a.types import AttackResult, DefenseConfig, JudgmentCategory, Stage, StageResult

logger = logging.getLogger(__name__)


class OnlineEvaluator(Protocol):
    """オンライン評価器のプロトコル."""

    def evaluate(self, attack: str, goal: str) -> tuple[StageResult, bool]: ...
    def evaluate_full(
        self, attack: str, goal: str
    ) -> tuple[StageResult, JudgmentCategory | None, bool]: ...


@dataclass
class C3AConfig:
    """C3A エージェント設定.

    Attributes:
        max_iterations: 最大試行回数
        seed: ランダムシード
        verbose: 詳細ログを出力するか
    """

    max_iterations: int = 20
    seed: int = 42
    verbose: bool = False


@dataclass
class C3AAgent:
    """Cascade-Aware Adaptive Agent.

    ステージ認識型の適応的攻撃エージェント。
    失敗したステージに応じて攻撃戦略を動的に調整する。

    Attributes:
        knowledge_base: Part1KnowledgeBase
        generator: 攻撃生成器
        evaluator: オンライン評価器
        config: エージェント設定
    """

    knowledge_base: Part1KnowledgeBase
    generator: AttackGenerator
    evaluator: OnlineEvaluator
    config: C3AConfig = field(default_factory=C3AConfig)

    def attack(
        self,
        goal: str,
        defense_config: DefenseConfig,
    ) -> tuple[bool, list[AttackResult]]:
        """防御構成に対して攻撃を実行する.

        Args:
            goal: 攻撃目標（有害な情報を引き出したい内容）
            defense_config: 対象の防御構成

        Returns:
            (成功したか, 試行履歴) のタプル
        """
        history: list[AttackResult] = []
        failed_attempts: list[FailedAttempt] = []

        if self.config.verbose:
            logger.info(f"Starting attack on {defense_config.name}")
            logger.info(f"Goal: {goal[:100]}...")

        # 初期攻撃を生成
        gen_result = self.generator.generate_initial(
            goal=goal,
            defense_config=defense_config,
            seed=self.config.seed,
        )
        current_attack = gen_result.attack_text
        current_technique = gen_result.technique
        current_example_prompts = gen_result.example_prompts
        current_kb_guidance = gen_result.kb_guidance

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                logger.info(f"  Technique: {current_technique}")
                logger.info(f"  Attack: {current_attack[:200]}...")

            # 評価（詳細な判定カテゴリを含む）
            stage_result, judgment_category, is_jailbreak = self.evaluator.evaluate_full(
                current_attack, goal
            )

            # 履歴に追加
            history.append(
                AttackResult(
                    attack_text=current_attack,
                    iteration=iteration,
                    stage_result=stage_result,
                    is_jailbreak=is_jailbreak,
                    judgment_category=judgment_category,
                    technique=current_technique,
                    example_prompts=current_example_prompts,
                    kb_guidance=current_kb_guidance,
                )
            )

            if self.config.verbose:
                logger.info(
                    f"  Stages: IG={'✓' if stage_result.ig_passed else '✗'} "
                    f"LLM={'✓' if stage_result.llm_passed else '✗'} "
                    f"OG={'✓' if stage_result.og_passed else '✗'}"
                )
                if stage_result.response:
                    logger.info(f"  Response: {stage_result.response[:200]}...")
                if judgment_category:
                    logger.info(f"  Judgment: {judgment_category.value}")
                logger.info(f"  Jailbreak: {is_jailbreak}")

            # 成功した場合は終了
            if is_jailbreak:
                if self.config.verbose:
                    logger.info(f"Success at iteration {iteration + 1}!")
                return True, history

            # 失敗試行を記録（次回の生成で使用）
            failed_stage = stage_result.failed_stage
            failed_attempts.append(
                FailedAttempt(
                    attack_text=current_attack,
                    technique=current_technique or "unknown",
                    failed_stage=failed_stage.value,
                )
            )

            # シードを更新して多様性を確保
            iteration_seed = (
                self.config.seed + iteration + 1 if self.config.seed else None
            )

            # 失敗ステージとイテレーション、履歴に基づいて次の攻撃を生成
            gen_result = self.generator.generate(
                goal=goal,
                defense_config=defense_config,
                failed_stage=failed_stage,
                current_attack=current_attack,
                iteration=iteration + 1,  # 次のイテレーション番号
                failed_attempts=failed_attempts,
                seed=iteration_seed,
            )
            current_attack = gen_result.attack_text
            current_technique = gen_result.technique
            current_example_prompts = gen_result.example_prompts
            current_kb_guidance = gen_result.kb_guidance

        if self.config.verbose:
            logger.info("Max iterations reached, attack failed")

        return False, history

    def get_stats(self, history: list[AttackResult]) -> dict:
        """攻撃履歴の統計を返す.

        Args:
            history: 攻撃履歴

        Returns:
            統計情報の辞書
        """
        if not history:
            return {"n_iterations": 0, "success": False}

        # 各ステージでの失敗回数をカウント
        stage_failures: dict[Stage, int] = {
            Stage.INPUT_GUARD: 0,
            Stage.TARGET_LLM: 0,
            Stage.OUTPUT_GUARD: 0,
            Stage.PASSED_GUARDS: 0,
        }

        # 各判定カテゴリの回数をカウント
        judgment_counts: dict[str, int] = {
            JudgmentCategory.JAILBREAK.value: 0,
            JudgmentCategory.FICTIONAL_COMPLIANCE.value: 0,
            JudgmentCategory.REFUSAL.value: 0,
            JudgmentCategory.PARTIAL.value: 0,
        }

        # 技法別の成功/失敗をカウント
        technique_stats: dict[str, dict[str, int]] = {}

        for result in history:
            stage_failures[result.stage_result.failed_stage] += 1
            if result.judgment_category:
                judgment_counts[result.judgment_category.value] += 1

            # 技法別統計
            technique = result.technique or "unknown"
            if technique not in technique_stats:
                technique_stats[technique] = {"attempts": 0, "passed_ig": 0, "passed_llm": 0, "passed_og": 0}
            technique_stats[technique]["attempts"] += 1
            if result.stage_result.ig_passed:
                technique_stats[technique]["passed_ig"] += 1
            if result.stage_result.llm_passed:
                technique_stats[technique]["passed_llm"] += 1
            if result.stage_result.og_passed:
                technique_stats[technique]["passed_og"] += 1

        success = history[-1].is_jailbreak

        return {
            "n_iterations": len(history),
            "success": success,
            "stage_failures": {s.value: c for s, c in stage_failures.items()},
            "judgment_counts": judgment_counts,
            "technique_stats": technique_stats,
            "first_success_iteration": (
                next(
                    (i + 1 for i, r in enumerate(history) if r.is_jailbreak),
                    None,
                )
                if success
                else None
            ),
        }


def create_c3a_agent(
    knowledge_base: Part1KnowledgeBase,
    generator: AttackGenerator,
    evaluator: OnlineEvaluator,
    max_iterations: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> C3AAgent:
    """C3A エージェントを作成する便利関数.

    Args:
        knowledge_base: Part1KnowledgeBase
        generator: 攻撃生成器
        evaluator: オンライン評価器
        max_iterations: 最大試行回数
        seed: ランダムシード
        verbose: 詳細ログを出力するか

    Returns:
        C3AAgent インスタンス
    """
    config = C3AConfig(
        max_iterations=max_iterations,
        seed=seed,
        verbose=verbose,
    )
    return C3AAgent(
        knowledge_base=knowledge_base,
        generator=generator,
        evaluator=evaluator,
        config=config,
    )
