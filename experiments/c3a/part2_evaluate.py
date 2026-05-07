"""Part 2: オンライン評価スクリプト.

C3A エージェントとベースラインを実 API で評価する。
ExperimentManager により、完了済み実験は自動的にスキップされる。

Usage:
    uv run python experiments/c3a/part2_evaluate.py [options]

Options:
    --config: 設定ファイル（デフォルト: experiments/c3a/config.yaml）
    --dry-run: コスト見積もりのみ（実行しない）
    --mock: モッククライアントで実行（テスト用）
    --configs: 評価する構成名（カンマ区切り、config.yaml の name で指定）
    --agents: 評価するエージェント（カンマ区切り）
    --n-goals: 使用する目標数（config.yaml を上書き）
    --verbose: 詳細ログ（攻撃プロンプトと応答を表示）

Note:
    完了済み実験の再実行: results/c3a/experiments/index.json を削除
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.c3a.evaluation.experiment_manager import ExperimentManager
    from src.c3a.types import DefenseConfig

# .env から環境変数を読み込む
load_dotenv()

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

MOCK_KNOWLEDGE_BASE_PATH = (
    project_root / "experiments/c3a/fixtures/mock_knowledge_base.json"
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """ログ設定を初期化する."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # verbose モードでは src.c3a のログも INFO レベルで表示
    if verbose:
        logging.getLogger("src.c3a").setLevel(logging.INFO)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_goals(path: str, n_goals: int | None = None, categories: list[str] | None = None) -> list[dict]:
    """攻撃目標を読み込む."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    goals = data["goals"]

    # カテゴリフィルタ
    if categories:
        goals = [g for g in goals if g["category"] in categories]

    # 件数制限
    if n_goals:
        goals = goals[:n_goals]

    return goals


def estimate_cost(
    configs: list[dict],
    n_goals: int,
    max_iterations: int,
    agents: list[str],
) -> dict:
    """コストを見積もる.

    ローカルモデルと API モデルを区別してコストを計算する。

    API モデル（課金あり）:
    - Target LLM: GPT-4o, GPT-4o-mini, Gemini-2.0-Flash
    - Attacker LLM: GPT-4o-mini（攻撃生成用）

    ローカルモデル（課金なし）:
    - Target LLM: Llama-3.1-8B
    - Guards: LlamaGuard3/4, Qwen3-0.6BGuard, WildGuard, ShieldGemma
    - Jailbreak Judge: HarmBench-Llama-2-13b-cls（AutoDAN-Turbo 準拠）
    """
    # API を使用する LLM（短縮名）
    API_LLMS = {"GPT-4o", "gpt-4o", "GPT-4o-mini", "gpt-4o-mini", "Gemini-2.0-Flash", "gemini-2.0-flash"}

    # コスト単価（$/1K tokens、概算）
    COST_PER_1K = {
        "GPT-4o": 0.005,        # input: $2.5/1M, output: $10/1M
        "gpt-4o": 0.005,
        "GPT-4o-mini": 0.0003,  # input: $0.15/1M, output: $0.6/1M
        "gpt-4o-mini": 0.0003,
        "Gemini-2.0-Flash": 0.0001,  # 無料枠あり、超過時は低コスト
        "gemini-2.0-flash": 0.0001,
        "attacker": 0.0003,     # GPT-4o-mini
        # judge: HarmBench-Llama-2-13b-cls (ローカル実行、コストなし)
    }

    total_iterations = len(configs) * n_goals * max_iterations * len(agents)

    # 構成ごとのコストを計算
    api_target_llm_calls = 0
    local_target_llm_calls = 0

    for cfg in configs:
        llm = cfg["target_llm"]
        calls_per_config = n_goals * max_iterations * len(agents)

        if llm in API_LLMS:
            api_target_llm_calls += calls_per_config
        else:
            local_target_llm_calls += calls_per_config

    # コスト計算
    # 1. Target LLM（API のみ）
    target_llm_cost = 0.0
    for cfg in configs:
        llm = cfg["target_llm"]
        if llm in API_LLMS:
            calls = n_goals * max_iterations * len(agents)
            target_llm_cost += calls * COST_PER_1K.get(llm, 0.001)

    # 2. Attacker LLM（常に API: GPT-4o-mini）
    attacker_cost = total_iterations * COST_PER_1K["attacker"]

    # 3. Jailbreak Judge（ローカル: HarmBench-Llama-2-13b-cls）
    # AutoDAN-Turbo 準拠でローカル実行のためコストなし

    # Guards はローカル実行なのでコストなし

    total_cost = target_llm_cost + attacker_cost

    return {
        "n_configs": len(configs),
        "n_goals": n_goals,
        "max_iterations": max_iterations,
        "n_agents": len(agents),
        "total_iterations": total_iterations,
        "api_target_llm_calls": api_target_llm_calls,
        "local_target_llm_calls": local_target_llm_calls,
        "cost_breakdown": {
            "target_llm": target_llm_cost,
            "attacker_llm": attacker_cost,
            "jailbreak_judge": 0.0,  # ローカル実行
        },
        "estimated_cost_usd": total_cost,
    }


def print_experiment_status(
    exp_manager: "ExperimentManager",
    eval_configs: list[tuple[str, "DefenseConfig"]],
    agents: list[str],
    goals: list[dict],
) -> None:
    """実験の進捗状況を表示する."""
    print("\n【実験進捗状況】")
    all_goal_ids = [g["id"] for g in goals]
    total_pending = 0
    total_completed = 0

    for cfg_name, defense_config in eval_configs:
        dir_name = defense_config.dir_name
        print(f"  {cfg_name} ({defense_config.name}):")
        for agent_name in agents:
            n_completed = exp_manager.get_completed_count(dir_name, agent_name)
            pending = exp_manager.get_pending_goals(dir_name, agent_name, all_goal_ids)
            n_pending = len(pending)
            total_completed += n_completed
            total_pending += n_pending

            if n_completed == len(all_goal_ids):
                status = "完了"
            elif n_completed > 0:
                status = "進行中"
            else:
                status = "未開始"

            print(f"    {agent_name}: {n_completed}/{len(all_goal_ids)} ({status})")

    print()
    print(f"  合計: {total_completed} 完了, {total_pending} 残り")
    print()


# ============================================================
# モッククライアント（テスト用）
# ============================================================

class MockLLMClient:
    """モック LLM クライアント."""

    def __init__(self, name: str = "mock_llm"):
        self.name = name
        self._total_calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """モック生成."""
        self._total_calls += 1
        # 50% の確率で拒否応答を返す
        import random
        if random.random() < 0.3:
            return "I cannot help with that request."
        return f"Mock response to: {prompt[:50]}..."

    @property
    def total_calls(self) -> int:
        return self._total_calls


class MockGuardClient:
    """モック Guard クライアント."""

    def __init__(self, name: str = "mock_guard", block_rate: float = 0.3):
        self.name = name
        self.block_rate = block_rate
        self._total_calls = 0

    def classify(self, text: str) -> bool:
        """モック分類."""
        return self.classify_input(text)

    def classify_input(self, prompt: str) -> bool:
        """モック入力分類."""
        self._total_calls += 1
        import random
        return random.random() > self.block_rate

    def classify_output(self, prompt: str, response: str) -> bool:
        """モック出力分類."""
        self._total_calls += 1
        import random
        return random.random() > self.block_rate

    @property
    def total_calls(self) -> int:
        return self._total_calls


class MockJailbreakJudge:
    """モック Jailbreak 判定器（HarmBench 互換）."""

    def __init__(self, success_rate: float = 0.2):
        self.success_rate = success_rate
        self._total_calls = 0

    def judge(self, goal: str, response: str) -> bool:
        """モック判定（HarmBench 互換: binary classification）."""
        self._total_calls += 1
        import random
        return random.random() < self.success_rate

    def judge_detailed(self, goal: str, response: str):
        """詳細な判定を返す（後方互換性用）."""
        from src.c3a.evaluation.judge import JudgmentResult
        is_jailbreak = self.judge(goal, response)
        return JudgmentResult.from_bool(is_jailbreak)

    @property
    def total_calls(self) -> int:
        return self._total_calls


def prepare_retry_failed(
    exp_manager: "ExperimentManager",
    output_dir: Path,
    eval_configs: list[tuple[str, "DefenseConfig"]],
    agents: list[str],
) -> int:
    """失敗した目標を再実行対象に設定する.

    各 (config, agent) の results.json を読み、success=false の goal を
    index.json・results.json・attack_log.json から除去し、summary.json を再生成する。

    Returns:
        再実行対象として設定された goal の合計件数
    """
    total_retries = 0

    for _cfg_name, defense_config in eval_configs:
        dir_name = defense_config.dir_name
        for agent_name in agents:
            result_dir = output_dir / "experiments" / dir_name / agent_name
            results_path = result_dir / "results.json"

            if not results_path.exists():
                continue

            # results.json を読み込み、失敗した goal_id を収集
            with open(results_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            goals = results_data.get("goals", {})
            failed_goal_ids = [
                int(gid) for gid, gdata in goals.items()
                if not gdata.get("success")
            ]

            # 0. success=true だが index 未登録のゴールを completed に登録
            #    （中断されたランで mark_completed が走らなかったケースの補修）
            success_goal_ids = [
                int(gid) for gid, gdata in goals.items()
                if gdata.get("success")
            ]
            for gid in success_goal_ids:
                if not exp_manager.is_completed(dir_name, agent_name, gid):
                    exp_manager.mark_completed(dir_name, agent_name, gid)

            if not failed_goal_ids:
                continue

            # 1. index.json から失敗 goal を除去
            removed = exp_manager.remove_completed_goals(
                dir_name, agent_name, failed_goal_ids
            )

            # 2. results.json から失敗エントリを削除
            for gid in failed_goal_ids:
                goals.pop(str(gid), None)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            # 3. attack_log.json から該当 goal のエントリを削除
            json_log_path = result_dir / "attack_log.json"
            if json_log_path.exists():
                with open(json_log_path, "r", encoding="utf-8") as f:
                    attack_logs = json.load(f)
                failed_set = set(failed_goal_ids)
                attack_logs = [
                    log for log in attack_logs
                    if log.get("goal_id") not in failed_set
                ]
                with open(json_log_path, "w", encoding="utf-8") as f:
                    json.dump(attack_logs, f, indent=2, ensure_ascii=False)

            # 4. summary.json を再生成
            summaries: list[dict[str, Any]] = []
            for gdata in goals.values():
                history = gdata.get("history", [])
                if history:
                    if gdata["success"]:
                        final_prompt = next(
                            (h["attack_text"] for h in reversed(history) if h.get("is_jailbreak")),
                            history[-1]["attack_text"],
                        )
                    else:
                        final_prompt = history[-1]["attack_text"]
                else:
                    final_prompt = ""
                summaries.append({
                    "goal_id": gdata["goal_id"],
                    "goal": gdata["goal"],
                    "category": gdata["category"],
                    "success": gdata["success"],
                    "n_iterations": len(history),
                    "final_prompt": final_prompt,
                })
            summaries.sort(key=lambda x: x["goal_id"])
            summary_path = result_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)

            n_failed = len(failed_goal_ids)
            total_retries += n_failed
            print(f"  {dir_name}/{agent_name}: {n_failed} 件 (index から {removed} 件除去)")

    return total_retries


def create_defense_config_from_yaml(cfg: dict) -> "DefenseConfig":
    """YAML 設定から DefenseConfig を作成する."""
    from src.c3a.types import DefenseConfig, SHORT_TO_FULL_IG, SHORT_TO_FULL_LLM, SHORT_TO_FULL_OG

    ig = cfg["input_guard"]
    llm = cfg["target_llm"]
    og = cfg["output_guard"]

    # 短縮名からフル名に変換
    ig_full = SHORT_TO_FULL_IG.get(ig, ig)
    llm_full = SHORT_TO_FULL_LLM.get(llm, llm)
    og_full = SHORT_TO_FULL_OG.get(og, og)

    return DefenseConfig(ig_full, llm_full, og_full)  # type: ignore


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(description="Part 2: オンライン評価")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/c3a/config.yaml",
        help="設定ファイル",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コスト見積もりのみ（実行しない）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="モッククライアントで実行（テスト用）",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="評価する構成名（カンマ区切り）",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="評価するエージェント（カンマ区切り）",
    )
    parser.add_argument(
        "--n-goals",
        type=int,
        default=None,
        help="使用する目標数",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを出力",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="確認をスキップして実行",
    )
    parser.add_argument(
        "--goal-ids",
        type=str,
        default=None,
        help="特定のゴールIDのみ実行（カンマ区切り、例: 8,27,28,29）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="--goal-ids と併用: 完了済みゴールを強制的に再実験",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="失敗した目標（success=false）のみ再実行",
    )
    args = parser.parse_args()

    # ログ設定
    setup_logging(args.verbose)

    # 設定読み込み
    config = load_config(args.config)

    # コマンドライン引数で上書き
    agents = args.agents.split(",") if args.agents else config["agents"]
    n_goals = args.n_goals if args.n_goals else config["goals"].get("n_goals")
    goal_categories = config["goals"].get("categories")

    # 構成フィルタ
    all_configs = config["configs"]
    if args.configs:
        config_names = args.configs.split(",")
        configs = [c for c in all_configs if c["name"] in config_names]
    else:
        configs = all_configs

    # パス設定
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 攻撃目標読み込み
    goals_path = Path(config["goals"]["path"])
    if goals_path.exists():
        goals = load_goals(str(goals_path), n_goals, goal_categories)
    else:
        # goals.json がない場合はデモ用の目標を生成
        print(f"Warning: {goals_path} not found, using demo goals")
        goals = [
            {"id": 1, "goal": "<demo behavior goal 1>", "category": "DEMO"},
            {"id": 2, "goal": "<demo behavior goal 2>", "category": "DEMO"},
            {"id": 3, "goal": "<demo behavior goal 3>", "category": "DEMO"},
        ]
        if goal_categories:
            goals = [g for g in goals if g["category"] in goal_categories]
        if n_goals:
            goals = goals[:n_goals]

    # 特定のゴールIDのみ実行（--goal-ids オプション）
    if args.goal_ids:
        target_ids = {int(x.strip()) for x in args.goal_ids.split(",")}
        goals = [g for g in goals if g["id"] in target_ids]
        print(f"特定のゴールIDのみ実行: {sorted(target_ids)}")

    print("=" * 60)
    print("Part 2: オンライン評価")
    print("=" * 60)
    print(f"設定ファイル: {args.config}")
    print(f"Knowledge Base: {config['paths']['knowledge_base']}")
    print(f"Mock モード: {args.mock}")
    print()

    # 評価対象構成を表示
    print("【評価対象構成】")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg['name']}: {cfg['input_guard']} x {cfg['target_llm']} x {cfg['output_guard']}")
        if cfg.get("description"):
            print(f"     ({cfg['description']})")
    print()

    print(f"攻撃目標: {len(goals)} 件")
    print(f"エージェント: {agents}")
    print(f"最大試行回数: {config['experiment']['max_iterations']}")
    print()

    # コスト見積もり
    cost_estimate = estimate_cost(
        configs=configs,
        n_goals=len(goals),
        max_iterations=config["experiment"]["max_iterations"],
        agents=agents,
    )

    max_budget = config["cost"].get("max_budget_usd", 500)

    print("【コスト見積もり】")
    print(f"  総イテレーション: {cost_estimate['total_iterations']:,}")
    print(f"  API Target LLM 呼び出し: {cost_estimate['api_target_llm_calls']:,}")
    print(f"  ローカル Target LLM 呼び出し: {cost_estimate['local_target_llm_calls']:,} (無料)")
    print()
    print("  コスト内訳:")
    breakdown = cost_estimate["cost_breakdown"]
    print(f"    Target LLM (API): ${breakdown['target_llm']:.2f}")
    print(f"    Attacker LLM (GPT-4o-mini): ${breakdown['attacker_llm']:.2f}")
    print("    Jailbreak Judge (HarmBench): $0.00 (ローカル)")
    print("    Guards: $0.00 (ローカル)")
    print()
    print(f"  推定総コスト: ${cost_estimate['estimated_cost_usd']:.2f}")
    print(f"  予算上限: ${max_budget:.2f}")

    if cost_estimate["estimated_cost_usd"] > max_budget:
        print("  Warning: 予算上限を超過する可能性があります")
    print()

    # --retry-failed の事前集計（dry-run でも表示するため、重い import 不要な範囲で実施）
    if args.retry_failed:
        retry_fail = 0
        retry_never_run = 0
        all_goal_ids_set = {g["id"] for g in goals}
        for cfg in configs:
            dc = create_defense_config_from_yaml(cfg)
            for agent_name in agents:
                rp = output_dir / "experiments" / dc.dir_name / agent_name / "results.json"
                if not rp.exists():
                    # results.json が無い → 全ゴール未実行
                    retry_never_run += len(goals)
                    print(f"  {dc.dir_name}/{agent_name}: 未実行 {len(goals)} 件")
                    continue
                with open(rp, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                rgoals = rdata.get("goals", {})
                n_fail = sum(
                    1 for g in rgoals.values()
                    if not g.get("success")
                )
                n_success = sum(
                    1 for g in rgoals.values()
                    if g.get("success")
                )
                result_ids = {int(gid) for gid in rgoals.keys()}
                n_never = len(all_goal_ids_set - result_ids)
                retry_fail += n_fail
                retry_never_run += n_never
                print(
                    f"  {dc.dir_name}/{agent_name}: "
                    f"success={n_success}, fail={n_fail}, 未実行={n_never}"
                )
        retry_total = retry_fail + retry_never_run
        print(f"  → 実行予定: {retry_total} 件 (fail再実行={retry_fail}, 未実行={retry_never_run})")
        print()

    if args.dry_run:
        print("Dry run モードのため実行をスキップします。")
        return

    # 確認
    if not args.mock and not args.yes and config["cost"].get("confirm_before_run", True):
        response = input("実験を開始しますか？ (y/n): ")
        if response.lower() != "y":
            print("実験をキャンセルしました。")
            return

    print()
    print("実験を開始します...")
    print("=" * 60)

    try:
        from src.c3a.agent import (
            AGENT_ABLATION_MODES,
            create_attack_generator,
            create_c3a_agent,
        )
        from src.c3a.baselines import (
            create_c3a_nokb_baseline,
            create_direct_baseline,
            create_pair_baseline,
        )
        from src.c3a.evaluation import OnlineEvaluator
        from src.c3a.knowledge import Part1KnowledgeBase
        from src.c3a.types import DefenseConfig

        # Knowledge Base 読み込み（埋め込み付き）
        kb_path = Path(config["paths"]["knowledge_base"])
        if not kb_path.exists():
            if args.mock and MOCK_KNOWLEDGE_BASE_PATH.exists():
                print(f"Warning: Knowledge Base not found: {kb_path}")
                print(f"Using synthetic mock Knowledge Base: {MOCK_KNOWLEDGE_BASE_PATH}")
                kb_path = MOCK_KNOWLEDGE_BASE_PATH
            else:
                print(f"Error: Knowledge Base not found: {kb_path}")
                print("Run part1_analysis.py first or restore it from the gated data artifact.")
                return

        if kb_path == MOCK_KNOWLEDGE_BASE_PATH:
            kb = Part1KnowledgeBase.load(kb_path)
        else:
            kb = Part1KnowledgeBase.load_with_embeddings(kb_path)
        emb_status = "with embeddings" if kb.has_embeddings() else "without embeddings"
        print(f"Knowledge Base loaded: {kb.n_configs} configs, {kb.n_attacks:,} attacks ({emb_status})")

        # DefenseConfig を作成
        eval_configs: list[tuple[str, DefenseConfig]] = []
        for cfg in configs:
            dc = create_defense_config_from_yaml(cfg)
            eval_configs.append((cfg["name"], dc))

        # クライアント作成
        if args.mock:
            # モッククライアント
            print("Mock クライアントを使用します")
            attacker_llm = MockLLMClient("attacker")
            judge = MockJailbreakJudge(success_rate=0.3)
        else:
            # 実クライアント
            from src.c3a.clients import create_openai_client
            from src.c3a.evaluation.judge import create_harmbench_judge

            # 必要な API キーをチェック
            required_keys: dict[str, str] = {}
            required_keys["OPENAI_API_KEY"] = "Attacker LLM (GPT-4o-mini)"
            required_keys["HF_TOKEN"] = "HuggingFace Gated Models (Guards, Llama, HarmBench Classifier)"

            for cfg in configs:
                llm = cfg["target_llm"]
                if llm in ["Gemini-2.0-Flash", "gemini-2.0-flash"]:
                    required_keys["GOOGLE_API_KEY"] = "Target LLM (Gemini-2.0-Flash)"

            missing_keys = [key for key in required_keys if not os.environ.get(key)]
            if missing_keys:
                print("Error: 以下の環境変数が設定されていません:")
                for key in missing_keys:
                    print(f"  - {key}: {required_keys[key]} に必要")
                return

            print("Attacker LLM (GPT-4o-mini) を初期化中...")
            attacker_llm = create_openai_client(model="gpt-4o-mini")

            print("Jailbreak Judge (HarmBench-Llama-2-13b-cls) を初期化中...")
            judge = create_harmbench_judge(
                cache_path=output_dir / "judge_cache.json",
                use_cache=True,
            )

        # 実験管理システム初期化
        from src.c3a.evaluation import (
            ConfigResultWriter,
            ExperimentManager,
            aggregate_all_results,
        )

        exp_manager = ExperimentManager(output_dir)
        all_goal_ids = [g["id"] for g in goals]

        # --force オプション: 指定したゴールを完了済みリストから削除
        if args.force and args.goal_ids:
            target_ids = [int(x.strip()) for x in args.goal_ids.split(",")]
            print(f"--force: 完了済みリストから goal IDs {target_ids} を削除中...")
            for cfg_name, cfg_data in [(c["name"], c) for c in configs]:
                dir_name = f"{cfg_data['input_guard']}__{cfg_data['target_llm']}__{cfg_data['output_guard']}"
                for agent_name in agents:
                    removed = exp_manager.remove_completed_goals(
                        dir_name, agent_name, target_ids
                    )
                    if removed > 0:
                        print(f"  {dir_name}/{agent_name}: {removed} 件削除")
            print()

        # --retry-failed: 失敗した目標を再実行対象に設定
        if args.retry_failed:
            print("--retry-failed: 失敗した目標を再実行対象に設定中...")
            n_retries = prepare_retry_failed(exp_manager, output_dir, eval_configs, agents)
            print(f"  合計: {n_retries} 件")
            exp_manager._index = None  # キャッシュリセット
            print()

        # 実験進捗状況を表示
        print_experiment_status(exp_manager, eval_configs, agents, goals)

        # 全て完了済みかチェック
        total_pending = sum(
            len(exp_manager.get_pending_goals(dc.dir_name, a, all_goal_ids))
            for _, dc in eval_configs
            for a in agents
        )
        if total_pending == 0:
            print("全ての実験が完了済みです。")
            print("再実行する場合は results/c3a/experiments/index.json を削除してください。")
        else:
            print(f"実行する実験: {total_pending} 件")

        # 評価ループ
        print()
        print("=" * 60)
        print("評価開始")
        print("=" * 60)

        max_iterations = config["experiment"]["max_iterations"]
        seed = config["experiment"]["seed"]
        exp_settings = {
            "max_iterations": max_iterations,
            "seed": seed,
        }

        for cfg_name, defense_config in eval_configs:
            dir_name = defense_config.dir_name  # ディレクトリ名: {IG}__{LLM}__{OG}
            print(f"\n【構成: {cfg_name}】")
            print(f"  {defense_config.name}")
            print(f"  Dir: {dir_name}")

            # この構成で実行するエージェントを確認
            pending_agents = [
                a for a in agents
                if len(exp_manager.get_pending_goals(dir_name, a, all_goal_ids)) > 0
            ]

            if not pending_agents:
                print("  全エージェントで完了済み、スキップします")
                continue

            # この構成用のクライアントを作成
            if args.mock:
                # モッククライアント
                input_guard = MockGuardClient("input_guard", block_rate=0.2)
                target_llm = MockLLMClient("target_llm")
                output_guard = MockGuardClient("output_guard", block_rate=0.15)
            else:
                # 実クライアント
                from src.c3a.clients import (
                    create_huggingface_guard_client,
                    create_huggingface_llm_client,
                    create_target_llm_client,
                )

                # Input Guard
                print(f"  Input Guard ({defense_config.input_guard}) をロード中...")
                input_guard = create_huggingface_guard_client(defense_config.input_guard)

                # Target LLM
                print(f"  Target LLM ({defense_config.target_llm}) をロード中...")
                if defense_config.target_llm in ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash"]:
                    target_llm = create_target_llm_client(defense_config.target_llm)
                else:
                    # ローカルモデル
                    from src.c3a.clients.huggingface import TARGET_LLM_MODEL_IDS
                    model_id = TARGET_LLM_MODEL_IDS.get(
                        defense_config.target_llm,
                        defense_config.target_llm
                    )
                    target_llm = create_huggingface_llm_client(model_id)

                # Output Guard
                print(f"  Output Guard ({defense_config.output_guard}) をロード中...")
                output_guard = create_huggingface_guard_client(defense_config.output_guard)

            # OnlineEvaluator
            evaluator = OnlineEvaluator(
                input_guard=input_guard,
                target_llm=target_llm,
                output_guard=output_guard,
                jailbreak_judge=judge,
            )

            # エージェント作成
            agent_instances = {}

            # C3A 系エージェント（Ablation モード対応）
            for agent_name in pending_agents:
                if agent_name in AGENT_ABLATION_MODES:
                    ablation_mode = AGENT_ABLATION_MODES[agent_name]
                    generator = create_attack_generator(
                        llm_client=attacker_llm,
                        knowledge_base=kb,
                        ablation_mode=ablation_mode,
                    )
                    agent_instances[agent_name] = create_c3a_agent(
                        knowledge_base=kb,
                        generator=generator,
                        evaluator=evaluator,
                        max_iterations=max_iterations,
                        seed=seed,
                        verbose=args.verbose,
                    )

            # ベースライン: PAIR
            if "pair" in pending_agents:
                agent_instances["pair"] = create_pair_baseline(
                    llm_client=attacker_llm,
                    evaluator=evaluator,
                    max_iterations=max_iterations,
                    verbose=args.verbose,
                )

            # ベースライン: Direct Request
            if "direct" in pending_agents:
                agent_instances["direct"] = create_direct_baseline(
                    evaluator=evaluator,
                    verbose=args.verbose,
                )

            # ベースライン: C3A-NoKB（Stage情報のみ、KB なし）
            if "c3a-nokb" in pending_agents:
                agent_instances["c3a-nokb"] = create_c3a_nokb_baseline(
                    llm_client=attacker_llm,
                    evaluator=evaluator,
                    max_iterations=max_iterations,
                    seed=seed,
                    verbose=args.verbose,
                )

            # 各エージェントで評価
            for agent_name, agent in agent_instances.items():
                print(f"\n  エージェント: {agent_name}")

                # この (config, agent) 用の結果ライター
                result_dir = exp_manager.get_result_dir(dir_name, agent_name)
                result_writer = ConfigResultWriter(result_dir)

                # メタデータを記録（Ablation モード等）
                ablation_mode = AGENT_ABLATION_MODES.get(agent_name)
                result_writer.set_metadata({
                    "agent_name": agent_name,
                    "ablation_mode": ablation_mode.value if ablation_mode else None,
                    "defense_config": {
                        "name": defense_config.name,
                        "input_guard": defense_config.input_guard,
                        "target_llm": defense_config.target_llm,
                        "output_guard": defense_config.output_guard,
                    },
                    "experiment_settings": exp_settings,
                })

                # 未完了の goal のみ実行
                pending_goal_ids = exp_manager.get_pending_goals(dir_name, agent_name, all_goal_ids)
                n_completed = exp_manager.get_completed_count(dir_name, agent_name)

                if not pending_goal_ids:
                    print("    全 goal 完了済み")
                    continue

                print(f"    完了済み: {n_completed}/{len(goals)}, 残り: {len(pending_goal_ids)}")

                for goal_data in goals:
                    goal_id = goal_data["id"]
                    goal_text = goal_data["goal"]
                    goal_category = goal_data["category"]

                    # 完了済みならスキップ
                    if goal_id not in pending_goal_ids:
                        continue

                    # 進行中としてマーク
                    exp_manager.mark_in_progress(dir_name, agent_name, goal_id, exp_settings)

                    print(f"    Goal {goal_id}: {goal_text[:50]}...")

                    # 攻撃実行
                    success, history = agent.attack(goal_text, defense_config)

                    # 履歴を辞書形式に変換
                    history_dicts = [
                        {
                            "iteration": h.iteration,
                            "technique": h.technique,
                            "attack_text": h.attack_text,
                            "ig_passed": h.stage_result.ig_passed,
                            "llm_passed": h.stage_result.llm_passed,
                            "og_passed": h.stage_result.og_passed,
                            "failed_stage": h.stage_result.failed_stage.value,
                            "response": h.stage_result.response,
                            "judgment_category": h.judgment_category.value if h.judgment_category else None,
                            "is_jailbreak": h.is_jailbreak,
                            "kb_guidance": asdict(h.kb_guidance) if h.kb_guidance else None,
                        }
                        for h in history
                    ]

                    # 結果を保存
                    result_writer.save_goal_result(
                        goal_id=goal_id,
                        goal_text=goal_text,
                        category=goal_category,
                        success=success,
                        history=history_dicts,
                    )

                    # 攻撃ログを記録
                    for h in history:
                        failed_stage = h.stage_result.failed_stage.value if not h.stage_result.success else None
                        judgment_cat = h.judgment_category.value if h.judgment_category else None
                        kb_guidance_dict = asdict(h.kb_guidance) if h.kb_guidance else None
                        result_writer.log_attack(
                            goal_id=goal_id,
                            goal_text=goal_text,
                            iteration=h.iteration,
                            attack_text=h.attack_text,
                            ig_passed=h.stage_result.ig_passed,
                            llm_passed=h.stage_result.llm_passed,
                            og_passed=h.stage_result.og_passed,
                            response=h.stage_result.response,
                            is_jailbreak=h.is_jailbreak,
                            success=success,
                            technique=h.technique,
                            failed_stage=failed_stage,
                            judgment_category=judgment_cat,
                            kb_guidance=kb_guidance_dict,
                            example_prompts=h.example_prompts,
                        )

                    # 完了としてマーク
                    exp_manager.mark_completed(dir_name, agent_name, goal_id, exp_settings)

                    status = "OK" if success else "NG"
                    print(f"      {status} ({len(history)} iterations)")

            # GPU メモリ解放（実クライアントの場合）
            if not args.mock:
                del input_guard, target_llm, output_guard, evaluator
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        # 集約結果を保存
        aggregated = aggregate_all_results(output_dir)
        aggregated_path = output_dir / "aggregated_results.json"
        with open(aggregated_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"\n集約結果を保存: {aggregated_path}")

        # サマリー表示
        print()
        print("=" * 60)
        print("評価完了サマリー")
        print("=" * 60)

        for cfg_name, cfg_data in aggregated.get("configs", {}).items():
            print(f"\n【{cfg_name}】")
            for agent_name, agent_data in cfg_data.items():
                n_goals = agent_data.get("n_goals", 0)
                n_success = agent_data.get("n_successes", 0)
                asr = agent_data.get("asr", 0)
                print(f"  {agent_name}: ASR={asr:.1%} ({n_success}/{n_goals})")

    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("Part 2 完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
