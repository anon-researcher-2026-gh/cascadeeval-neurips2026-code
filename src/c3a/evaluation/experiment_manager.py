"""実験管理システム.

実験結果の保存・読み込み・重複スキップを管理する。
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_VERSION = "1.0"


@dataclass
class ExperimentIndex:
    """実験インデックス."""

    version: str = INDEX_VERSION
    created_at: str = ""
    updated_at: str = ""
    experiments: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "experiments": self.experiments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentIndex":
        """辞書から生成."""
        return cls(
            version=data.get("version", INDEX_VERSION),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            experiments=data.get("experiments", {}),
        )


class ExperimentManager:
    """実験管理クラス.

    - 完了済み実験の追跡
    - 結果ディレクトリの管理
    - 自動スキップ判定
    """

    def __init__(self, base_dir: Path):
        """初期化.

        Args:
            base_dir: 結果保存のベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.index_path = self.experiments_dir / "index.json"
        self._index: ExperimentIndex | None = None

    def _ensure_dirs(self) -> None:
        """必要なディレクトリを作成."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> ExperimentIndex:
        """インデックスを読み込む."""
        if self._index is not None:
            return self._index

        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = ExperimentIndex.from_dict(data)
        else:
            self._index = ExperimentIndex(
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )

        return self._index

    def _save_index(self) -> None:
        """インデックスを保存."""
        self._ensure_dirs()
        index = self._load_index()
        index.updated_at = datetime.now().isoformat()

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, indent=2, ensure_ascii=False)

    def get_result_dir(self, config_name: str, agent_name: str) -> Path:
        """結果ディレクトリを取得（なければ作成）.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名

        Returns:
            結果ディレクトリのパス
        """
        result_dir = self.experiments_dir / config_name / agent_name
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def is_completed(self, config_name: str, agent_name: str, goal_id: int) -> bool:
        """goal が完了済みか判定.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名
            goal_id: 目標 ID

        Returns:
            完了済みなら True
        """
        index = self._load_index()
        exp = index.experiments.get(config_name, {}).get(agent_name, {})
        return goal_id in exp.get("completed_goals", [])

    def mark_completed(
        self,
        config_name: str,
        agent_name: str,
        goal_id: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """goal を完了としてマーク.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名
            goal_id: 目標 ID
            settings: 実験設定（初回のみ保存）
        """
        index = self._load_index()

        # 構成エントリを初期化
        if config_name not in index.experiments:
            index.experiments[config_name] = {}

        if agent_name not in index.experiments[config_name]:
            index.experiments[config_name][agent_name] = {
                "completed_goals": [],
                "in_progress_goal": None,
                "settings": settings or {},
                "last_updated": "",
            }

        exp = index.experiments[config_name][agent_name]

        # 完了リストに追加
        if goal_id not in exp["completed_goals"]:
            exp["completed_goals"].append(goal_id)
            exp["completed_goals"].sort()

        # in_progress をクリア
        if exp.get("in_progress_goal") == goal_id:
            exp["in_progress_goal"] = None

        exp["last_updated"] = datetime.now().isoformat()

        self._save_index()

    def mark_in_progress(
        self,
        config_name: str,
        agent_name: str,
        goal_id: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """goal を進行中としてマーク.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名
            goal_id: 目標 ID
            settings: 実験設定
        """
        index = self._load_index()

        if config_name not in index.experiments:
            index.experiments[config_name] = {}

        if agent_name not in index.experiments[config_name]:
            index.experiments[config_name][agent_name] = {
                "completed_goals": [],
                "in_progress_goal": None,
                "settings": settings or {},
                "last_updated": "",
            }

        exp = index.experiments[config_name][agent_name]
        exp["in_progress_goal"] = goal_id
        exp["last_updated"] = datetime.now().isoformat()

        self._save_index()

    def get_pending_goals(
        self, config_name: str, agent_name: str, all_goal_ids: list[int]
    ) -> list[int]:
        """未完了の goal リストを取得.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名
            all_goal_ids: 全 goal ID のリスト

        Returns:
            未完了の goal ID リスト
        """
        index = self._load_index()
        exp = index.experiments.get(config_name, {}).get(agent_name, {})
        completed = set(exp.get("completed_goals", []))
        return [gid for gid in all_goal_ids if gid not in completed]

    def remove_completed_goals(
        self, config_name: str, agent_name: str, goal_ids: list[int]
    ) -> int:
        """完了済みリストから指定したゴールを削除（再実験用）.

        Args:
            config_name: 防御構成名
            agent_name: エージェント名
            goal_ids: 削除するゴールIDのリスト

        Returns:
            実際に削除された件数
        """
        index = self._load_index()
        exp = index.experiments.get(config_name, {}).get(agent_name, {})

        if not exp:
            return 0

        completed = exp.get("completed_goals", [])
        original_count = len(completed)
        goal_ids_set = set(goal_ids)
        exp["completed_goals"] = [gid for gid in completed if gid not in goal_ids_set]
        removed_count = original_count - len(exp["completed_goals"])

        if removed_count > 0:
            exp["last_updated"] = datetime.now().isoformat()
            self._save_index()
            logger.info(
                f"Removed {removed_count} goals from completed list: "
                f"{config_name}/{agent_name}"
            )

        return removed_count

    def get_completed_count(self, config_name: str, agent_name: str) -> int:
        """完了済み goal 数を取得."""
        index = self._load_index()
        exp = index.experiments.get(config_name, {}).get(agent_name, {})
        return len(exp.get("completed_goals", []))

    def get_summary(self) -> dict[str, Any]:
        """全実験のサマリーを取得."""
        index = self._load_index()
        summary: dict[str, Any] = {
            "total_configs": len(index.experiments),
            "configs": {},
        }

        for config_name, agents in index.experiments.items():
            config_summary: dict[str, Any] = {
                "agents": {},
                "total_completed": 0,
            }
            for agent_name, exp_data in agents.items():
                n_completed = len(exp_data.get("completed_goals", []))
                config_summary["agents"][agent_name] = {
                    "completed": n_completed,
                    "in_progress": exp_data.get("in_progress_goal"),
                }
                config_summary["total_completed"] += n_completed

            summary["configs"][config_name] = config_summary

        return summary


class ConfigResultWriter:
    """構成別結果ライター.

    各 (config, agent) の結果を独立したファイルに書き込む。
    """

    CSV_HEADERS = [
        "timestamp",
        "goal_id",
        "goal",
        "iteration",
        "technique",
        "failed_stage",
        "attack_text",
        "ig_passed",
        "llm_passed",
        "og_passed",
        "response",
        "judgment_category",
        "is_jailbreak",
        "success",
    ]

    def __init__(self, result_dir: Path):
        """初期化.

        Args:
            result_dir: 結果ディレクトリ
        """
        self.result_dir = result_dir
        self.results_path = result_dir / "results.json"
        self.summary_path = result_dir / "summary.json"  # goal ごとのサマリー
        self.csv_path = result_dir / "attack_log.csv"
        self.json_log_path = result_dir / "attack_log.json"
        self.judge_cache_path = result_dir / "judge_cache.json"

        self._results: dict[str, Any] = self._load_results()
        self._attack_logs: list[dict[str, Any]] = self._load_attack_logs()
        self._csv_initialized = self.csv_path.exists()

    def _load_results(self) -> dict[str, Any]:
        """既存の結果を読み込む."""
        if self.results_path.exists():
            with open(self.results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"metadata": {}, "goals": {}}

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """メタデータを設定する.

        Args:
            metadata: 実験メタデータ（ablation_mode, config 等）
        """
        self._results["metadata"] = metadata
        self._results["metadata"]["created_at"] = datetime.now().isoformat()
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

    def _load_attack_logs(self) -> list[dict[str, Any]]:
        """既存の攻撃ログを読み込む."""
        if self.json_log_path.exists():
            with open(self.json_log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_goal_result(
        self,
        goal_id: int,
        goal_text: str,
        category: str,
        success: bool,
        history: list[dict[str, Any]],
    ) -> None:
        """goal の結果を保存.

        Args:
            goal_id: 目標 ID
            goal_text: 目標テキスト
            category: カテゴリ
            success: 成功フラグ
            history: 攻撃履歴
        """
        self._results["goals"][str(goal_id)] = {
            "goal_id": goal_id,
            "goal": goal_text,
            "category": category,
            "success": success,
            "n_iterations": len(history),
            "history": history,
            "completed_at": datetime.now().isoformat(),
        }

        # 即時保存（詳細版）
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

        # サマリー再生成（簡潔版: goal ごとに1行）
        self._regenerate_summary()

    def _regenerate_summary(self) -> None:
        """サマリーファイルを再生成する.

        results.json の内容から summary.jsonl を生成する。
        goal ごとに1行で、必要最小限の情報のみを含む。
        """
        summaries: list[dict[str, Any]] = []

        for goal_data in self._results.get("goals", {}).values():
            goal_id = goal_data["goal_id"]
            goal_text = goal_data["goal"]
            category = goal_data["category"]
            success = goal_data["success"]
            history = goal_data.get("history", [])
            n_iterations = len(history)

            # 成功時は成功したプロンプト、失敗時は最後のプロンプトを保存
            if history:
                if success:
                    # 成功した iteration を探す
                    final_prompt = next(
                        (h["attack_text"] for h in reversed(history) if h.get("is_jailbreak")),
                        history[-1]["attack_text"],
                    )
                else:
                    final_prompt = history[-1]["attack_text"]
            else:
                final_prompt = ""

            summaries.append({
                "goal_id": goal_id,
                "goal": goal_text,
                "category": category,
                "success": success,
                "n_iterations": n_iterations,
                "final_prompt": final_prompt,
            })

        # goal_id でソートして JSON 配列形式で保存
        summaries.sort(key=lambda x: x["goal_id"])
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    def log_attack(
        self,
        goal_id: int,
        goal_text: str,
        iteration: int,
        attack_text: str,
        ig_passed: bool,
        llm_passed: bool,
        og_passed: bool,
        response: str | None,
        is_jailbreak: bool,
        success: bool,
        technique: str | None = None,
        failed_stage: str | None = None,
        judgment_category: str | None = None,
        kb_guidance: dict[str, Any] | None = None,
        example_prompts: list[str] | None = None,
    ) -> None:
        """攻撃ログを記録.

        Args:
            goal_id: 目標 ID
            goal_text: 目標テキスト
            iteration: イテレーション番号
            attack_text: 攻撃テキスト
            ig_passed: Input Guard 通過フラグ
            llm_passed: Target LLM 通過フラグ
            og_passed: Output Guard 通過フラグ
            response: レスポンス
            is_jailbreak: Jailbreak 成功フラグ
            success: 目標達成フラグ
            technique: 使用技法
            failed_stage: 失敗ステージ
            judgment_category: 判定カテゴリ
            kb_guidance: KB ガイダンス情報
            example_prompts: 参照した Few-shot 例
        """
        timestamp = datetime.now().isoformat()

        record = {
            "timestamp": timestamp,
            "goal_id": goal_id,
            "goal": goal_text,
            "iteration": iteration,
            "technique": technique,
            "failed_stage": failed_stage,
            "attack_text": attack_text,
            "ig_passed": ig_passed,
            "llm_passed": llm_passed,
            "og_passed": og_passed,
            "response": response,
            "judgment_category": judgment_category,
            "is_jailbreak": is_jailbreak,
            "success": success,
            "kb_guidance": kb_guidance,
            "example_prompts": example_prompts,
        }

        # CSV に追記
        mode = "a" if self._csv_initialized else "w"
        with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True

            # CSV 用に省略（複雑なフィールドは除外）
            csv_record = {k: v for k, v in record.items() if k in self.CSV_HEADERS}
            csv_record["goal"] = (goal_text[:200] + "...") if len(goal_text) > 200 else goal_text
            csv_record["attack_text"] = (
                (attack_text[:500] + "...") if len(attack_text) > 500 else attack_text
            )
            csv_record["response"] = (
                ((response or "")[:500] + "...") if response and len(response) > 500 else (response or "")
            )
            writer.writerow(csv_record)

        # JSON に追加
        self._attack_logs.append(record)
        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(self._attack_logs, f, indent=2, ensure_ascii=False)

    def get_results(self) -> dict[str, Any]:
        """保存済み結果を取得."""
        return self._results


def aggregate_all_results(base_dir: Path) -> dict[str, Any]:
    """全結果を集約.

    Args:
        base_dir: 結果のベースディレクトリ

    Returns:
        集約された結果
    """
    experiments_dir = base_dir / "experiments"
    if not experiments_dir.exists():
        return {"error": "No experiments found"}

    aggregated: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "summary": {
            "total_goals": 0,
            "total_successes": 0,
            "by_config": {},
        },
    }

    for config_dir in experiments_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == "index.json":
            continue

        config_name = config_dir.name
        aggregated["configs"][config_name] = {}
        aggregated["summary"]["by_config"][config_name] = {}

        for agent_dir in config_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_name = agent_dir.name
            results_path = agent_dir / "results.json"

            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)

                goals = results.get("goals", {})
                n_goals = len(goals)
                n_successes = sum(1 for g in goals.values() if g.get("success"))

                aggregated["configs"][config_name][agent_name] = {
                    "n_goals": n_goals,
                    "n_successes": n_successes,
                    "asr": n_successes / n_goals if n_goals > 0 else 0,
                    "goals": goals,
                }

                aggregated["summary"]["total_goals"] += n_goals
                aggregated["summary"]["total_successes"] += n_successes
                aggregated["summary"]["by_config"][config_name][agent_name] = {
                    "n_goals": n_goals,
                    "n_successes": n_successes,
                    "asr": n_successes / n_goals if n_goals > 0 else 0,
                }

    return aggregated
