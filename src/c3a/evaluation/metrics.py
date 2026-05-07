"""メトリクス計算モジュール.

Part 2 オンライン評価のメトリクスを計算する。
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.c3a.types import AttackResult, Stage


@dataclass
class EvaluationMetrics:
    """評価メトリクス.

    Attributes:
        asr: Attack Success Rate
        avg_queries: 成功時の平均クエリ数
        std_queries: 成功時のクエリ数標準偏差
        stage_failure_dist: ステージ別失敗分布
        n_total: 総試行数
        n_success: 成功数
    """

    asr: float
    avg_queries: float
    std_queries: float
    stage_failure_dist: dict[str, float]
    n_total: int
    n_success: int


def compute_asr(results: list[tuple[bool, list[AttackResult]]]) -> float:
    """Attack Success Rate を計算する.

    Args:
        results: [(success, history), ...] のリスト

    Returns:
        ASR (0.0 - 1.0)
    """
    if not results:
        return 0.0
    n_success = sum(1 for success, _ in results if success)
    return n_success / len(results)


def compute_avg_queries(results: list[tuple[bool, list[AttackResult]]]) -> tuple[float, float]:
    """成功時の平均クエリ数と標準偏差を計算する.

    Args:
        results: [(success, history), ...] のリスト

    Returns:
        (平均クエリ数, 標準偏差)
    """
    query_counts = [
        len(history) for success, history in results if success and history
    ]
    if not query_counts:
        return 0.0, 0.0
    return float(np.mean(query_counts)), float(np.std(query_counts))


def compute_stage_failure_distribution(
    results: list[tuple[bool, list[AttackResult]]]
) -> dict[str, float]:
    """ステージ別失敗分布を計算する.

    Args:
        results: [(success, history), ...] のリスト

    Returns:
        {stage: failure_rate} の辞書
    """
    stage_counts: dict[Stage, int] = {
        Stage.INPUT_GUARD: 0,
        Stage.TARGET_LLM: 0,
        Stage.OUTPUT_GUARD: 0,
        Stage.PASSED_GUARDS: 0,
    }
    total = 0

    for success, history in results:
        if not history:
            continue
        # 最後の試行の失敗ステージをカウント
        last_result = history[-1]
        if not success:
            stage_counts[last_result.stage_result.failed_stage] += 1
        else:
            stage_counts[Stage.PASSED_GUARDS] += 1
        total += 1

    if total == 0:
        return {s.value: 0.0 for s in stage_counts}

    return {s.value: count / total for s, count in stage_counts.items()}


def compute_evaluation_metrics(
    results: list[tuple[bool, list[AttackResult]]]
) -> EvaluationMetrics:
    """全評価メトリクスを計算する.

    Args:
        results: [(success, history), ...] のリスト

    Returns:
        EvaluationMetrics インスタンス
    """
    asr = compute_asr(results)
    avg_queries, std_queries = compute_avg_queries(results)
    stage_dist = compute_stage_failure_distribution(results)
    n_success = sum(1 for success, _ in results if success)

    return EvaluationMetrics(
        asr=asr,
        avg_queries=avg_queries,
        std_queries=std_queries,
        stage_failure_dist=stage_dist,
        n_total=len(results),
        n_success=n_success,
    )


def compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """信頼区間を計算する.

    Args:
        values: 値のリスト
        confidence: 信頼水準（デフォルト 95%）

    Returns:
        (下限, 上限) のタプル
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)

    mean = np.mean(values)
    std_err = stats.sem(values)
    ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=std_err)
    return (float(ci[0]), float(ci[1]))


def compute_bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 法で信頼区間を計算する.

    Args:
        values: 値のリスト
        n_bootstrap: ブートストラップ回数
        confidence: 信頼水準
        seed: ランダムシード

    Returns:
        (下限, 上限) のタプル
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)

    rng = np.random.default_rng(seed)
    values_arr = np.array(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def compare_asr(
    results_a: list[tuple[bool, list[AttackResult]]],
    results_b: list[tuple[bool, list[AttackResult]]],
) -> dict:
    """2 つの結果セットの ASR を比較する.

    Args:
        results_a: 結果 A（[(success, history), ...]）
        results_b: 結果 B

    Returns:
        比較結果の辞書（p値、差分、信頼区間など）
    """
    successes_a = [1 if s else 0 for s, _ in results_a]
    successes_b = [1 if s else 0 for s, _ in results_b]

    asr_a = np.mean(successes_a)
    asr_b = np.mean(successes_b)
    diff = asr_a - asr_b

    # McNemar's test（対応のある二値データ）
    # または Fisher's exact test（独立サンプル）
    # ここでは独立サンプルと仮定して chi2 test を使用
    contingency = [
        [sum(successes_a), len(successes_a) - sum(successes_a)],
        [sum(successes_b), len(successes_b) - sum(successes_b)],
    ]
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    return {
        "asr_a": float(asr_a),
        "asr_b": float(asr_b),
        "difference": float(diff),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def format_metrics_table(
    metrics_dict: dict[str, EvaluationMetrics]
) -> str:
    """メトリクスをテーブル形式でフォーマットする.

    Args:
        metrics_dict: {agent_name: metrics} の辞書

    Returns:
        フォーマットされたテーブル文字列
    """
    header = f"{'Agent':<15} {'ASR':>8} {'Avg Q':>8} {'Std Q':>8} {'N':>6} {'Success':>8}"
    separator = "-" * len(header)

    rows = [header, separator]
    for name, m in metrics_dict.items():
        row = (
            f"{name:<15} {m.asr:>8.1%} {m.avg_queries:>8.1f} "
            f"{m.std_queries:>8.1f} {m.n_total:>6} {m.n_success:>8}"
        )
        rows.append(row)

    return "\n".join(rows)
