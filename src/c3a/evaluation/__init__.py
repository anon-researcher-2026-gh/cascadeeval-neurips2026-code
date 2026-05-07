"""C3A 評価層.

オンライン評価と Jailbreak 判定を提供する。
"""

from src.c3a.evaluation.analysis import (
    AnalysisResult,
    compute_asr_matrix,
    compute_bottleneck_distribution,
    compute_config_rankings,
    compute_stage_block_rates,
    compute_technique_effectiveness,
    print_analysis_summary,
    run_full_analysis,
)
from src.c3a.evaluation.estimator import StageEstimator
from src.c3a.evaluation.experiment_manager import (
    ConfigResultWriter,
    ExperimentManager,
    aggregate_all_results,
)
from src.c3a.evaluation.judge import JailbreakJudge
from src.c3a.evaluation.metrics import (
    EvaluationMetrics,
    compare_asr,
    compute_asr,
    compute_avg_queries,
    compute_bootstrap_ci,
    compute_confidence_interval,
    compute_evaluation_metrics,
    compute_stage_failure_distribution,
    format_metrics_table,
)
from src.c3a.evaluation.online import OnlineEvaluator
from src.c3a.evaluation.visualize import (
    create_all_figures,
    plot_asr_heatmap,
    plot_bottleneck_distribution,
    plot_config_ranking,
    plot_stage_block_rates,
    plot_technique_effectiveness,
)

__all__ = [
    # 分析
    "AnalysisResult",
    "run_full_analysis",
    "print_analysis_summary",
    "compute_asr_matrix",
    "compute_stage_block_rates",
    "compute_technique_effectiveness",
    "compute_bottleneck_distribution",
    "compute_config_rankings",
    # 可視化
    "create_all_figures",
    "plot_asr_heatmap",
    "plot_stage_block_rates",
    "plot_technique_effectiveness",
    "plot_bottleneck_distribution",
    "plot_config_ranking",
    # オンライン評価
    "OnlineEvaluator",
    "JailbreakJudge",
    "StageEstimator",
    # 実験管理
    "ExperimentManager",
    "ConfigResultWriter",
    "aggregate_all_results",
    # メトリクス
    "EvaluationMetrics",
    "compute_evaluation_metrics",
    "compute_asr",
    "compute_avg_queries",
    "compute_stage_failure_distribution",
    "compute_confidence_interval",
    "compute_bootstrap_ci",
    "compare_asr",
    "format_metrics_table",
]
