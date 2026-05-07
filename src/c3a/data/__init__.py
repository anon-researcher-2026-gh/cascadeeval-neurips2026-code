"""C3A データ層.

データの読み込みと防御構成の管理を提供する。
"""

from src.c3a.data import columns
from src.c3a.data.config import (
    ALL_CONFIGS,
    filter_configs_by_ig,
    filter_configs_by_llm,
    filter_configs_by_og,
    generate_all_configs,
    get_component_index,
    get_config,
    get_config_by_name,
    get_config_indices,
    iter_configs,
)
from src.c3a.data.loader import (
    AttackDataset,
    compute_pass_rate,
    compute_stage_block_rates,
    get_stage_result_from_labels,
    load_dataset,
)

__all__ = [
    # モジュール
    "columns",
    # データローダー
    "AttackDataset",
    "load_dataset",
    "compute_pass_rate",
    "compute_stage_block_rates",
    "get_stage_result_from_labels",
    # 構成管理
    "ALL_CONFIGS",
    "generate_all_configs",
    "get_config",
    "get_config_by_name",
    "get_config_indices",
    "filter_configs_by_ig",
    "filter_configs_by_llm",
    "filter_configs_by_og",
    "iter_configs",
    "get_component_index",
]
