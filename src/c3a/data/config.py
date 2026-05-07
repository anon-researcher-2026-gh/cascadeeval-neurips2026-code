"""防御構成マネージャー.

225構成（5 IG × 9 LLM × 5 OG）の生成と管理を提供する。
gemini-2.0-flash を除くと 200 構成（5 IG × 8 LLM × 5 OG）。
"""

from typing import Iterator

from src.c3a.types import (
    DefenseConfig,
    INPUT_GUARDS,
    InputGuardName,
    OUTPUT_GUARDS,
    OutputGuardName,
    Stage,
    TARGET_LLMS,
    TargetLLMName,
)


def generate_all_configs(exclude_gemini: bool = True) -> list[DefenseConfig]:
    """全構成を生成する.

    Args:
        exclude_gemini: gemini-2.0-flash を除外するか

    Returns:
        構成のリスト
    """
    configs: list[DefenseConfig] = []
    for ig in INPUT_GUARDS:
        for llm in TARGET_LLMS:
            # gemini は除外オプション
            if exclude_gemini and llm == "gemini-2.0-flash":
                continue
            for og in OUTPUT_GUARDS:
                configs.append(DefenseConfig(ig, llm, og))
    return configs


# 全構成の定数（gemini 除外で 200 構成）
ALL_CONFIGS: list[DefenseConfig] = generate_all_configs(exclude_gemini=True)

# gemini を含む全構成（225 構成）
ALL_CONFIGS_WITH_GEMINI: list[DefenseConfig] = generate_all_configs(exclude_gemini=False)


def get_config_by_name(name: str) -> DefenseConfig | None:
    """構成名から DefenseConfig を取得する.

    Args:
        name: 構成名（パイプ区切り形式）

    Returns:
        対応する DefenseConfig、見つからない場合は None
    """
    try:
        return DefenseConfig.from_name(name)
    except ValueError:
        return None


def get_config(
    ig: InputGuardName, llm: TargetLLMName, og: OutputGuardName
) -> DefenseConfig:
    """コンポーネント名から DefenseConfig を取得する.

    Args:
        ig: Input Guard 名
        llm: Target LLM 名
        og: Output Guard 名

    Returns:
        対応する DefenseConfig
    """
    return DefenseConfig(ig, llm, og)


def filter_configs_by_ig(ig: InputGuardName, configs: list[DefenseConfig] | None = None) -> list[DefenseConfig]:
    """指定した Input Guard を持つ構成をフィルタリングする.

    Args:
        ig: Input Guard 名
        configs: 対象構成リスト（None の場合は ALL_CONFIGS）

    Returns:
        該当する構成のリスト
    """
    if configs is None:
        configs = ALL_CONFIGS
    return [c for c in configs if c.input_guard == ig]


def filter_configs_by_llm(llm: TargetLLMName, configs: list[DefenseConfig] | None = None) -> list[DefenseConfig]:
    """指定した Target LLM を持つ構成をフィルタリングする.

    Args:
        llm: Target LLM 名
        configs: 対象構成リスト（None の場合は ALL_CONFIGS）

    Returns:
        該当する構成のリスト
    """
    if configs is None:
        configs = ALL_CONFIGS
    return [c for c in configs if c.target_llm == llm]


def filter_configs_by_og(og: OutputGuardName, configs: list[DefenseConfig] | None = None) -> list[DefenseConfig]:
    """指定した Output Guard を持つ構成をフィルタリングする.

    Args:
        og: Output Guard 名
        configs: 対象構成リスト（None の場合は ALL_CONFIGS）

    Returns:
        該当する構成のリスト
    """
    if configs is None:
        configs = ALL_CONFIGS
    return [c for c in configs if c.output_guard == og]


def iter_configs(exclude_gemini: bool = True) -> Iterator[DefenseConfig]:
    """全構成をイテレートする."""
    if exclude_gemini:
        yield from ALL_CONFIGS
    else:
        yield from ALL_CONFIGS_WITH_GEMINI


def get_config_indices(configs: list[DefenseConfig] | None = None) -> dict[str, int]:
    """構成名からインデックスへのマッピングを返す."""
    if configs is None:
        configs = ALL_CONFIGS
    return {c.name: i for i, c in enumerate(configs)}


def get_component_index(stage: Stage, name: str) -> int:
    """コンポーネント名からインデックスを取得する.

    Args:
        stage: ステージ（INPUT_GUARD, TARGET_LLM, OUTPUT_GUARD）
        name: コンポーネント名

    Returns:
        インデックス

    Raises:
        ValueError: 不正なステージまたは名前の場合
    """
    if stage == Stage.INPUT_GUARD:
        return INPUT_GUARDS.index(name)  # type: ignore[arg-type]
    elif stage == Stage.TARGET_LLM:
        return TARGET_LLMS.index(name)  # type: ignore[arg-type]
    elif stage == Stage.OUTPUT_GUARD:
        return OUTPUT_GUARDS.index(name)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Invalid stage for component index: {stage}")
