"""CSV カラム名マッピング定義.

en_results.csv のカラム名を定数として定義する。
"""

from typing import Final

from src.c3a.types import INPUT_GUARDS, TARGET_LLMS

# =============================================================================
# 識別カラム
# =============================================================================

UID: Final[str] = "unique_id"
ORIGINAL_ID: Final[str] = "original_id"
PROMPT_ID: Final[str] = "prompt_id"
PROMPT_TEXT_EN: Final[str] = "prompt_text_en"
CHAR_COUNT: Final[str] = "char_count"
RANK: Final[str] = "rank"

# =============================================================================
# Intent / Technique カラム
# =============================================================================

INTENT: Final[str] = "taxonomy_intent"

TECHNIQUE_LEX: Final[str] = "taxonomy_technique_LEX"
TECHNIQUE_MSYN: Final[str] = "taxonomy_technique_MSYN"
TECHNIQUE_SEM: Final[str] = "taxonomy_technique_SEM"
TECHNIQUE_PRAG: Final[str] = "taxonomy_technique_PRAG"
TECHNIQUE_ORTH: Final[str] = "taxonomy_technique_ORTH"

TECHNIQUE_COLUMNS: Final[tuple[str, ...]] = (
    TECHNIQUE_LEX,
    TECHNIQUE_MSYN,
    TECHNIQUE_SEM,
    TECHNIQUE_PRAG,
    TECHNIQUE_ORTH,
)


# =============================================================================
# Input Guard カラム
# =============================================================================

def get_input_guard_column(ig_name: str) -> str:
    """Input Guard のカラム名を生成する.

    Args:
        ig_name: Input Guard のモデルID

    Returns:
        カラム名（例: "input_guard_allenai_wildguard_is_safe"）
    """
    return f"input_guard_{ig_name}_is_safe"


# Input Guard カラム名の順序（INPUT_GUARDS と同順）
INPUT_GUARD_COLUMNS: Final[tuple[str, ...]] = tuple(
    get_input_guard_column(ig) for ig in INPUT_GUARDS
)


# =============================================================================
# Target LLM カラム
# =============================================================================

def get_llm_responded_column(llm_name: str) -> str:
    """Target LLM の responded カラム名を生成する.

    Args:
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "llm_gpt-4o_responded"）
    """
    return f"llm_{llm_name}_responded"


def get_llm_refusal_column(llm_name: str) -> str:
    """Target LLM の refusal カラム名を生成する.

    Args:
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "llm_gpt-4o_is_refusal"）
    """
    return f"llm_{llm_name}_is_refusal"


# Target LLM responded カラム名の順序（TARGET_LLMS と同順）
TARGET_LLM_COLUMNS: Final[tuple[str, ...]] = tuple(
    get_llm_responded_column(llm) for llm in TARGET_LLMS
)

TARGET_LLM_REFUSAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    get_llm_refusal_column(llm) for llm in TARGET_LLMS
)


# =============================================================================
# Output Guard カラム
# =============================================================================

def get_output_guard_column(og_name: str, llm_name: str) -> str:
    """Output Guard のカラム名を生成する.

    Args:
        og_name: Output Guard のモデルID
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "output_guard_allenai_wildguard_on_gpt-4o_is_safe"）
    """
    return f"output_guard_{og_name}_on_{llm_name}_is_safe"


# =============================================================================
# Multi-Judge カラム
# =============================================================================

def get_multijudge_harmful_column(llm_name: str) -> str:
    """Multi-Judge の is_harmful カラム名を生成する.

    Args:
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "multijudge_on_gpt-4o_is_harmful"）
    """
    return f"multijudge_on_{llm_name}_is_harmful"


def get_multijudge_score_column(llm_name: str) -> str:
    """Multi-Judge の strongreject_score カラム名を生成する.

    Args:
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "multijudge_on_gpt-4o_strongreject_score"）
    """
    return f"multijudge_on_{llm_name}_strongreject_score"


def get_multijudge_category_column(llm_name: str) -> str:
    """Multi-Judge の category カラム名を生成する.

    Args:
        llm_name: Target LLM のモデルID

    Returns:
        カラム名（例: "multijudge_on_gpt-4o_category"）
    """
    return f"multijudge_on_{llm_name}_category"
