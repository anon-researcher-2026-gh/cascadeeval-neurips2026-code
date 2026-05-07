"""C3A: Cascade-Aware Adaptive Agent.

多段防御システム（Input Guard × Target LLM × Output Guard）に対する
適応的攻撃エージェントを実装するパッケージ。
"""

from src.c3a.types import (
    AttackResult,
    DefenseConfig,
    INPUT_GUARDS,
    IntentName,
    InputGuardName,
    OUTPUT_GUARDS,
    OutputGuardName,
    Stage,
    StageResult,
    TARGET_LLMS,
    TECHNIQUES,
    TargetLLMName,
    TechniqueName,
)

__all__ = [
    # 型
    "Stage",
    "InputGuardName",
    "TargetLLMName",
    "OutputGuardName",
    "TechniqueName",
    "IntentName",
    # データ構造
    "DefenseConfig",
    "StageResult",
    "AttackResult",
    # 定数
    "INPUT_GUARDS",
    "TARGET_LLMS",
    "OUTPUT_GUARDS",
    "TECHNIQUES",
]
