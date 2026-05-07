"""C3A エージェント層.

攻撃生成とエージェント本体を提供する。
"""

from src.c3a.agent.c3a import C3AAgent, C3AConfig, create_c3a_agent
from src.c3a.agent.generator import (
    AGENT_ABLATION_MODES,
    AblationMode,
    AttackGenerator,
    GeneratorConfig,
    STAGE_STRATEGIES,
    create_attack_generator,
)

__all__ = [
    # Agent
    "C3AAgent",
    "C3AConfig",
    "create_c3a_agent",
    # Generator
    "AttackGenerator",
    "GeneratorConfig",
    "AblationMode",
    "create_attack_generator",
    "AGENT_ABLATION_MODES",
    "STAGE_STRATEGIES",
]
