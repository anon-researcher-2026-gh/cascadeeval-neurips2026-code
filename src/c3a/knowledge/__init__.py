"""C3A Knowledge Layer.

Part 1 分析結果を管理し、Part 2 の攻撃生成にガイダンスを提供する。
"""

from src.c3a.knowledge.kb_loader import (
    Part1KnowledgeBase,
    StrategyGuidance,
    load_knowledge_base,
)
from src.c3a.knowledge.embeddings import (
    compute_embedding,
    compute_embeddings_batch,
    save_embeddings,
    load_embeddings,
)

__all__ = [
    "Part1KnowledgeBase",
    "StrategyGuidance",
    "load_knowledge_base",
    "compute_embedding",
    "compute_embeddings_batch",
    "save_embeddings",
    "load_embeddings",
]
