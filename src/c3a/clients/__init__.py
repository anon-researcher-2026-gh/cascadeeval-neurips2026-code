"""C3A API クライアント層.

LLM および Guard API へのアクセスを提供する。
"""

from src.c3a.clients.base import GuardClient, LLMClient
from src.c3a.clients.guards import (
    GuardClientConfig,
    GuardType,
    create_guard_client,
)
from src.c3a.clients.guards import GuardClient as GuardClientImpl
from src.c3a.clients.huggingface import (
    GUARD_MODEL_IDS,
    TARGET_LLM_MODEL_IDS,
    HuggingFaceClientConfig,
    HuggingFaceGuardClient,
    HuggingFaceLLMClient,
    create_huggingface_guard_client,
    create_huggingface_llm_client,
)
from src.c3a.clients.llms import (
    TargetLLMConfig,
    TargetLLMType,
    create_target_llm_client,
)
from src.c3a.clients.llms import TargetLLMClient
from src.c3a.clients.openai import (
    OpenAIClient,
    OpenAIClientConfig,
    create_openai_client,
)

__all__ = [
    # プロトコル
    "LLMClient",
    "GuardClient",
    # OpenAI
    "OpenAIClient",
    "OpenAIClientConfig",
    "create_openai_client",
    # Guard (API)
    "GuardClientImpl",
    "GuardClientConfig",
    "GuardType",
    "create_guard_client",
    # Target LLM (API)
    "TargetLLMClient",
    "TargetLLMConfig",
    "TargetLLMType",
    "create_target_llm_client",
    # HuggingFace (ローカル)
    "HuggingFaceClientConfig",
    "HuggingFaceLLMClient",
    "HuggingFaceGuardClient",
    "create_huggingface_llm_client",
    "create_huggingface_guard_client",
    "GUARD_MODEL_IDS",
    "TARGET_LLM_MODEL_IDS",
]
