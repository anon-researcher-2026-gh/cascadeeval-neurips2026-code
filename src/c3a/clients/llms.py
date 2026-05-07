"""Target LLM クライアント.

GPT-4o, GPT-4o-mini, Gemini, Llama, Qwen, Mistral, Gemma のラッパー。
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

# 短縮名
TargetLLMType = Literal[
    "gpt-4o", "gpt-4o-mini", "gemini-2.0-flash",
    "Llama-3.1-8B", "Qwen2.5-7B", "Qwen2.5-14B", "Qwen3-8B",
    "Ministral-8B", "gemma-3-12b"
]

# フルモデルID（データカラム名と同じ）
TargetLLMFullName = Literal[
    "gpt-4o", "gpt-4o-mini", "gemini-2.0-flash",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen3-8B",
    "mistralai_Ministral-8B-Instruct-2410",
    "google_gemma-3-12b-it",
]


@dataclass
class TargetLLMConfig:
    """Target LLM クライアント設定.

    Attributes:
        llm_type: LLM の種類
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        timeout: タイムアウト（秒）
        temperature: 生成温度
        max_tokens: 最大トークン数
    """

    llm_type: str  # TargetLLMType or TargetLLMFullName
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048


# LLM タイプと API 情報のマッピング（短縮名 + フル名両対応）
LLM_CONFIG_MAP: dict[str, dict[str, str]] = {
    # OpenAI
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    # Google Gemini
    "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash"},
    # ローカルモデル（短縮名）
    "Llama-3.1-8B": {"provider": "local", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    "Qwen2.5-7B": {"provider": "local", "model": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen2.5-14B": {"provider": "local", "model": "Qwen/Qwen2.5-14B-Instruct"},
    "Qwen3-8B": {"provider": "local", "model": "Qwen/Qwen3-8B"},
    "Ministral-8B": {"provider": "local", "model": "mistralai/Ministral-8B-Instruct-2410"},
    "gemma-3-12b": {"provider": "local", "model": "google/gemma-3-12b-it"},
    # ローカルモデル（フル名）
    "meta-llama_Llama-3.1-8B-Instruct": {"provider": "local", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    "Qwen_Qwen2.5-7B-Instruct": {"provider": "local", "model": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen_Qwen2.5-14B-Instruct": {"provider": "local", "model": "Qwen/Qwen2.5-14B-Instruct"},
    "Qwen_Qwen3-8B": {"provider": "local", "model": "Qwen/Qwen3-8B"},
    "mistralai_Ministral-8B-Instruct-2410": {"provider": "local", "model": "mistralai/Ministral-8B-Instruct-2410"},
    "google_gemma-3-12b-it": {"provider": "local", "model": "google/gemma-3-12b-it"},
    # GPT-OSS-20B（短縮名 + フル名）
    "GPT-OSS-20B": {"provider": "local", "model": "openai/gpt-oss-20b"},
    "openai_gpt-oss-20b": {"provider": "local", "model": "openai/gpt-oss-20b"},
    # Phi-4
    "phi-4": {"provider": "local", "model": "microsoft/phi-4"},
    "microsoft_phi-4": {"provider": "local", "model": "microsoft/phi-4"},
}


@dataclass
class TargetLLMClient:
    """Target LLM クライアント.

    LLMClient プロトコルを実装する。
    """

    config: TargetLLMConfig
    _client: OpenAI | None = field(default=None, repr=False)
    _model: str = field(default="", repr=False)
    _total_calls: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """クライアントを初期化する."""
        llm_config = LLM_CONFIG_MAP.get(self.config.llm_type)
        if llm_config is None:
            raise ValueError(f"Unknown LLM type: {self.config.llm_type}")

        provider = llm_config["provider"]
        self._model = llm_config["model"]

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)

        elif provider == "google":
            # Google Gemini は OpenAI 互換 API を使用
            api_key = os.environ.get("GOOGLE_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not api_key:
                logger.warning("GOOGLE_API_KEY is not set")
            self._client = OpenAI(
                api_key=api_key or "dummy",
                base_url=base_url,
                timeout=self.config.timeout,
            )

        elif provider == "local":
            # ローカルサーバーを使用（vLLM, Ollama 等）
            base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://localhost:8000/v1")
            api_key = os.environ.get("LOCAL_MODEL_API_KEY", "dummy")
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """テキスト生成を実行する.

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト

        Returns:
            生成されたテキスト
        """
        if self._client is None:
            raise RuntimeError("LLM client is not initialized")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                self._total_calls += 1
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    return ""
                return content

            except RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})"
                )
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                logger.error(f"LLM API call failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise RuntimeError(f"Failed after retries: {last_error}")

    @property
    def total_calls(self) -> int:
        """総 API 呼び出し回数を返す."""
        return self._total_calls


def create_target_llm_client(
    llm_type: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> TargetLLMClient:
    """Target LLM クライアントを作成する便利関数.

    Args:
        llm_type: LLM の種類（短縮名 or フル名）
        temperature: 生成温度
        max_tokens: 最大トークン数

    Returns:
        TargetLLMClient インスタンス
    """
    config = TargetLLMConfig(
        llm_type=llm_type,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return TargetLLMClient(config=config)
