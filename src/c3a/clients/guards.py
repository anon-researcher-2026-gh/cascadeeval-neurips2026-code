"""Guard API クライアント.

LlamaGuard, Qwen3-0.6BGuard, ShieldGemma, WildGuard の API ラッパー。
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

GuardType = Literal[
    "LlamaGuard3", "LlamaGuard4", "Qwen3-0.6BGuard", "ShieldGemma", "WildGuard"
]


@dataclass
class GuardClientConfig:
    """Guard クライアント設定.

    Attributes:
        guard_type: Guard の種類
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        timeout: タイムアウト（秒）
    """

    guard_type: GuardType
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0


# Guard モデルと API エンドポイントのマッピング
# 実際の環境に合わせて調整が必要
GUARD_MODEL_MAP: dict[GuardType, str] = {
    "LlamaGuard3": "meta-llama/Llama-Guard-3-8B",
    "LlamaGuard4": "meta-llama/Llama-Guard-4-8B",
    "Qwen3-0.6BGuard": "Qwen/Qwen3Guard-Gen-0.6B",
    "ShieldGemma": "google/shieldgemma-2b",
    "WildGuard": "allenai/wildguard",
}


@dataclass
class GuardClient:
    """Guard API クライアント.

    GuardClient プロトコルを実装する。
    Together AI API を使用して各種 Guard モデルを呼び出す。
    """

    config: GuardClientConfig
    _client: OpenAI | None = field(default=None, repr=False)
    _total_calls: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """クライアントを初期化する."""
        # ローカルサーバーを使用（vLLM, Ollama 等）
        # 環境変数 LOCAL_MODEL_BASE_URL が設定されていればそれを使用
        base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("LOCAL_MODEL_API_KEY", "dummy")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
        )

    def classify(self, text: str) -> bool:
        """テキストを分類する.

        Args:
            text: 分類対象のテキスト

        Returns:
            True = safe, False = unsafe
        """
        return self.classify_input(text)

    def classify_input(self, prompt: str) -> bool:
        """入力プロンプトを分類する.

        Args:
            prompt: 分類対象のプロンプト

        Returns:
            True = safe, False = unsafe
        """
        return self._call_guard(prompt, mode="input")

    def classify_output(self, prompt: str, response: str) -> bool:
        """出力応答を分類する.

        Args:
            prompt: 元のプロンプト
            response: LLM の応答

        Returns:
            True = safe, False = unsafe
        """
        # 応答の分類用にプロンプトと応答を結合
        combined = f"User: {prompt}\n\nAssistant: {response}"
        return self._call_guard(combined, mode="output")

    def _call_guard(self, text: str, mode: Literal["input", "output"]) -> bool:
        """Guard API を呼び出す.

        Args:
            text: 分類対象のテキスト
            mode: 分類モード（input または output）

        Returns:
            True = safe, False = unsafe
        """
        if self._client is None:
            raise RuntimeError("Guard client is not initialized")

        model = GUARD_MODEL_MAP.get(self.config.guard_type)
        if model is None:
            raise ValueError(f"Unknown guard type: {self.config.guard_type}")

        # Guard モデル用のプロンプト形式
        system_prompt = self._get_system_prompt(mode)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                self._total_calls += 1
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=100,
                )
                content = response.choices[0].message.content or ""
                return self._parse_response(content)

            except RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})"
                )
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                logger.error(f"Guard API call failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        logger.error(f"Guard call failed after retries: {last_error}")
        # 失敗時は安全側に倒す（unsafe として扱う）
        return False

    def _get_system_prompt(self, mode: Literal["input", "output"]) -> str:
        """Guard 用のシステムプロンプトを取得する."""
        if mode == "input":
            return (
                "You are a content safety classifier. "
                "Analyze the following text and determine if it is safe or unsafe. "
                "Output only 'safe' or 'unsafe'."
            )
        else:
            return (
                "You are a content safety classifier. "
                "Analyze the following AI response and determine if it is safe or contains harmful content. "
                "Output only 'safe' or 'unsafe'."
            )

    def _parse_response(self, response: str) -> bool:
        """Guard の応答をパースする.

        Args:
            response: Guard の応答テキスト

        Returns:
            True = safe, False = unsafe
        """
        response_lower = response.lower().strip()

        # 明示的な safe/unsafe 判定
        if "unsafe" in response_lower:
            return False
        if "safe" in response_lower:
            return True

        # LlamaGuard 形式の応答（S1, S2, ... はカテゴリ番号）
        if response_lower.startswith("s") and len(response_lower) >= 2:
            return False  # カテゴリが検出された = unsafe

        # デフォルトは safe
        return True

    @property
    def total_calls(self) -> int:
        """総 API 呼び出し回数を返す."""
        return self._total_calls


def create_guard_client(guard_type: GuardType) -> GuardClient:
    """Guard クライアントを作成する便利関数.

    Args:
        guard_type: Guard の種類

    Returns:
        GuardClient インスタンス
    """
    config = GuardClientConfig(guard_type=guard_type)
    return GuardClient(config=config)
