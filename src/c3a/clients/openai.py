"""OpenAI API クライアント."""

import logging
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class OpenAIClientConfig:
    """OpenAI クライアント設定.

    Attributes:
        model: モデル名
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        timeout: タイムアウト（秒）
        temperature: 生成温度
        max_tokens: 最大トークン数
        seed: 再現性のためのシード値（None で毎回異なる出力）
    """

    model: str = "gpt-4o-mini"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: int | None = None


@dataclass
class OpenAIClient:
    """OpenAI API クライアント.

    LLMClient プロトコルを実装する。
    """

    config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    _client: OpenAI | None = field(default=None, repr=False)
    _total_calls: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """クライアントを初期化する."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)

    def generate(
        self, prompt: str, system_prompt: str | None = None, seed: int | None = None
    ) -> str:
        """テキスト生成を実行する.

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト
            seed: このリクエスト用のシード値（None で config のデフォルトを使用）

        Returns:
            生成されたテキスト

        Raises:
            RuntimeError: API 呼び出しが失敗した場合
        """
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # seed: 引数 > config > None
        effective_seed = seed if seed is not None else self.config.seed

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                self._total_calls += 1
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=effective_seed,
                )
                content = response.choices[0].message.content
                if content is None:
                    return ""
                return content

            except RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{self.config.max_retries})"
                )
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                logger.error(f"API call failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise RuntimeError(f"Failed after {self.config.max_retries} retries: {last_error}")

    @property
    def total_calls(self) -> int:
        """総 API 呼び出し回数を返す."""
        return self._total_calls

    def reset_call_count(self) -> None:
        """呼び出し回数をリセットする."""
        self._total_calls = 0


def create_openai_client(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: int | None = None,
) -> OpenAIClient:
    """OpenAI クライアントを作成する便利関数.

    Args:
        model: モデル名
        temperature: 生成温度
        max_tokens: 最大トークン数
        seed: 再現性のためのシード値（None で毎回異なる出力）

    Returns:
        OpenAIClient インスタンス
    """
    config = OpenAIClientConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    return OpenAIClient(config=config)
