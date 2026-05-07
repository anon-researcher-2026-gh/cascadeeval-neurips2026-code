"""API クライアントの基底プロトコル定義."""

from typing import Protocol


class LLMClient(Protocol):
    """LLM API クライアントのプロトコル."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """テキスト生成を実行する.

        Args:
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（オプション）

        Returns:
            生成されたテキスト
        """
        ...


class GuardClient(Protocol):
    """Guard API クライアントのプロトコル."""

    def classify(self, text: str) -> bool:
        """テキストを分類する.

        Args:
            text: 分類対象のテキスト

        Returns:
            True = safe（通過）, False = unsafe（ブロック）
        """
        ...

    def classify_input(self, prompt: str) -> bool:
        """入力プロンプトを分類する.

        Args:
            prompt: 分類対象のプロンプト

        Returns:
            True = safe, False = unsafe
        """
        ...

    def classify_output(self, prompt: str, response: str) -> bool:
        """出力応答を分類する.

        Args:
            prompt: 元のプロンプト
            response: LLM の応答

        Returns:
            True = safe, False = unsafe
        """
        ...
