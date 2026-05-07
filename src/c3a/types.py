"""C3A 型定義モジュール.

3段階防御システム（Input Guard × Target LLM × Output Guard）に関する
型定義を提供する。

注: モデル ID は en_results.csv のカラム名に基づく。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Stage(Enum):
    """防御ステージの列挙型."""

    INPUT_GUARD = "IG"
    TARGET_LLM = "LLM"
    OUTPUT_GUARD = "OG"
    PASSED_GUARDS = "PASSED_GUARDS"  # 全ガード通過（Judge判定待ち）


class JudgmentCategory(Enum):
    """Judge の判定カテゴリ.

    Deep Research "Fictional Framing" の知見に基づく判定分類:
    - JAILBREAK: 実際に有害なコンテンツを生成（ターゲット一致）
    - FICTIONAL_COMPLIANCE: 架空エンティティへの置換による回避（False Positive）
    - REFUSAL: 明示的な拒否
    - PARTIAL: 部分的な回答（教育的文脈など）
    """

    JAILBREAK = "JAILBREAK"  # 真の Jailbreak 成功
    FICTIONAL_COMPLIANCE = "FICTIONAL"  # 架空エンティティで回避（Safe Hallucination）
    REFUSAL = "REFUSAL"  # 拒否
    PARTIAL = "PARTIAL"  # 部分的回答


# 各コンポーネントの名前リテラル型（実データのモデルID）
InputGuardName = Literal[
    "allenai_wildguard",
    "meta-llama_Llama-Guard-3-8B",
    "meta-llama_Llama-Guard-4-12B",
    "Qwen_Qwen3Guard-Gen-0.6B",
    "google_shieldgemma-2b",
]

TargetLLMName = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.0-flash",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen3-8B",
    "mistralai_Ministral-8B-Instruct-2410",
    "google_gemma-3-12b-it",
]

OutputGuardName = Literal[
    "allenai_wildguard",
    "meta-llama_Llama-Guard-3-8B",
    "meta-llama_Llama-Guard-4-12B",
    "Qwen_Qwen3Guard-Gen-0.6B",
    "google_shieldgemma-2b",
]

TechniqueName = Literal["LEX", "MSYN", "SEM", "PRAG", "ORTH"]
IntentName = Literal["IL", "MCG", "PD"]

# 各コンポーネントのリスト（CSVカラム名の順序に対応）
INPUT_GUARDS: list[InputGuardName] = [
    "allenai_wildguard",
    "meta-llama_Llama-Guard-3-8B",
    "meta-llama_Llama-Guard-4-12B",
    "Qwen_Qwen3Guard-Gen-0.6B",
    "google_shieldgemma-2b",
]

TARGET_LLMS: list[TargetLLMName] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.0-flash",
    "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen3-8B",
    "mistralai_Ministral-8B-Instruct-2410",
    "google_gemma-3-12b-it",
]

OUTPUT_GUARDS: list[OutputGuardName] = [
    "allenai_wildguard",
    "meta-llama_Llama-Guard-3-8B",
    "meta-llama_Llama-Guard-4-12B",
    "Qwen_Qwen3Guard-Gen-0.6B",
    "google_shieldgemma-2b",
]

TECHNIQUES: list[TechniqueName] = ["LEX", "MSYN", "SEM", "PRAG", "ORTH"]

# 短縮名からフルモデルIDへのマッピング（複数のエイリアスをサポート）
SHORT_TO_FULL_IG: dict[str, InputGuardName] = {
    # 標準形式
    "wildguard": "allenai_wildguard",
    "WildGuard": "allenai_wildguard",
    "LlamaGuard3": "meta-llama_Llama-Guard-3-8B",
    "LlamaGuard4": "meta-llama_Llama-Guard-4-12B",
    "Qwen3Guard": "Qwen_Qwen3Guard-Gen-0.6B",
    "Qwen3-0.6BGuard": "Qwen_Qwen3Guard-Gen-0.6B",
    "ShieldGemma": "google_shieldgemma-2b",
    # フルID
    "allenai_wildguard": "allenai_wildguard",
    "meta-llama_Llama-Guard-3-8B": "meta-llama_Llama-Guard-3-8B",
    "meta-llama_Llama-Guard-4-12B": "meta-llama_Llama-Guard-4-12B",
    "Qwen_Qwen3Guard-Gen-0.6B": "Qwen_Qwen3Guard-Gen-0.6B",
    "google_shieldgemma-2b": "google_shieldgemma-2b",
}

SHORT_TO_FULL_LLM: dict[str, TargetLLMName] = {
    # 標準形式
    "gpt-4o": "gpt-4o",
    "GPT-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "GPT-4o-mini": "gpt-4o-mini",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Llama-3.1-8B": "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-14B": "Qwen_Qwen2.5-14B-Instruct",
    "Qwen3-8B": "Qwen_Qwen3-8B",
    "Ministral-8B": "mistralai_Ministral-8B-Instruct-2410",
    "gemma-3-12b": "google_gemma-3-12b-it",
    # フルID
    "meta-llama_Llama-3.1-8B-Instruct": "meta-llama_Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B-Instruct": "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen3-8B": "Qwen_Qwen3-8B",
    "mistralai_Ministral-8B-Instruct-2410": "mistralai_Ministral-8B-Instruct-2410",
    "google_gemma-3-12b-it": "google_gemma-3-12b-it",
}

SHORT_TO_FULL_OG: dict[str, OutputGuardName] = SHORT_TO_FULL_IG  # type: ignore

# フルモデルIDから短縮名へのマッピング
FULL_TO_SHORT_IG: dict[InputGuardName, str] = {v: k for k, v in SHORT_TO_FULL_IG.items()}
FULL_TO_SHORT_LLM: dict[TargetLLMName, str] = {v: k for k, v in SHORT_TO_FULL_LLM.items()}
FULL_TO_SHORT_OG: dict[OutputGuardName, str] = {v: k for k, v in SHORT_TO_FULL_OG.items()}


@dataclass(frozen=True)
class DefenseConfig:
    """防御構成の定義.

    Input Guard × Target LLM × Output Guard の組み合わせを表す。
    frozen=True によりハッシュ可能で、dict のキーとして使用可能。

    Attributes:
        input_guard: Input Guard のモデルID
        target_llm: Target LLM のモデルID
        output_guard: Output Guard のモデルID
    """

    input_guard: InputGuardName
    target_llm: TargetLLMName
    output_guard: OutputGuardName

    @property
    def name(self) -> str:
        """構成名を返す（パイプ区切り形式）."""
        ig_short = FULL_TO_SHORT_IG.get(self.input_guard, self.input_guard)
        llm_short = FULL_TO_SHORT_LLM.get(self.target_llm, self.target_llm)
        og_short = FULL_TO_SHORT_OG.get(self.output_guard, self.output_guard)
        return f"{ig_short}|{llm_short}|{og_short}"

    @property
    def full_name(self) -> str:
        """フルモデルID形式の構成名を返す."""
        return f"{self.input_guard}|{self.target_llm}|{self.output_guard}"

    @property
    def dir_name(self) -> str:
        """ディレクトリ名として使える形式を返す（{IG}_{LLM}_{OG}）."""
        return f"{self.input_guard}__{self.target_llm}__{self.output_guard}"

    def __str__(self) -> str:
        """文字列表現."""
        return self.name

    @classmethod
    def from_name(cls, name: str) -> "DefenseConfig":
        """構成名から DefenseConfig を作成する.

        Args:
            name: パイプ区切りの構成名

        Returns:
            DefenseConfig インスタンス
        """
        parts = name.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid config name: {name}")

        ig, llm, og = parts
        # フル名かショート名かを判定
        ig_full = SHORT_TO_FULL_IG.get(ig, ig)  # type: ignore
        llm_full = SHORT_TO_FULL_LLM.get(llm, llm)  # type: ignore
        og_full = SHORT_TO_FULL_OG.get(og, og)  # type: ignore

        return cls(ig_full, llm_full, og_full)  # type: ignore


@dataclass
class StageResult:
    """各ステージの評価結果.

    3段階防御システムの各ステージの通過/ブロック状態を保持する。

    Attributes:
        ig_passed: Input Guard を通過したか
        llm_passed: Target LLM から応答を得たか（拒否されなかったか）
        og_passed: Output Guard を通過したか
        response: LLM の応答テキスト（通過時のみ）
    """

    ig_passed: bool
    llm_passed: bool
    og_passed: bool
    response: str | None = None

    @property
    def success(self) -> bool:
        """全ステージを通過したか."""
        return self.ig_passed and self.llm_passed and self.og_passed

    @property
    def failed_stage(self) -> Stage:
        """失敗したステージを返す.

        全て通過した場合は PASSED_GUARDS を返す。
        最初に失敗したステージを返す。
        """
        if not self.ig_passed:
            return Stage.INPUT_GUARD
        if not self.llm_passed:
            return Stage.TARGET_LLM
        if not self.og_passed:
            return Stage.OUTPUT_GUARD
        return Stage.PASSED_GUARDS


@dataclass
class KBGuidanceInfo:
    """KB ガイダンス情報（ログ用）.

    Attributes:
        best_technique: KB が推奨した技法
        technique_effectiveness: その技法の有効性（0.0-1.0）
        stage_pass_rate: 対象ステージの通過率
        bottleneck_stage: 構成のボトルネックステージ
        n_examples: 取得した Few-shot 例の数
        config_found_in_kb: 構成が KB に存在したか
    """

    best_technique: str
    technique_effectiveness: float
    stage_pass_rate: float
    bottleneck_stage: str
    n_examples: int
    config_found_in_kb: bool


@dataclass
class AttackResult:
    """攻撃試行の結果.

    Attributes:
        attack_text: 攻撃プロンプトのテキスト
        iteration: 試行回数（0始まり）
        stage_result: 各ステージの結果
        is_jailbreak: Jailbreak 成功と判定されたか
        judgment_category: 詳細な判定カテゴリ（オプション）
        technique: 使用した攻撃技法（オプション）
        example_prompts: 参照した Few-shot 例（オプション）
        kb_guidance: KB から取得したガイダンス情報（オプション）
    """

    attack_text: str
    iteration: int
    stage_result: StageResult
    is_jailbreak: bool
    judgment_category: JudgmentCategory | None = None
    technique: str | None = None
    example_prompts: list[str] | None = None
    kb_guidance: KBGuidanceInfo | None = None
