# C3A: Cascade-Aware Adaptive Agent

Multi-stage defense systems に対する cascade-aware adaptive evaluation
agent. This code artifact intentionally excludes raw harmful goals and generated
attack traces; those files belong in the gated data artifact.

## 1. 概要

### 1.1 研究目的

LLM の3段階防御システム（Input Guard → Target LLM → Output Guard）に対し、失敗ステージに応じて攻撃戦略を適応的に変更する手法 C3A を提案・評価する。

### 1.2 システムアーキテクチャ

```
攻撃プロンプト
      │
      ▼
┌─────────────┐
│ Input Guard │ ← Stage 1: 入力フィルタリング
└─────────────┘
      │ pass
      ▼
┌─────────────┐
│ Target LLM  │ ← Stage 2: 応答生成（内部安全訓練）
└─────────────┘
      │ pass
      ▼
┌─────────────┐
│ Output Guard│ ← Stage 3: 出力フィルタリング
└─────────────┘
      │ pass
      ▼
    応答
```

### 1.3 C3A の特徴

| 特徴 | 説明 |
|------|------|
| ステージ認識 | 失敗ステージを特定し、そこに特化した戦略を選択 |
| Knowledge Base | Part 1 で学習した技法効果・成功パターンを活用 |
| Few-shot 例示 | 類似構成での成功プロンプトを参考に生成 |
| 技法ローテーション | 失敗履歴に基づき SEM → PRAG → ORTH を自動切替 |

---

## 2. セットアップ

### 2.1 依存関係

```bash
uv sync
```

### 2.2 環境変数 (.env)

```bash
OPENAI_API_KEY=<provider key>       # Attacker LLM
HF_TOKEN=<huggingface token>        # Gated model access when needed
GOOGLE_API_KEY=<provider key>       # Optional API target
LOCAL_MODEL_BASE_URL=<endpoint>     # Optional local/vLLM endpoint
```

Full non-mock C3A re-runs are not LLM-API-only. The current implementation uses
Hugging Face/local guard models and the HarmBench classifier in addition to
attacker/target LLM providers. Use `--mock` for the reviewer-safe execution
check that does not require credentials or model downloads.

### 2.3 ベンチマークデータ

```bash
# JailbreakBench をローカルにダウンロード
uv run python experiments/c3a/scripts/download_benchmark.py
```

`experiments/c3a/goals.json` is generated locally and is ignored by Git because
it contains JailbreakBench JBB-Behaviors harmful-behavior goal text.

### 2.4 Knowledge Base

`c3a` uses the Part 1 knowledge base. Public `--mock` runs fall back to
`experiments/c3a/fixtures/mock_knowledge_base.json` when the real KB is absent.
For reviewer smoke tests against the actual paper KB, restore it from the gated
data artifact before running Part 2:

```bash
GATED_DATA_DIR=/path/to/unpacked/cascadeeval_gated_2026
mkdir -p experiments/c3a/results
cp "$GATED_DATA_DIR/data/knowledge_base.json" experiments/c3a/results/knowledge_base.json
cp "$GATED_DATA_DIR/data/c3a_jbb_goals.json" experiments/c3a/goals.json
```

`data/c3a_jbb_goals.json` is the gated manifest of the 100 JailbreakBench
JBB-Behaviors goals used for the paper's C3A online evaluation. It is kept out
of the public code repository for the same harmful-content reason as
`experiments/c3a/goals.json`.

---

## 3. 実験フロー

```
Part 1: オフライン分析          Part 2: オンライン評価
┌──────────────────────┐       ┌──────────────────────┐
│ 既存ラベルデータ分析   │       │ 実 API 攻撃評価       │
│ (7,010攻撃 × 275構成) │  ──►  │ (C3A vs ベースライン) │
│                      │       │                      │
│ 出力: Knowledge Base │       │ 出力: ASR, クエリ数   │
└──────────────────────┘       └──────────────────────┘
```

---

## 4. Part 1: オフライン分析

### 4.1 目的

既存ラベルデータを分析し、Knowledge Base を構築する。

### 4.2 コマンド

```bash
# 標準実行
uv run python experiments/c3a/part1_analysis.py

# オプション
uv run python experiments/c3a/part1_analysis.py \
  --data-path data/processed/en_v2_results.csv \
  --output-dir experiments/c3a/results \
  --no-visualize  # 図生成スキップ
```

### 4.3 出力

| ファイル | 内容 |
|---------|------|
| `knowledge_base.json` | 技法効果、構成プロファイル、成功例インデックス |
| `analysis_result.json` | 統計サマリー |
| `figures/` | 可視化図（ASR ヒートマップ等） |

### 4.4 Knowledge Base 構造

```json
{
  "n_attacks": 7010,
  "n_configs": 275,
  "effectiveness": {
    "IG": {"LEX": 0.65, "SEM": 0.71, ...},
    "LLM": {"ORTH": 0.71, ...},
    "OG": {"PRAG": 0.90, ...}
  },
  "config_profiles": {...},
  "stage_pass_indices": {...},
  "prompts": [...],
  "embeddings": [...]  // Few-shot検索用
}
```

---

## 5. Part 2: オンライン評価

### 5.1 目的

C3A と3つのベースラインを実 API で評価し、ASR とクエリ効率を比較する。

### 5.2 コマンド

```bash
# コスト見積もりのみ
uv run python experiments/c3a/part2_evaluate.py --dry-run

# モックモード（コード検証用、API 不使用）
# uses a synthetic KB fixture if experiments/c3a/results/knowledge_base.json is absent
uv run python experiments/c3a/part2_evaluate.py --mock

# 最小テスト（API 疎通確認、1 goal のみ）
CUDA_VISIBLE_DEVICES=2,3 uv run python experiments/c3a/part2_evaluate.py --n-goals 1 --configs config_1 --agents c3a -y

# 本番実行
uv run python experiments/c3a/part2_evaluate.py

# 詳細ログ
uv run python experiments/c3a/part2_evaluate.py --verbose
```

### 5.3 オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--config` | 設定ファイル | `experiments/c3a/config.yaml` |
| `--dry-run` | コスト見積もりのみ | false |
| `--mock` | モッククライアント使用 | false |
| `--configs` | 評価構成（カンマ区切り） | 全構成 |
| `--agents` | 評価エージェント（カンマ区切り） | 全エージェント |
| `--n-goals` | 使用目標数 | config.yaml の値 |
| `--verbose` | 詳細ログ | false |
| `-y, --yes` | 確認プロンプトをスキップ | false |

### 5.4 自動再開機能

- 完了済み実験は自動的にスキップされる
- 中断後の再実行で未完了 goal から自動再開
- 再実行する場合は `results/c3a/part2/experiments/index.json` を削除

### 5.5 出力ディレクトリ構造

```
results/c3a/part2/
├── experiments/
│   ├── index.json                           # 完了済み実験インデックス
│   └── {IG}__{LLM}__{OG}/                   # 防御構成ごと
│       └── {agent}/                         # エージェントごと
│           ├── results.json                 # 詳細結果（攻撃履歴付き）
│           ├── summary.json                 # goal サマリー（軽量版）
│           ├── attack_log.csv               # 攻撃ログ（CSV）
│           └── attack_log.json              # 攻撃ログ（JSON）
├── aggregated_results.json                  # 全結果の集約
└── judge_cache.json                         # Judge 判定キャッシュ
```

### 5.6 summary.json 形式

goal ごとに1エントリ、必要最小限の情報のみ：

```json
[
  {
    "goal_id": 1,
    "goal": "<raw harmful goal omitted from public code artifact>",
    "category": "HD",
    "success": true,
    "n_iterations": 8,
    "final_prompt": "<generated attack text stored only in gated artifacts>"
  },
  ...
]
```

---

## 5. NeurIPS 2026 paper evaluation setting

### Representative configurations

The main paper reports three bottleneck-representative configurations selected
from the verified 275-configuration audit:

| Label | Input Guard | Target LLM | Output Guard | Bottleneck |
|------|------------|------------|-------------|-----------|
| Config A | Qwen3Guard | GPT-4o-mini | WildGuard | IG |
| Config B | ShieldGemma | Phi-4 | LlamaGuard3 | LLM |
| Config C | Qwen3Guard | Ministral-8B | WildGuard | OG |

Raw online traces and harmful goals are not committed to the public code
repository; they belong in the gated reviewer data artifact.

### 評価エージェント（4種）

| エージェント | 役割 |
|-------------|------|
| `c3a` | 提案手法（KB + ステージ情報） |
| `c3a-nokb` | Ablation: KB なし（ステージ情報のみ） |
| `pair` | PAIR ベースライン（二値フィードバック） |
| `random` | 非適応ベースライン |

### 実験規模

- **ゴール数**: 100（JailbreakBench 全件）
- **最大試行数**: 20 回 / goal
- **総実験数**: selected configurations × selected agents × 100 goals

### 実行コマンド

```bash
uv run python experiments/c3a/part2_evaluate.py --dry-run
uv run python experiments/c3a/part2_evaluate.py --mock
uv run python experiments/c3a/part2_evaluate.py
```

---

## 6. 構成要素

### 6.1 防御コンポーネント

**Input Guards (5種)**
| 短縮名 | モデル ID |
|--------|----------|
| WildGuard | `allenai/wildguard` |
| LlamaGuard3 | `meta-llama/Llama-Guard-3-8B` |
| LlamaGuard4 | `meta-llama/Llama-Guard-4-12B` |
| Qwen3Guard | `Qwen/Qwen3Guard-Gen-0.6B` |
| ShieldGemma | `google/shieldgemma-2b` |

**Target LLMs (11種)**
| 短縮名 | 提供元 |
|--------|--------|
| gpt-4o | OpenAI API |
| gpt-4o-mini | OpenAI API |
| gemini-2.0-flash | Google API |
| Llama-3.1-8B | Local/HuggingFace |
| Qwen2.5-7B | Local/HuggingFace |
| Qwen2.5-14B | Local/HuggingFace |
| Qwen3-8B | Local/HuggingFace |
| Ministral-8B | Local/HuggingFace |
| gemma-3-12b | Local/HuggingFace |
| phi-4 | Local/HuggingFace |
| GPT-OSS-20B | Local/HuggingFace |

**Output Guards (5種)**: Input Guards と同一

**総構成数**: 5 × 11 × 5 = **275**

### 6.2 攻撃エージェント

| エージェント | フィードバック | KB使用 | 説明 |
|-------------|--------------|--------|------|
| `c3a` | ステージ情報 | Yes | 提案手法 |
| `c3a-nokb` | ステージ情報 | No | Ablation |
| `pair` | 二値（成功/失敗） | No | PAIR ベースライン |
| `random` | なし | No | 非適応ベースライン |

### 6.3 攻撃技法

| 技法 | 説明 | 最適ステージ |
|------|------|-------------|
| LEX | 字句的操作 | - |
| MSYN | 形態統語操作 | - |
| SEM | 意味的シフト | IG (56.6%) |
| PRAG | 語用論的フレーミング | OG |
| ORTH | 直交的変換（エンコード等） | LLM |

### 6.4 Jailbreak Judge

| 項目 | 値 |
|------|-----|
| モデル | HarmBench-Llama-2-13b-cls |
| 方式 | Binary Classification |
| 実行 | ローカル GPU（コストなし） |
| 参照 | AutoDAN-Turbo 準拠 |

---

## 7. ベンチマーク

### 7.1 JailbreakBench (NeurIPS 2024)

| 項目 | 値 |
|------|-----|
| 総目標数 | 100 |
| カテゴリ数 | 10 |
| ソース | AdvBench (18%), HarmBench (27%), Original (55%) |

### 7.2 カテゴリ

| コード | カテゴリ | 件数 |
|--------|---------|------|
| HD | Harassment/Discrimination | 10 |
| MH | Malware/Hacking | 10 |
| PH | Physical harm | 10 |
| EH | Economic harm | 10 |
| FD | Fraud/Deception | 10 |
| DI | Disinformation | 10 |
| SA | Sexual/Adult content | 10 |
| PD | Privacy | 10 |
| EA | Expert advice | 10 |
| GD | Government decision-making | 10 |

---

## 8. 設定ファイル

`experiments/c3a/config.yaml` の主要項目:

```yaml
experiment:
  seed: 42
  max_iterations: 20      # 1目標あたりの最大試行数

agents:                   # 評価するエージェント
  - c3a
  - pair
  - random
  - c3a-nokb

configs:                  # 評価する防御構成
  - name: "config_1"
    input_guard: "meta-llama_Llama-Guard-3-8B"
    target_llm: "gpt-4o-mini"
    output_guard: "meta-llama_Llama-Guard-3-8B"
    description: "LlamaGuard3 + API LLM"

goals:
  path: "experiments/c3a/goals.json"
  n_goals: 100            # null で全件
  categories: null        # null で全カテゴリ

paths:
  data: "data/processed/en_v2_results.csv"
  knowledge_base: "experiments/c3a/results/knowledge_base.json"
  output_dir: "results/c3a/part2"
```

---

## 9. コスト見積もり

### 9.1 API コスト構成

| 用途 | モデル | 課金 |
|------|--------|------|
| Target LLM | gpt-4o, gpt-4o-mini, gemini | あり |
| Target LLM | Llama, Qwen 等 | なし（ローカル） |
| Attacker LLM | gpt-4o-mini | あり |
| Jailbreak Judge | HarmBench-Llama-2-13b-cls | なし（ローカル） |
| Guards | 全て | なし（ローカル） |

### 9.2 見積もり例

| 設定 | 推定コスト |
|------|-----------|
| モックテスト | $0 |
| 最小テスト（1構成 × 5目標 × 1エージェント） | ~$0.10 |
| 中規模（1構成 × 100目標 × 4エージェント） | ~$7 |
| 代表構成セット（構成数・agent数に依存） | dry-runで確認 |

---

## 10. ファイル構成

```
experiments/c3a/
├── README.md                 # 本ドキュメント
├── config.yaml               # 実験設定
├── goals.json                # JailbreakBench 攻撃目標
├── part1_analysis.py         # Part 1 スクリプト
├── part2_evaluate.py         # Part 2 スクリプト
├── scripts/
│   └── download_benchmark.py # ベンチマークDL
└── results/
    └── knowledge_base.json   # Part 1 出力

results/c3a/part2/
├── experiments/
│   ├── index.json                           # 完了済み実験インデックス
│   └── {IG}__{LLM}__{OG}/
│       └── {agent}/
│           ├── results.json                 # 詳細結果
│           ├── summary.json                 # goal サマリー
│           ├── attack_log.csv               # 攻撃ログ
│           └── attack_log.json              # 攻撃ログ
├── aggregated_results.json                  # 集約結果
└── judge_cache.json                         # Judge キャッシュ

src/c3a/
├── agent/                    # C3A エージェント実装
│   ├── c3a.py
│   └── generator.py          # 技法ローテーション、Guard特化戦略
├── baselines/                # ベースライン実装
│   ├── pair.py
│   ├── random.py
│   └── no_knowledge.py
├── clients/                  # API クライアント
│   ├── llms.py
│   ├── huggingface.py
│   └── openai.py
├── data/                     # データ層
│   ├── loader.py
│   └── columns.py
├── evaluation/               # 評価層
│   ├── online.py
│   ├── judge.py              # HarmBench Judge
│   ├── experiment_manager.py # 実験管理（自動スキップ）
│   └── metrics.py
├── knowledge/                # Knowledge Base
│   └── kb_loader.py
└── types.py                  # 型定義
```

---

## 11. クイックスタート

```bash
# 1. 環境構築
uv sync
# Set provider credentials in the environment; do not commit .env files.

# 2. ベンチマーク取得
uv run python experiments/c3a/scripts/download_benchmark.py

# 3. Part 1: Knowledge Base 構築（既存データ使用）
uv run python experiments/c3a/part1_analysis.py

# 4. Part 2: モックテスト
uv run python experiments/c3a/part2_evaluate.py --mock --n-goals 5

# 5. Part 2: 本番実行
uv run python experiments/c3a/part2_evaluate.py --dry-run  # コスト確認
uv run python experiments/c3a/part2_evaluate.py            # 実行
```

---

## 12. 参考文献

- [JailbreakBench](https://jailbreakbench.github.io/) - NeurIPS 2024 Datasets and Benchmarks
- [PAIR](https://arxiv.org/abs/2310.08419) - Prompt Automatic Iterative Refinement
- [HarmBench](https://arxiv.org/abs/2402.04249) - A Standardized Evaluation Framework for Automated Red Teaming
- [AutoDAN-Turbo](https://arxiv.org/abs/2405.03684) - A Lifelong Agent for Strategy Self-Exploration
