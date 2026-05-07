"""JailbreakBench データセットをダウンロードして goals.json を生成する.

Usage:
    uv run python experiments/c3a/scripts/download_benchmark.py
"""

import json
from pathlib import Path

from datasets import load_dataset

# カテゴリマッピング（JailbreakBench → 短縮形）
CATEGORY_MAP = {
    "Harassment/Discrimination": "HD",
    "Malware/Hacking": "MH",
    "Physical harm": "PH",
    "Economic harm": "EH",
    "Fraud/Deception": "FD",
    "Disinformation": "DI",
    "Sexual/Adult content": "SA",
    "Privacy": "PD",  # Personal Data / Privacy
    "Expert advice": "EA",
    "Government decision-making": "GD",
}


def main() -> None:
    """メイン関数."""
    print("JailbreakBench データセットをダウンロード中...")

    # データセットを取得
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    df_harmful = ds["harmful"].to_pandas()

    print(f"Harmful behaviors: {len(df_harmful)}")
    print(f"Categories: {df_harmful['Category'].nunique()}")

    # goals.json 形式に変換
    goals = []
    for i, row in df_harmful.iterrows():
        goal = {
            "id": int(row["Index"]) + 1,  # 1-indexed
            "goal": row["Goal"],
            "target": row["Target"],
            "category": CATEGORY_MAP.get(row["Category"], row["Category"]),
            "category_full": row["Category"],
            "source": row["Source"],
            "behavior_id": row["Behavior"],
        }
        goals.append(goal)

    # 出力
    output_data = {
        "name": "JailbreakBench JBB-Behaviors",
        "description": "Standard benchmark for jailbreaking evaluation (NeurIPS 2024)",
        "source": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        "n_goals": len(goals),
        "categories": list(CATEGORY_MAP.values()),
        "goals": goals,
    }

    output_path = Path("experiments/c3a/goals.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n出力: {output_path}")
    print(f"総目標数: {len(goals)}")
    print("\nカテゴリ別内訳:")
    for cat, count in df_harmful["Category"].value_counts().items():
        short = CATEGORY_MAP.get(cat, cat)
        print(f"  {short} ({cat}): {count}")


if __name__ == "__main__":
    main()
