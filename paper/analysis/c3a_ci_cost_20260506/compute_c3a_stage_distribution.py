#!/usr/bin/env python3
"""Build a unified C3A stage-distribution table for Config A/B/C.

The earlier `bc_full_100_stage_distribution.csv` was intentionally scoped to
the B/C full-100 run. This script adds Config A hold-out rows from the separate
KB-ablation run and writes a single paper-facing table without prompt text.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/analysis/c3a_ci_cost_20260506/c3a_stage_distribution.csv"


@dataclass(frozen=True)
class RunSpec:
    config_label: str
    bottleneck: str
    input_guard: str
    target_llm: str
    output_guard: str
    config_key: str
    result_json: Path
    method_map: dict[str, str]


SPECS = [
    RunSpec(
        config_label="Config A",
        bottleneck="IG",
        input_guard="Qwen3Guard",
        target_llm="GPT-4o-mini",
        output_guard="WildGuard",
        config_key="Qwen_Qwen3Guard-Gen-0.6B__gpt-4o-mini__allenai_wildguard",
        result_json=ROOT
        / "paper/data/gpu_results_20260505/raw/results/c3a/kb_holdout_config16_full/aggregated_results.json",
        method_map={
            "direct": "direct",
            "c3a-nokb": "c3a-nokb",
            "c3a": "c3a-full-kb",
        },
    ),
    RunSpec(
        config_label="Config A",
        bottleneck="IG",
        input_guard="Qwen3Guard",
        target_llm="GPT-4o-mini",
        output_guard="WildGuard",
        config_key="Qwen_Qwen3Guard-Gen-0.6B__gpt-4o-mini__allenai_wildguard",
        result_json=ROOT
        / "paper/data/gpu_results_20260505/raw/results/c3a/kb_holdout_config16_reduced/aggregated_results.json",
        method_map={"c3a": "c3a-config-holdout-kb"},
    ),
    RunSpec(
        config_label="Config B",
        bottleneck="LLM",
        input_guard="ShieldGemma",
        target_llm="Phi-4",
        output_guard="LlamaGuard3",
        config_key="google_shieldgemma-2b__microsoft_phi-4__meta-llama_Llama-Guard-3-8B",
        result_json=ROOT
        / "paper/data/gpu_results_20260505/raw/results/c3a/part2_bc_full_20260503/aggregated_results.json",
        method_map={
            "direct": "direct",
            "pair": "pair",
            "c3a-nokb": "c3a-nokb",
            "c3a": "c3a-full-kb",
        },
    ),
    RunSpec(
        config_label="Config C",
        bottleneck="OG",
        input_guard="Qwen3Guard",
        target_llm="Ministral-8B",
        output_guard="WildGuard",
        config_key="Qwen_Qwen3Guard-Gen-0.6B__mistralai_Ministral-8B-Instruct-2410__allenai_wildguard",
        result_json=ROOT
        / "paper/data/gpu_results_20260505/raw/results/c3a/part2_bc_full_20260503/aggregated_results.json",
        method_map={
            "direct": "direct",
            "pair": "pair",
            "c3a-nokb": "c3a-nokb",
            "c3a": "c3a-full-kb",
        },
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_for_attempt(attempt: dict) -> str:
    failed_stage = attempt.get("failed_stage")
    if failed_stage in {"IG", "LLM", "OG"}:
        return failed_stage
    if attempt.get("ig_passed") and attempt.get("llm_passed") and attempt.get("og_passed"):
        return "PASSED_GUARDS"
    return "UNKNOWN"


def rows_for_spec(spec: RunSpec) -> list[dict[str, object]]:
    data = json.loads(spec.result_json.read_text())
    config = data["configs"][spec.config_key]
    source_rel = spec.result_json.relative_to(ROOT).as_posix()
    source_sha = sha256(spec.result_json)
    rows = []

    for raw_method, paper_method in spec.method_map.items():
        method_data = config[raw_method]
        counts: Counter[str] = Counter()
        total = 0
        for goal in method_data["goals"].values():
            for attempt in goal.get("history", []):
                counts[stage_for_attempt(attempt)] += 1
                total += 1

        for stage in ["IG", "LLM", "OG", "PASSED_GUARDS", "UNKNOWN"]:
            count = counts.get(stage, 0)
            if count == 0 and stage == "UNKNOWN":
                continue
            rows.append(
                {
                    "config_label": spec.config_label,
                    "bottleneck": spec.bottleneck,
                    "input_guard": spec.input_guard,
                    "target_llm": spec.target_llm,
                    "output_guard": spec.output_guard,
                    "config_key": spec.config_key,
                    "method": paper_method,
                    "stage": stage,
                    "count": count,
                    "share": f"{(count / total if total else 0):.4f}",
                    "total_iterations_with_stage": total,
                    "local_source": source_rel,
                    "sha256": source_sha,
                }
            )

    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    for spec in SPECS:
        rows.extend(rows_for_spec(spec))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
