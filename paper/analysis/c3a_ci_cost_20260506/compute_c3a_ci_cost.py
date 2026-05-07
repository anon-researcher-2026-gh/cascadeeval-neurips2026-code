#!/usr/bin/env python3
"""Compute C3A ASR confidence intervals and cost proxies.

Inputs are gated local raw result artifacts under
paper/data/gpu_results_20260505/raw/. The script writes small derived CSVs used
by the paper notes and C3A section.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RAW = ROOT / "paper/data/gpu_results_20260505/raw/results/c3a"

N_BOOT = 10_000
BOOT_SEED = 20260506


@dataclass(frozen=True)
class RunSpec:
    config_label: str
    bottleneck: str
    pipeline: str
    target_llm: str
    method_label: str
    method_key: str
    source_json: Path
    config_key: str
    kb_setting: str


BC_JSON = RAW / "part2_bc_full_20260503/aggregated_results.json"
A_FULL_JSON = RAW / "kb_holdout_config16_full/aggregated_results.json"
A_REDUCED_JSON = RAW / "kb_holdout_config16_reduced/aggregated_results.json"

B_KEY = "google_shieldgemma-2b__microsoft_phi-4__meta-llama_Llama-Guard-3-8B"
C_KEY = "Qwen_Qwen3Guard-Gen-0.6B__mistralai_Ministral-8B-Instruct-2410__allenai_wildguard"
A_KEY = "Qwen_Qwen3Guard-Gen-0.6B__gpt-4o-mini__allenai_wildguard"


RUNS: list[RunSpec] = [
    RunSpec("Config A", "IG", "Qwen3Guard -> GPT-4o-mini -> WildGuard", "gpt-4o-mini", "Direct", "direct", A_FULL_JSON, A_KEY, "none"),
    RunSpec("Config A", "IG", "Qwen3Guard -> GPT-4o-mini -> WildGuard", "gpt-4o-mini", "C3A w/o KB", "c3a-nokb", A_FULL_JSON, A_KEY, "none"),
    RunSpec("Config A", "IG", "Qwen3Guard -> GPT-4o-mini -> WildGuard", "gpt-4o-mini", "C3A config-holdout KB", "c3a", A_REDUCED_JSON, A_KEY, "config_holdout"),
    RunSpec("Config A", "IG", "Qwen3Guard -> GPT-4o-mini -> WildGuard", "gpt-4o-mini", "C3A full KB", "c3a", A_FULL_JSON, A_KEY, "full"),
    RunSpec("Config B", "LLM", "ShieldGemma -> Phi-4 -> LlamaGuard3", "microsoft_phi-4", "Direct", "direct", BC_JSON, B_KEY, "none"),
    RunSpec("Config B", "LLM", "ShieldGemma -> Phi-4 -> LlamaGuard3", "microsoft_phi-4", "PAIR", "pair", BC_JSON, B_KEY, "none"),
    RunSpec("Config B", "LLM", "ShieldGemma -> Phi-4 -> LlamaGuard3", "microsoft_phi-4", "C3A w/o KB", "c3a-nokb", BC_JSON, B_KEY, "none"),
    RunSpec("Config B", "LLM", "ShieldGemma -> Phi-4 -> LlamaGuard3", "microsoft_phi-4", "C3A full KB", "c3a", BC_JSON, B_KEY, "full"),
    RunSpec("Config C", "OG", "Qwen3Guard -> Ministral-8B -> WildGuard", "mistralai_Ministral-8B-Instruct-2410", "Direct", "direct", BC_JSON, C_KEY, "none"),
    RunSpec("Config C", "OG", "Qwen3Guard -> Ministral-8B -> WildGuard", "mistralai_Ministral-8B-Instruct-2410", "PAIR", "pair", BC_JSON, C_KEY, "none"),
    RunSpec("Config C", "OG", "Qwen3Guard -> Ministral-8B -> WildGuard", "mistralai_Ministral-8B-Instruct-2410", "C3A w/o KB", "c3a-nokb", BC_JSON, C_KEY, "none"),
    RunSpec("Config C", "OG", "Qwen3Guard -> Ministral-8B -> WildGuard", "mistralai_Ministral-8B-Instruct-2410", "C3A full KB", "c3a", BC_JSON, C_KEY, "full"),
]


TARGET_RATE_PER_CALL = {
    "gpt-4o-mini": 0.0003,
    "gpt-4o": 0.005,
    "gemini-2.0-flash": 0.0001,
}
ATTACKER_RATE_PER_CALL = 0.0003  # GPT-4o-mini proxy used by the run estimator.
ATTACKER_METHODS = {"pair", "c3a-nokb", "c3a"}


def load_run(spec: RunSpec) -> dict:
    data = json.loads(spec.source_json.read_text())
    return data["configs"][spec.config_key][spec.method_key]


def goal_rows(run: dict) -> list[dict]:
    rows = list(run["goals"].values())
    return sorted(rows, key=lambda r: int(r["goal_id"]))


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    stats = []
    for _ in range(N_BOOT):
        stats.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(stats, 0.025), percentile(stats, 0.975)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_delta_ci(left: dict[int, int], right: dict[int, int], rng: random.Random) -> tuple[float, float, float, int]:
    ids = sorted(set(left) & set(right))
    deltas = [left[i] - right[i] for i in ids]
    estimate = mean(deltas)
    lo, hi = bootstrap_mean_ci(deltas, rng)
    return estimate, lo, hi, len(ids)


def uses_attacker_api(spec: RunSpec) -> bool:
    return spec.method_key in ATTACKER_METHODS


def target_api_rate(spec: RunSpec) -> float:
    return TARGET_RATE_PER_CALL.get(spec.target_llm, 0.0)


def main() -> None:
    rng = random.Random(BOOT_SEED)
    asr_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    success_by_run: dict[tuple[str, str, str], dict[int, int]] = {}

    for spec in RUNS:
        run = load_run(spec)
        rows = goal_rows(run)
        successes = [1 if r["success"] else 0 for r in rows]
        n = len(successes)
        n_successes = sum(successes)
        asr = n_successes / n
        boot_lo, boot_hi = bootstrap_mean_ci(successes, rng)
        wilson_lo, wilson_hi = wilson_ci(n_successes, n)

        total_iterations = sum(int(r.get("n_iterations", 0)) for r in rows)
        mean_iterations = total_iterations / n
        elapsed_values = [float(r.get("elapsed_seconds", 0.0) or 0.0) for r in rows]
        total_wallclock_s = sum(elapsed_values)
        mean_wallclock_s = total_wallclock_s / n

        attacker_calls = total_iterations if uses_attacker_api(spec) else 0
        target_api_calls = total_iterations if target_api_rate(spec) else 0
        estimated_cost = (
            attacker_calls * ATTACKER_RATE_PER_CALL
            + target_api_calls * target_api_rate(spec)
        )
        estimated_cost_per_behavior = estimated_cost / n
        estimated_cost_per_success = "" if n_successes == 0 else estimated_cost / n_successes

        source_rel = spec.source_json.relative_to(ROOT).as_posix()
        asr_rows.append({
            "config": spec.config_label,
            "bottleneck": spec.bottleneck,
            "pipeline": spec.pipeline,
            "method": spec.method_label,
            "n": n,
            "successes": n_successes,
            "asr": f"{asr:.4f}",
            "bootstrap95_lo": f"{boot_lo:.4f}",
            "bootstrap95_hi": f"{boot_hi:.4f}",
            "wilson95_lo": f"{wilson_lo:.4f}",
            "wilson95_hi": f"{wilson_hi:.4f}",
            "source_json": source_rel,
        })
        cost_rows.append({
            "config": spec.config_label,
            "bottleneck": spec.bottleneck,
            "method": spec.method_label,
            "n": n,
            "successes": n_successes,
            "mean_iterations": f"{mean_iterations:.2f}",
            "total_iterations": total_iterations,
            "attacker_api_calls_proxy": attacker_calls,
            "target_api_calls_proxy": target_api_calls,
            "estimated_api_cost_usd_proxy": f"{estimated_cost:.4f}",
            "estimated_api_cost_usd_per_behavior_proxy": f"{estimated_cost_per_behavior:.4f}",
            "estimated_api_cost_usd_per_success_proxy": (
                "" if estimated_cost_per_success == "" else f"{estimated_cost_per_success:.4f}"
            ),
            "mean_wallclock_seconds": f"{mean_wallclock_s:.2f}",
            "total_wallclock_minutes": f"{total_wallclock_s / 60:.2f}",
            "cost_assumption": "attacker GPT-4o-mini $0.0003/proxy-call; API target if applicable; local guards/judge $0; token-level usage not logged",
        })
        success_by_run[(spec.config_label, spec.method_label, spec.kb_setting)] = {
            int(r["goal_id"]): 1 if r["success"] else 0 for r in rows
        }

    delta_specs = [
        ("Config A", "C3A full KB", "full", "C3A w/o KB", "none", "full KB - no KB"),
        ("Config A", "C3A config-holdout KB", "config_holdout", "C3A w/o KB", "none", "config-holdout KB - no KB"),
        ("Config A", "C3A full KB", "full", "Direct", "none", "full KB - direct"),
        ("Config B", "C3A full KB", "full", "C3A w/o KB", "none", "full KB - no KB"),
        ("Config B", "C3A full KB", "full", "PAIR", "none", "C3A full - PAIR"),
        ("Config B", "PAIR", "none", "Direct", "none", "PAIR - direct"),
        ("Config C", "C3A full KB", "full", "C3A w/o KB", "none", "full KB - no KB"),
        ("Config C", "C3A full KB", "full", "PAIR", "none", "C3A full - PAIR"),
        ("Config C", "PAIR", "none", "Direct", "none", "PAIR - direct"),
    ]
    delta_rows: list[dict[str, object]] = []
    for config, left_method, left_kb, right_method, right_kb, comparison in delta_specs:
        left = success_by_run[(config, left_method, left_kb)]
        right = success_by_run[(config, right_method, right_kb)]
        estimate, lo, hi, n_pairs = paired_delta_ci(left, right, rng)
        delta_rows.append({
            "config": config,
            "comparison": comparison,
            "left": left_method,
            "right": right_method,
            "n_pairs": n_pairs,
            "delta_asr": f"{estimate:.4f}",
            "paired_bootstrap95_lo": f"{lo:.4f}",
            "paired_bootstrap95_hi": f"{hi:.4f}",
        })

    write_csv(OUT_DIR / "c3a_asr_ci.csv", asr_rows)
    write_csv(OUT_DIR / "c3a_delta_ci.csv", delta_rows)
    write_csv(OUT_DIR / "c3a_cost_table.csv", cost_rows)
    write_tex_table(OUT_DIR / "table_c3a_asr_ci.tex", asr_rows)
    write_tex_table(OUT_DIR / "table_c3a_cost.tex", cost_rows, cost=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    return f"{float(value) * 100:.1f}"


def write_tex_table(path: Path, rows: list[dict[str, object]], cost: bool = False) -> None:
    with path.open("w") as f:
        f.write("% Auto-generated by compute_c3a_ci_cost.py. Do not edit by hand.\n")
        if not cost:
            f.write("\\begin{tabular}{@{}llrrr@{}}\n")
            f.write("\\toprule\n")
            f.write("Config & Method & ASR & Bootstrap 95\\% CI & Wilson 95\\% CI \\\\\n")
            f.write("\\midrule\n")
            for row in rows:
                ci = f"[{pct(row['bootstrap95_lo'])}, {pct(row['bootstrap95_hi'])}]"
                wilson = f"[{pct(row['wilson95_lo'])}, {pct(row['wilson95_hi'])}]"
                f.write(
                    f"{row['config']} & {row['method']} & {pct(row['asr'])}\\% & "
                    f"{ci}\\% & {wilson}\\% \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n")
        else:
            f.write("\\begin{tabular}{@{}llrrrr@{}}\n")
            f.write("\\toprule\n")
            f.write("Config & Method & Mean iter. & API proxy calls & Cost/behavior & Wall-clock/behavior \\\\\n")
            f.write("\\midrule\n")
            for row in rows:
                calls = int(row["attacker_api_calls_proxy"]) + int(row["target_api_calls_proxy"])
                calls_per_behavior = calls / int(row["n"])
                f.write(
                    f"{row['config']} & {row['method']} & {row['mean_iterations']} & "
                    f"{calls_per_behavior:.2f} & \\${float(row['estimated_api_cost_usd_per_behavior_proxy']):.4f} & "
                    f"{row['mean_wallclock_seconds']}s \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
