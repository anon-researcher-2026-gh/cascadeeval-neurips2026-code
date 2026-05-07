#!/usr/bin/env python3
"""Compute audit-level CIs from the 275-configuration raw bundle.

This script reconstructs end-to-end ASR for each complete configuration from
the copied `en_v2_results_7010.csv` and the 275-configuration KB profile.
It then reports cluster bootstrap CIs over configurations for aggregate audit
metrics, plus Wilson CIs for each per-configuration ASR.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median, quantiles


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PART1 = ROOT / "paper/data/gpu_results_20260505/raw/results/c3a/paper_ready/verified/part1/raw"
CSV_PATH = PART1 / "en_v2_results_7010.csv"
KB_PATH = PART1 / "knowledge_base.json"

N_BOOT = 10_000
BOOT_SEED = 20260506


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_config_metric(values: list[dict], metric_fn, rng: random.Random) -> tuple[float, float]:
    n = len(values)
    stats = []
    for _ in range(N_BOOT):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(metric_fn(sample))
    return percentile(stats, 0.025), percentile(stats, 0.975)


def main() -> None:
    kb = json.loads(KB_PATH.read_text())
    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    config_rows = []

    for config_name, profile in kb["config_profiles"].items():
        ig = profile["ig"]
        llm = profile["llm"]
        og = profile["og"]
        ig_col = f"input_guard_{ig}_is_safe"
        refusal_col = f"llm_{llm}_is_refusal"
        harmful_col = f"multijudge_on_{llm}_is_harmful"
        og_col = f"output_guard_{og}_on_{llm}_is_safe"
        required = [ig_col, refusal_col, harmful_col, og_col]
        missing = [col for col in required if col not in rows[0]]
        if missing:
            raise KeyError(f"{config_name}: missing columns {missing}")

        after_ig = 0
        after_llm = 0
        after_og = 0
        successes = 0
        for row in rows:
            passes_ig = truth(row[ig_col])
            passes_llm = passes_ig and not truth(row[refusal_col])
            passes_og = passes_llm and truth(row[og_col])
            success = passes_og and truth(row[harmful_col])

            after_ig += int(passes_ig)
            after_llm += int(passes_llm)
            after_og += int(passes_og)
            if success:
                successes += 1

        n = len(rows)
        asr = successes / n
        wilson_lo, wilson_hi = wilson_ci(successes, n)
        config_rows.append({
            "config_name": config_name,
            "input_guard": ig,
            "target_llm": llm,
            "output_guard": og,
            "bottleneck": profile["bottleneck"],
            "n": n,
            "after_ig": after_ig,
            "after_llm": after_llm,
            "after_og": after_og,
            "successes": successes,
            "survival_after_ig": after_ig / n,
            "survival_after_llm": after_llm / n,
            "survival_after_og": after_og / n,
            "asr": asr,
            "harmful_given_emitted": successes / after_og if after_og else math.nan,
            "asr_wilson95_lo": wilson_lo,
            "asr_wilson95_hi": wilson_hi,
            "ig_block_rate": 1 - profile["ig_pass_rate"],
            "llm_block_rate": 1 - profile["llm_pass_rate"],
            "og_block_rate": 1 - profile["og_pass_rate"],
        })

    write_csv(OUT_DIR / "audit_configuration_asr.csv", [
        {
            **row,
            "survival_after_ig": f"{row['survival_after_ig']:.6f}",
            "survival_after_llm": f"{row['survival_after_llm']:.6f}",
            "survival_after_og": f"{row['survival_after_og']:.6f}",
            "asr": f"{row['asr']:.6f}",
            "harmful_given_emitted": f"{row['harmful_given_emitted']:.6f}",
            "asr_wilson95_lo": f"{row['asr_wilson95_lo']:.6f}",
            "asr_wilson95_hi": f"{row['asr_wilson95_hi']:.6f}",
            "ig_block_rate": f"{row['ig_block_rate']:.6f}",
            "llm_block_rate": f"{row['llm_block_rate']:.6f}",
            "og_block_rate": f"{row['og_block_rate']:.6f}",
        }
        for row in config_rows
    ])

    rng = random.Random(BOOT_SEED)
    asrs = [row["asr"] for row in config_rows]
    q1, _, q3 = quantiles(asrs, n=4, method="inclusive")
    summary_specs = [
        ("audit_n_configs", len(config_rows), "", ""),
        ("audit_n_prompts", len(rows), "", ""),
        ("asr_mean_over_configs", mean(asrs), *bootstrap_config_metric(config_rows, lambda s: mean(r["asr"] for r in s), rng)),
        ("asr_median_over_configs", median(asrs), *bootstrap_config_metric(config_rows, lambda s: median(r["asr"] for r in s), rng)),
        ("asr_q25_over_configs", q1, *bootstrap_config_metric(config_rows, lambda s: quantiles([r["asr"] for r in s], n=4, method="inclusive")[0], rng)),
        ("asr_q75_over_configs", q3, *bootstrap_config_metric(config_rows, lambda s: quantiles([r["asr"] for r in s], n=4, method="inclusive")[2], rng)),
        ("asr_min_observed", min(asrs), "", ""),
        ("asr_max_observed", max(asrs), "", ""),
    ]

    for stage in ["IG", "LLM", "OG"]:
        key = stage.lower() + "_block_rate"
        summary_specs.append((
            f"marginal_{stage.lower()}_block_rate_mean_over_configs",
            mean(row[key] for row in config_rows),
            *bootstrap_config_metric(config_rows, lambda s, k=key: mean(r[k] for r in s), rng),
        ))

    for stage in ["IG", "LLM", "OG"]:
        summary_specs.append((
            f"bottleneck_share_{stage}",
            sum(1 for row in config_rows if row["bottleneck"] == stage) / len(config_rows),
            *bootstrap_config_metric(config_rows, lambda s, st=stage: sum(1 for r in s if r["bottleneck"] == st) / len(s), rng),
        ))

    for stage in ["ig", "llm", "og"]:
        key = f"survival_after_{stage}"
        summary_specs.append((
            f"{key}_mean_over_configs",
            mean(row[key] for row in config_rows),
            *bootstrap_config_metric(config_rows, lambda s, k=key: mean(r[k] for r in s), rng),
        ))

    summary_specs.append((
        "harmful_given_emitted_mean_over_configs",
        mean(row["harmful_given_emitted"] for row in config_rows),
        *bootstrap_config_metric(
            config_rows,
            lambda s: mean(r["harmful_given_emitted"] for r in s),
            rng,
        ),
    ))

    summary_rows = []
    for metric, value, lo, hi in summary_specs:
        summary_rows.append({
            "metric": metric,
            "estimate": value if isinstance(value, int) else f"{float(value):.6f}",
            "bootstrap95_lo": "" if lo == "" else f"{float(lo):.6f}",
            "bootstrap95_hi": "" if hi == "" else f"{float(hi):.6f}",
            "n_boot": "" if lo == "" else N_BOOT,
            "resample_unit": "" if lo == "" else "configuration",
            "source": "en_v2_results_7010.csv + knowledge_base.json",
        })
    write_csv(OUT_DIR / "audit_summary_ci.csv", summary_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
