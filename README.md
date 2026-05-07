# CascadeEval

Anonymous code artifact for a NeurIPS 2026 Evaluations & Datasets submission
on configuration-wide, stage-aware evaluation of multi-stage LLM defense
pipelines.

This repository is the public reviewer-facing code surface. It intentionally
does not contain raw prompts, model responses, adaptive attack traces, human
annotation sheets, private run directories, or generated archive files. Those
materials are handled through the separate gated artifact package described in
`artifacts/cascadeeval_hf_dataset_20260506/`.

## Repository Layout

- `paper/`: figure PDFs, figure-generation scripts, derived analysis outputs,
  and numeric source-of-truth files. Paper TeX source is intentionally omitted
  because the submission PDF is provided separately.
- `paper/data/numbers.yaml`: paper-level numeric source of truth for reported
  scalar claims.
- `paper/analysis/c3a_ci_cost_20260506/`: derived analysis CSVs and scripts
  used for C3A confidence intervals, cost accounting, and appendix diagnostics.
- `src/c3a/`: C3A implementation.
- `experiments/c3a/`: representative experiment entry points and configuration.
- `artifacts/cascadeeval_hf_dataset_20260506/`: dataset card, datasheet,
  Croissant metadata, licensing notes, reproducibility instructions, and the
  packaging script for the gated/public artifact archives.

## Quick Checks

```bash
uv sync --frozen
python3 -m compileall -q src/c3a experiments/c3a paper/analysis/c3a_ci_cost_20260506 paper/figures/scripts
uv run python experiments/c3a/part2_evaluate.py --mock --configs config_16 --agents direct,c3a --n-goals 1 -y
python3 paper/figures/scripts/fig2_asr_distribution.py
python3 paper/figures/scripts/fig3_bottleneck_funnel.py
python3 paper/figures/scripts/fig5_technique_heatmap.py
python3 paper/figures/scripts/fig6_og_llm_heatmap.py
python3 paper/figures/scripts/fig7_c3a_stage_distribution.py
python3 paper/figures/scripts/fig8_c3a_guidance_distribution.py
```

## Data Access

The public repository ships only derived, reviewer-safe analysis artifacts.
Raw prompt/response records, stage-level logs, online C3A traces, and raw human
evaluation materials are excluded from Git. Reviewers who receive gated artifact
access can materialize the upload archives with:

```bash
bash artifacts/cascadeeval_hf_dataset_20260506/scripts/prepare_hf_dataset.sh
```

That packaging script expects the gated raw bundle to be restored locally under
the paths documented in the artifact reproducibility notes.

The public C3A mock check uses a synthetic knowledge-base fixture when the real
Part 1 knowledge base is absent. Full C3A re-runs require gated
`data/knowledge_base.json`, the gated `data/c3a_jbb_goals.json` JailbreakBench
JBB-Behaviors goal manifest, provider credentials, and local/gated model access.
They are not API-only: the current implementation loads Hugging Face/local guard
models and the HarmBench classifier. Use the public `--mock` command above for
the no-credential execution check.

## Anonymity

The public repository is prepared from an allowlist and should be published to a
fresh anonymous GitHub repository. Do not push the private working repository or
its Git history for review.
