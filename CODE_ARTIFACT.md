# CascadeEval Code Artifact

This repository includes two reproducibility surfaces:

- `src/c3a/` and `experiments/c3a/`: C3A experiment implementation and entry
  points copied from the private experiment worktree for anonymous review.
- `paper/analysis/` and `paper/figures/scripts/`: analysis and figure code
  used to regenerate paper measurements from staged derived data.

The code artifact intentionally excludes raw harmful goals, model outputs,
adaptive attack traces, and private result directories. Those files belong in
the gated data artifact, not in the public code repository.

## Local Checks

```bash
python3 -m compileall src/c3a experiments/c3a paper/analysis paper/figures/scripts
python3 paper/figures/scripts/fig2_asr_distribution.py
python3 paper/figures/scripts/fig3_bottleneck_funnel.py
python3 paper/figures/scripts/fig5_technique_heatmap.py
python3 paper/figures/scripts/fig6_og_llm_heatmap.py
python3 paper/figures/scripts/fig7_c3a_stage_distribution.py
python3 paper/figures/scripts/fig8_c3a_guidance_distribution.py
```

Optional C3A online-code smoke test:

```bash
uv sync --frozen

uv run python experiments/c3a/part2_evaluate.py \
  --dry-run --configs config_16 --agents direct,c3a --n-goals 1 -y
uv run python experiments/c3a/part2_evaluate.py \
  --mock --configs config_16 --agents direct,c3a --n-goals 1 -y
```

The public `--mock` smoke test uses
`experiments/c3a/fixtures/mock_knowledge_base.json` if the real Part 1
knowledge base is absent. To smoke-test against the actual paper KB, unpack the
gated data artifact and copy `data/knowledge_base.json` to
`experiments/c3a/results/knowledge_base.json` before running Part 2.
The gated artifact also includes `data/c3a_jbb_goals.json`; copy it to
`experiments/c3a/goals.json` for C3A re-runs with the same 100 JailbreakBench
JBB-Behaviors goals used by the paper.

`compute_audit_ci.py`, `compute_c3a_ci_cost.py`, and
`artifacts/cascadeeval_hf_dataset_20260506/scripts/prepare_hf_dataset.sh`
require gated raw stage logs or online-run traces. They are included for
reviewers with gated data access; the public code repository ships the derived
CSV outputs used by the paper.

`compute_c3a_ci_cost.py` additionally requires gated raw online-run
`aggregated_results.json` files at the paths recorded in
`paper/analysis/c3a_ci_cost_20260506/c3a_asr_ci.csv`. When those raw traces
are absent, the staged derived CSVs are the reproducibility surface for the
C3A table and appendix diagnostics.

For online C3A re-runs, provide the gated `knowledge_base.json`, generate or
provide the benchmark goals locally, and set the required provider credentials
in the execution environment. The gated artifact provides the paper-run goals
as `data/c3a_jbb_goals.json`; these are JailbreakBench JBB-Behaviors goals and
contain harmful goal text. Do not commit credentials, raw goals, or raw
generated traces to the public repository.

Full non-mock C3A re-runs are not API-only. In addition to attacker/target LLM
provider keys, the current implementation loads Hugging Face/local guard models
and the HarmBench classifier, so reviewers need `HF_TOKEN`, accepted model
licenses, and a local environment capable of running those models. The public
`--mock` path is the no-credential code-execution check.

Raw human-evaluation sheets, assignment logs, and prompt/response samples are
excluded from the public code repository. The reviewer artifact includes only
the anonymized aggregate calibration summary unless a separately anonymized
gated release is prepared.
