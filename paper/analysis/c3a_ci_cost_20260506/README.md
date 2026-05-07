# C3A CI / Cost Analysis 2026-05-06

This directory contains derived, reproducible analysis outputs for the audit
and C3A results used by the paper.

## Inputs

- `compute_audit_ci.py` requires the gated raw stage-level table and KB under
  `paper/data/gpu_results_20260505/raw/...`.
- `compute_c3a_ci_cost.py` requires gated raw online-run
  `aggregated_results.json` files at the paths recorded in `c3a_asr_ci.csv`.
- Figure scripts use the derived CSVs committed in this directory where
  possible, so reviewers can rebuild paper figures without raw prompt text.

## Outputs

- `audit_summary_ci.csv`: cluster-bootstrap CIs over 275 configurations,
  including marginal block rates and true cascade-sequential survival rates.
- `audit_configuration_asr.csv`: reconstructed per-configuration ASR,
  cascade survival counts/rates, harmfulness among emitted attempts, and
  Wilson CI.
- `c3a_asr_ci.csv`: ASR, percentile bootstrap 95% CI, and Wilson 95% CI.
- `c3a_delta_ci.csv`: paired bootstrap 95% CI for matched method differences.
- `c3a_cost_table.csv`: observed iterations, wall-clock, and API-cost proxy.
- `technique_stage_pass_rates.csv`: derived stage-by-technique matrix for
  Figure 5 without raw prompt/response text.
- `table_c3a_asr_ci.tex`: LaTeX snippet generated from `c3a_asr_ci.csv`.
- `table_c3a_cost.tex`: LaTeX snippet generated from `c3a_cost_table.csv`.

## Cost Caveat

The run artifacts do not contain token-level OpenAI usage records. The cost
table therefore reports a proxy derived from the estimator in the archived
execution code: attacker GPT-4o-mini is counted at `$0.0003` per observed
iteration, API target LLMs are counted at the same proxy rate where applicable,
and local guards / local judge are counted as `$0`.

Use this as a lightweight routine-stress-test cost estimate, not as a billing
receipt.

## Reproduction

```bash
python3 paper/analysis/c3a_ci_cost_20260506/compute_audit_ci.py
python3 paper/analysis/c3a_ci_cost_20260506/compute_c3a_ci_cost.py
```

If the gated raw inputs are absent, use the committed derived CSVs as the
reviewer reproducibility surface for the paper's reported tables and figures.
