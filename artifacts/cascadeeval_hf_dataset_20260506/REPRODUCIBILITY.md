# Reproducibility

## Local Inputs

The paper-ready local result bundle is:

`paper/data/gpu_results_20260505/`

Important files:

- `raw/results/c3a/paper_ready/verified/part1/raw/en_v2_results_7010.csv`
- `raw/results/c3a/paper_ready/verified/part1/raw/knowledge_base.json`
- `raw/results/c3a/part2_bc_full_20260503/aggregated_results.json`
- `raw/results/c3a/kb_holdout_config16_full/aggregated_results.json`
- `raw/results/c3a/kb_holdout_config16_reduced/aggregated_results.json`
- `c3a_stage_distribution.csv`
- `experiment_provenance.csv`
- `PROVENANCE.md`

The ignored raw-result bundle is not committed to Git. It is materialized into
the reviewer artifact by `scripts/prepare_hf_dataset.sh`.

The gated artifact also includes `data/c3a_jbb_goals.json`, a deterministic
manifest of the 100 JailbreakBench JBB-Behaviors goals used for the C3A online
evaluation. This is the paper-run goal file to restore as
`experiments/c3a/goals.json`; it is not placed in the public archive because it
contains harmful goal text.

## Recompute CI / Cost Tables

From the repository root:

```bash
python3 paper/analysis/c3a_ci_cost_20260506/compute_audit_ci.py
```

This recomputes the configuration-audit CI tables from the staged
stage-level records.

The C3A ASR, delta, and cost tables can be recomputed only when the gated raw
online-run JSON files are available at the paths recorded in
`paper/analysis/c3a_ci_cost_20260506/c3a_asr_ci.csv`:

```bash
python3 paper/analysis/c3a_ci_cost_20260506/compute_c3a_ci_cost.py
python3 paper/analysis/c3a_ci_cost_20260506/compute_c3a_stage_distribution.py
```

In the gated artifact archive, these files are staged under
`data/raw/results/c3a/`, and the C3A goal manifest is staged as
`data/c3a_jbb_goals.json`. To restore the paths expected by the code artifact,
copy the raw trace directory into the code repository's ignored raw-result
bundle:

```bash
mkdir -p paper/data/gpu_results_20260505/raw/results/c3a
cp -R data/raw/results/c3a/* paper/data/gpu_results_20260505/raw/results/c3a/
```

To re-run C3A with the same benchmark goals used by the paper, restore the
gated goal manifest to the path expected by the code artifact:

```bash
cp data/c3a_jbb_goals.json experiments/c3a/goals.json
```

If those raw `aggregated_results.json` files are absent, use the packaged
derived CSVs below as the reproducibility surface for the paper's C3A table and
appendix diagnostics.

Outputs:

- `paper/analysis/c3a_ci_cost_20260506/audit_summary_ci.csv`
- `paper/analysis/c3a_ci_cost_20260506/audit_configuration_asr.csv`
- `paper/analysis/c3a_ci_cost_20260506/c3a_asr_ci.csv`
- `paper/analysis/c3a_ci_cost_20260506/c3a_delta_ci.csv`
- `paper/analysis/c3a_ci_cost_20260506/c3a_cost_table.csv`
- `paper/analysis/c3a_ci_cost_20260506/c3a_guidance_distribution.csv`

The staged artifact copy script materializes the raw stage table and KB
together with these derived CI, stage-distribution, and cost tables under
`artifacts/cascadeeval_hf_dataset_20260506/data/`.

## Recompute Paper Claims

Use:

- `paper/data/numbers.yaml`
- `paper/analysis/c3a_ci_cost_20260506/*.csv`

Do not use unresolved legacy claims unless their matched provenance is restored.

## Recompute Paper Tables and Figures

From the repository root:

```bash
python3 paper/figures/scripts/fig2_asr_distribution.py
python3 paper/figures/scripts/fig3_bottleneck_funnel.py
python3 paper/figures/scripts/fig5_technique_heatmap.py
python3 paper/figures/scripts/fig6_og_llm_heatmap.py
python3 paper/figures/scripts/fig7_c3a_stage_distribution.py
python3 paper/figures/scripts/fig8_c3a_guidance_distribution.py
make -C paper
```

For a containerized paper build:

```bash
make -C paper docker-build
```

The PDF build does not re-run API experiments. It uses the committed analysis
tables and the ignored local result bundle described above.

## Re-run C3A Experiments

The anonymous code artifact contains the C3A implementation under `src/c3a/`
and the experiment entry points under `experiments/c3a/`. The paper tables can
be recomputed locally from the staged logs, but full online C3A re-runs require
external model access.

Full non-mock C3A re-runs are not API-only. They require the gated paper KB and
JailbreakBench goal manifest, attacker/target LLM provider credentials, a
Hugging Face token with accepted licenses for the selected guard/judge models,
and local compute capable of loading those Hugging Face/local models. The
public `--mock` command is the no-credential code-execution check.

```bash
uv run python experiments/c3a/part2_evaluate.py --mock
uv run python experiments/c3a/part2_evaluate.py --dry-run
```

Use the non-mock command only after setting provider credentials and accepting
the relevant model/dataset licenses. For paper-goal re-runs, copy
`data/c3a_jbb_goals.json` from the gated artifact to
`experiments/c3a/goals.json`. Raw goals and generated attack traces must remain
in the gated artifact or local ignored storage.

## Random Seed

The C3A online configuration uses seed `42` and maximum 20 iterations per goal
unless overridden in `experiments/c3a/config.yaml`. The paper-ready provenance
bundle records source-file checksums and experiment-level provenance for the
staged results.

## Hardware

The CI/cost recomputation and figure generation are CPU-only. Full online C3A
evaluation can use API targets and local guard/model inference; local
open-weight guards require an environment compatible with the selected
Hugging Face models and sufficient accelerator memory for those models.

## API Costs

C3A and baseline online re-runs consume attacker-model and target-model API
calls. The paper's cost table is regenerated by
`paper/analysis/c3a_ci_cost_20260506/compute_c3a_ci_cost.py`; it should be
treated as a proxy estimator tied to the recorded run metadata, not a billing
guarantee.

## API Keys

Re-running adaptive attacks requires API credentials for the attacker LLM and,
for API target configurations, target LLM providers. The staged artifact does
not include API keys.

## Artifact Archives

Run:

```bash
bash artifacts/cascadeeval_hf_dataset_20260506/scripts/prepare_hf_dataset.sh
```

This refreshes the staged `data/` directory, verifies checksums, and writes:

- `artifacts/cascadeeval_hf_dataset_20260506/dist/cascadeeval_public_2026.zip`
- `artifacts/cascadeeval_hf_dataset_20260506/dist/cascadeeval_gated_2026.zip`
- `artifacts/cascadeeval_hf_dataset_20260506/dist/submission_sha256.txt`
