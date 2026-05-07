# CascadeEval Artifact Staging

This directory is a local Hugging Face / reviewer-accessible artifact staging
area for the CascadeEval submission.

Status: **release-preparation staging, not uploaded by this script**. Upload
requires a separate anonymous or non-identifying reviewer artifact account.

## What Is Included

- Dataset card draft for HF Datasets.
- Datasheet for Datasets.
- Croissant metadata draft.
- Reproducibility instructions.
- Ethics / misuse warning.
- License files for public/code/gated tiers.
- An anonymized aggregate human-evaluation summary.
- Gated C3A benchmark goals (`c3a_jbb_goals.json`): the 100 JailbreakBench
  JBB-Behaviors goals used for the online evaluation runs.
- Gated raw C3A online-run `aggregated_results.json` files needed to recompute
  C3A ASR, delta, cost, stage, and guidance tables from raw traces.
- Source manifest. Checksums are generated when the data archives are
  materialized locally.
- A copy script that materializes data files from the local ignored result
  bundle and creates separate public-safe and gated archives.

## What Is Not Committed Here

The raw prompt / response data contains harmful jailbreak prompts and generated
model outputs. Those files are intentionally kept out of Git and should be
uploaded only to a gated or reviewer-accessible artifact location.

Run this from the repository root to materialize the upload directory and build
the two upload archives:

```bash
bash artifacts/cascadeeval_hf_dataset_20260506/scripts/prepare_hf_dataset.sh
```

Outputs are written to `artifacts/cascadeeval_hf_dataset_20260506/dist/`:

- `cascadeeval_public_2026.zip`
- `cascadeeval_gated_2026.zip`
- `submission_sha256.txt`

## Source Bundle

Local source bundle:

`paper/data/gpu_results_20260505/`

This directory is ignored by Git and contains the raw copied results from the
private source environment.
