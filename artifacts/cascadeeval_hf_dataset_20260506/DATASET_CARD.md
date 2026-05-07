---
pretty_name: CascadeEval
license: other
language:
  - en
  - ja
tags:
  - llm-safety
  - llm-evaluation
  - guardrails
  - jailbreak-evaluation
  - croissant
size_categories:
  - 1K<n<10K
---

# CascadeEval

## Dataset Summary

CascadeEval is a staged evaluation artifact for multi-stage LLM defense
pipelines of the form:

`Input Guard -> Target LLM -> Output Guard`

The current submission snapshot contains:

- 7,010 jailbreak prompts.
- 275 complete defense configurations.
- Stage-level records for Input Guard, Target LLM refusal, Output Guard, and harmfulness judge outcomes.
- A stage-level Knowledge Base (KB) of configuration-, stage-, and technique-conditioned profiles.
- C3A adaptive stress-test outputs for Config A/B/C.
- A gated manifest of the 100 JailbreakBench JBB-Behaviors goals used for the
  C3A online evaluation runs.
- Human-evaluation calibration summary for harmfulness measurement. Raw
  annotation sheets are excluded from the public archive and should be shared
  only after anonymization and gated review.

## Intended Use

This artifact is intended for research on:

- Evaluation of multi-stage LLM safety pipelines.
- Stage-level attribution of blocking decisions.
- Reproducibility checks for CascadeEval tables.
- Analysis of bottleneck-dependent adaptive stress testing.
- Auditing whether a reported ASR conclusion changes under cascade-aware
  adaptive stress testing.

## Out-of-Scope / Misuse Warning

The prompt and response data include harmful jailbreak prompts and potentially
harmful model outputs. The dataset must not be used to deploy harmful systems,
optimize real-world abuse, or create operational jailbreak tooling.

Out-of-scope uses include:

- Training or fine-tuning models to generate harmful instructions.
- Building prompt libraries, jailbreak products, or attack automation for
  unauthorized systems.
- Publishing raw harmful prompts, model responses, adaptive traces, or KB
  retrieval exemplars outside the gated release process.
- Re-identifying annotators or attempting to recover operational metadata from
  human-evaluation files.
- Treating CascadeEval ASR as a deployment certification for a guardrail or
  model.

## Access

Recommended release mode: an anonymous or otherwise non-identifying gated
Hugging Face Dataset, with a reviewer-accessible fallback archive if the gated
workflow is unavailable during review.

Do not publish raw prompt / response files as an unrestricted public dataset
without an explicit responsible-access decision by the authors.

The staging script creates two physical archives:

- `cascadeeval_public_2026.zip`: documentation, metadata, aggregate statistics,
  paper-ready derived tables, and analysis inputs that do not contain raw
  harmful prompt/response text. It also includes an anonymized human-evaluation
  aggregate summary.
- `cascadeeval_gated_2026.zip`: raw harmful prompts, model responses,
  stage-level records, KB retrieval exemplars, the 100 JailbreakBench
  JBB-Behaviors goals used for C3A, and raw C3A online-run
  `aggregated_results.json` traces. This archive is only for NeurIPS reviewers
  and bona fide safety researchers who accept the gated research-use terms.

OpenReview should not expose a private download link in a public comment. The
intended review workflow is either a gated dataset access request labelled
"NeurIPS reviewer" or a time-limited reviewer token supplied through the
submission system's non-public artifact field.

## License

License split:

- Dataset card, datasheet, ethics statement, manifests, Croissant metadata,
  aggregate statistics, and paper-ready derived tables: CC BY 4.0.
- Reproducibility, experiment, and figure-generation scripts authored for
  CascadeEval: MIT.
- Raw harmful prompts, model responses, adaptive traces, and KB retrieval
  exemplars: Custom Gated Research-Use Terms. Access is limited to reviewers
  and bona fide safety researchers who agree not to redistribute raw content or
  use it for harmful generation.

The prompt corpus is derived from atmaCup #21 materials and subsequent
processing. Public pages identify atmaCup #21 as an LLM attack-defense
competition. The authors have obtained permission to include atmaCup #21-derived
prompt data in the CascadeEval submission artifact under gated or
reviewer-accessible research access. Raw prompt text therefore remains gated and
is not treated as unrestricted public data.

If atmaCup #21 appears inside a prompt string, that occurrence is part of the
original task framing or competition provenance. It is not an author
attribution signal for the anonymous submission.

## Provenance

Primary local source bundle:

`paper/data/gpu_results_20260505/`

Code/config provenance:

`paper/data/gpu_results_20260505/PROVENANCE.md`

Remote source:

`<retained in private provenance records for post-review verification>`

Remote git HEAD at collection time:

`<retained in private provenance records for post-review verification>`

The remote worktree was dirty at collection time. Relevant code/config snapshots
are included in the anonymous code artifact where they can be released safely;
raw data provenance details that could identify authors are retained outside the
public metadata and can be restored after de-anonymization.
