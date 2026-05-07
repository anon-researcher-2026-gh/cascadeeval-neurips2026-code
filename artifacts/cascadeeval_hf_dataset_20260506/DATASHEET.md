# Datasheet for CascadeEval

This datasheet follows the seven-part structure of Gebru et al. (2021). It is
written for the NeurIPS Evaluations & Datasets submission artifact and should be
read together with `DATASET_CARD.md`, `ETHICS.md`, `REPRODUCIBILITY.md`,
`CHECKSUMS.sha256`, and `metadata/croissant.json`.

## Quick Index for Reviewers

- Safeguards / misuse: Sections 5 and 6
- New assets / provenance: Sections 2, 3, and 6
- Human subjects / annotator protection: Sections 3, 4, and 7
- Licensing / access control: Section 6 and `LICENSE_PROPOSAL.md`
- Reproducibility: Section 4 and `REPRODUCIBILITY.md`

## 1. Motivation

**Q1. For what purpose was the dataset created?**
CascadeEval was created to evaluate multi-stage LLM defense pipelines of the
form Input Guard -> Target LLM -> Output Guard with stage-level attribution,
rather than scoring individual components in isolation.

**Q2. Who created the dataset?**
The dataset was created by the anonymous submission authors. Identifying
affiliations are withheld during double-blind review.

**Q3. Who funded or supported the dataset creation?**
The public artifact does not identify funding sources during anonymous review.
Funding and institutional acknowledgments are handled in the camera-ready or
de-anonymized artifact if disclosure is appropriate.

**Q4. What gap does it address?**
It addresses the gap between end-to-end ASR reporting and stage-level evidence:
the same ASR can arise from different Input Guard, Target LLM, or Output Guard
failure modes.

**Q5. Who is the intended audience?**
The intended audience is LLM safety researchers, benchmark reviewers, guardrail
developers, and practitioners running authorized safety evaluations.

## 2. Composition

**Q1. What does each instance contain?**
Stage-level evaluation records contain a prompt identifier, prompt text,
taxonomy labels, defense configuration identifiers, Input Guard decision,
Target LLM response/refusal signal, Output Guard decision, harmfulness-judge
score/label, blocking stage, and success indicator.

**Q2. How many instances are included?**
The verified audit snapshot contains 7,010 prompts evaluated over 275 complete
defense configurations. Derived C3A result tables use 100-behavior runs for the
reported bottleneck-representative configurations. Those 100 C3A behaviors are
JailbreakBench JBB-Behaviors goals and are distributed only in the gated
`data/c3a_jbb_goals.json` manifest.

**Q3. What labels are included?**
The prompt corpus includes intent labels and technique labels. Stage records
include stage pass/block/refusal decisions, calibrated harmfulness labels, and
configuration-level bottleneck labels.

**Q4. Does the dataset include sensitive or harmful content?**
Yes. Raw prompts include jailbreak and harmful-instruction content, and model
responses may contain harmful outputs. These fields are gated and are not
released as unrestricted public data.

**Q5. Does the dataset contain personal information?**
No personally identifying information is intentionally collected as a dataset
field. Because free-form prompts and responses can contain incidental personal
or credential-like strings, the raw release is gated and subject to takedown
review.

**Q6. Are there missing data?**
The main paper uses the 275 configurations with complete stage-level records.
Earlier partial configuration passes are not used for headline numbers.

## 3. Collection Process

**Q1. What are the prompt sources?**
The corpus starts from atmaCup #21 prompt-battle materials and subsequent
cleaning, English normalization, deduplication, and taxonomy labeling. The C3A
online evaluation goals are the 100 JailbreakBench JBB-Behaviors goals restored
as the gated `data/c3a_jbb_goals.json` manifest.

**Q2. What external evidence supports the source description?**
Public atmaCup pages describe atmaCup #21 as an LLM attack-defense competition
and a public third-party writeup describes the competition as an
attack-vs-defense prompt battle with stage-based evaluation.

**Q3. What is the collection period?**
The artifact snapshot is dated 2026-05-06. Earlier source collection and remote
result generation are documented in the provenance files referenced by
`REPRODUCIBILITY.md`.

**Q4. Was consent obtained from prompt authors?**
Competition participation was governed by the competition platform and rules.
The authors have obtained permission to include atmaCup #21-derived prompt data
in the CascadeEval submission artifact under gated or reviewer-accessible
research access. This permission does not make raw harmful prompt text an
unrestricted public dataset; raw prompt text remains gated under the release
terms in Section 6.

**Q5. How were human annotators involved?**
Human annotators labeled a stratified calibration sample for harmfulness
measurement. Annotators saw prompt-response pairs without configuration
metadata, blocking-stage metadata, or judge scores.

**Q6. Were annotators compensated?**
The repository contains the operational requirement to specify compensation,
but payment records are not included in the public artifact because they are
private operational records. The public artifact should state that annotators
were engaged under agreed compensation terms while keeping payment details out
of the release bundle.

**Q7. Was ethical review performed?**
The artifact is treated as an LLM safety evaluation resource with harmful
content. The release uses an IRB-equivalent internal safeguard review: gated
raw access, misuse restrictions, anonymized human-evaluation outputs, and a
reporting channel for takedown or harmful-content concerns.

## 4. Preprocessing, Cleaning, and Labeling

**Q1. What preprocessing was applied?**
The source prompts were cleaned, normalized to English for the main evaluation
snapshot, deduplicated, and assigned intent and technique taxonomy labels.

**Q2. How were stage outcomes produced?**
Each prompt was evaluated against fixed Input Guard, Target LLM, and Output
Guard configurations. Logs record the per-stage decision fields needed to
compute ASR, bottlenecks, survival funnels, and C3A feedback.

**Q3. How was harmfulness labeled?**
A HarmBench-style automated judge produced raw scores. The paper uses a
calibrated threshold selected against a 240-sample human-evaluation set.

**Q4. Were any records excluded?**
Headline results exclude configurations without complete stage-level records.
They also exclude unresolved legacy claims whose provenance is not restored in
`paper/data/numbers.yaml`.

**Q5. Can the preprocessing be reproduced?**
Derived CI and cost tables can be regenerated using the scripts in
`paper/analysis/c3a_ci_cost_20260506/`. Full raw regeneration requires the
ignored local source bundle described in `REPRODUCIBILITY.md`.

## 5. Uses

**Q1. What are intended uses?**
Intended uses are authorized LLM safety evaluation, stage-level guardrail
diagnosis, reproduction of CascadeEval paper tables, and defensive research on
multi-stage pipeline behavior.

**Q2. What uses are out of scope?**
Out-of-scope uses include harmful model training, jailbreak product
development, unauthorized attacks against deployed systems, and unrestricted
redistribution of raw harmful prompts or successful adaptive traces.

**Q3. What misuse risks exist?**
Raw prompts, model responses, adaptive traces, and KB exemplars could help an
actor improve jailbreak attempts. Those assets are gated and covered by
research-use terms.

**Q4. Does the dataset certify deployment safety?**
No. CascadeEval provides benchmark measurements under a defined snapshot and
query budget. It is not a certification that a guardrail or pipeline is safe in
deployment.

**Q5. How should researchers cite limitations?**
Researchers should cite the stage-level scope, harmfulness-judge calibration
error, gated raw access, and bottleneck-dependent C3A interpretation.

## 6. Distribution

**Q1. How will the artifact be distributed?**
The staged release is designed for a gated Hugging Face Dataset or
reviewer-accessible archive. The local directory intentionally excludes raw
data files from Git.

**Q2. Which assets are public or reviewer-accessible?**
Dataset card, datasheet, ethics statement, manifests, Croissant metadata,
aggregate statistics, derived CI tables, and figure-generation scripts can be
public or reviewer-accessible.

**Q3. Which assets are gated?**
Raw harmful prompts, raw model responses, successful adaptive traces, and KB
retrieval exemplars are gated. The JailbreakBench JBB-Behaviors goal manifest
used for C3A online evaluation is also gated because it contains harmful goal
text.

**Q4. What licenses apply?**
Documentation, metadata, aggregate statistics, and derived tables are proposed
for CC BY 4.0. CascadeEval-authored scripts are proposed for MIT. Raw harmful
content is proposed for Custom Gated Research-Use Terms.

**Q5. What is the atmaCup #21 redistribution status?**
Permission has been obtained to include atmaCup #21-derived prompt data in the
gated or reviewer-accessible CascadeEval artifact. The artifact should credit
atmaCup #21 as the prompt-source competition and should not redistribute raw
prompt text as unrestricted public data unless a broader public license is
separately granted.

**Q6. How are versions identified?**
The Croissant metadata version is `1.0.0`, with snapshot date `2026-05-06`.
File-level checksums are listed in `CHECKSUMS.sha256` and
`metadata/MANIFEST.csv`.

## 7. Maintenance

**Q1. Who maintains the dataset?**
The anonymous authors maintain the dataset during review. The de-anonymized
project owner should be listed in the public artifact after review.

**Q2. How can problems be reported?**
During anonymous review, use the OpenReview anonymous discussion/contact
mechanism. After de-anonymization, the public artifact should list a maintained
project mailbox or issue tracker.

**Q3. What is the takedown process?**
Reports of credentials, personal data, or harmful content exceeding the release
scope should be triaged by the maintainers; affected gated files should be
removed or redacted and checksums regenerated.

**Q4. Will the dataset be updated?**
Updates should be versioned. Adding or updating a guard, target model, prompt
source, or harmfulness judge requires regenerating the affected stage-level
logs, KB entries, and derived tables.

**Q5. What long-term support is promised?**
N/A as a service-level guarantee: this research artifact does not promise
production support. The maintenance commitment is limited to versioned
corrections, takedown handling, and reproducibility documentation.
