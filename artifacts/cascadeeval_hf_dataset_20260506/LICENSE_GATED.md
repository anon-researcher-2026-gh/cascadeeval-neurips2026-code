# CascadeEval Custom Gated Research-Use Terms

These terms apply to raw harmful prompts, model responses, adaptive attack
traces, stage-level records containing raw prompt/response text, and Knowledge
Base retrieval exemplars in the CascadeEval reviewer or gated release.

## Permitted Use

You may use the gated files only for:

- Peer review of the CascadeEval submission.
- Authorized safety evaluation research.
- Defensive analysis of multi-stage LLM safety systems.
- Reproducing the paper's reported measurements under the same responsible-use
  constraints.

## Prohibited Use

You may not:

- Redistribute raw gated files or substantial excerpts outside the approved
  access channel.
- Use the raw prompts, responses, traces, or KB exemplars to train, fine-tune,
  optimize, benchmark, or deploy systems for harmful generation.
- Build operational jailbreak tooling, prompt libraries, abuse automation, or
  evasion services.
- Attack real systems without explicit authorization.
- Attempt to re-identify annotators, authors, or private infrastructure from
  metadata or free-form fields.

## Handling Requirements

- Store gated files in access-controlled storage.
- Do not publish raw harmful examples in papers, blog posts, public repos, or
  issue trackers.
- Share only aggregate statistics, sanitized snippets, or derived non-harmful
  summaries unless the maintainers approve otherwise.
- Delete local copies when reviewer access or research need ends.

## Reporting and Takedown

Report suspected sensitive-data exposure, misuse risk, or takedown requests
through the anonymous review contact during submission. After de-anonymization,
use the maintained project contact listed in the public repository or dataset
card.

The maintainers may revoke access, rotate files, or remove affected records if
misuse or unintended sensitive-data exposure is discovered.
