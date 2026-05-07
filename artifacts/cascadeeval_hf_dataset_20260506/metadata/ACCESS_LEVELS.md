# CascadeEval Access Levels

The manifest uses these release-access labels:

- `public_or_reviewer`: Derived tables, aggregate statistics, documentation,
  metadata, and scripts that do not contain raw harmful prompt/response text.
  These files are placed in `cascadeeval_public_2026.zip`.
- `reviewer_accessible_only`: Raw adaptive traces or reviewer-only diagnostics
  that are safe for review but should not be public. If raw C3A traces contain
  prompt text, model responses, or successful adaptive refinements, classify
  them as `gated` instead.
- `gated`: Raw harmful prompts, model responses, stage-level logs containing
  raw text, and KB retrieval exemplars. These files are placed only in
  `cascadeeval_gated_2026.zip` and are governed by `LICENSE_GATED.md`.

When in doubt, classify a file as `gated` if it contains raw prompt text,
model responses to harmful prompts, successful adaptive refinements, or
retrieval exemplars that could materially improve jailbreak attempts.
