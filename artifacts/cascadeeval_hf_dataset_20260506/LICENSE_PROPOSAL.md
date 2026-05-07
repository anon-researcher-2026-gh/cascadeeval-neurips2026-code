# License Proposal

## Recommendation

Use a two-tier release.

| Tier | Assets | Recommended terms | Rationale |
|---|---|---|---|
| Tier 1: public / reviewer-accessible | Dataset card, datasheet, ethics statement, Croissant metadata, manifests, aggregate statistics, CI tables, figure-generation scripts | CC BY 4.0 for documentation/data tables; MIT for CascadeEval-authored scripts | These assets support review, citation, and reproduction without exposing operational harmful content. |
| Tier 2: gated | Raw harmful prompts, model responses, successful adaptive traces, and KB retrieval exemplars | Custom Gated Research-Use Terms | These assets can enable jailbreak optimization if redistributed without controls. |

## Custom Gated Research-Use Terms

Access to Tier 2 should require the requester to agree that they will:

- Use the data only for authorized safety evaluation, auditing, or defensive
  research.
- Not redistribute raw prompts, model responses, successful adaptive traces, or
  KB exemplars.
- Not use the data to train or fine-tune models for harmful generation.
- Not use the data to attack deployed systems without authorization.
- Report credential leaks, personal data, or unexpectedly harmful records
  through the artifact reporting channel.

## atmaCup #21 Status

The corpus is derived from atmaCup #21 materials and subsequent processing.
Public pages reviewed for this staging copy identify the competition as an LLM
attack-defense prompt competition. The authors have obtained permission to
include atmaCup #21-derived prompt data in the
CascadeEval submission artifact under gated or reviewer-accessible research
access. This permission supports Tier 2 submission/reviewer distribution; it
does not convert the raw prompt text into unrestricted public data.

## Release Decision

Use the two-tier split for the submission artifact. If a broader public atmaCup
data license is later granted, only the access label for the affected raw prompt
files should be reconsidered; successful adaptive traces, model responses, and
KB retrieval exemplars should remain gated because their operational misuse risk
is independent of the source license.
