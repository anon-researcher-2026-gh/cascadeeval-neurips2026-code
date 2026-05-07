# Ethics and Responsible Access

CascadeEval includes harmful jailbreak prompts and model responses. The artifact
is designed for safety evaluation research, not for enabling abuse.

## Responsible Release

- Raw harmful prompts and responses should be distributed through gated or
  reviewer-accessible channels.
- Public metadata should avoid exposing full harmful prompt text unless the
  release policy explicitly allows it.
- API keys, personal credentials, and host-specific secrets must not be included.
- Successful adaptive traces and KB retrieval exemplars are treated as
  potentially more operational than aggregate metrics and should follow the
  same gated access path as raw prompts.

## Human Evaluation

Human-evaluation artifacts should include only the minimum information needed to
reproduce aggregate results. Before release, confirm that annotator identifiers,
free-text comments, and any operational metadata are anonymized.

The public artifact should use stable anonymous IDs such as `annotator_01` and
should not include real names, private contact details, payment records, or
free-text comments that contain identifying information. Compensation and
consent records are operational records and are not part of the public artifact.

## Misuse Restrictions

Do not use this artifact to:

- Construct real-world harassment, malware, fraud, or abuse workflows.
- Optimize attacks against deployed systems outside an authorized evaluation.
- Publish ungated harmful prompt corpora without author approval.

## Reporting Problems

During anonymous review, takedown, credential-leak, or harmful-content concerns
should be reported through the OpenReview anonymous discussion/contact
mechanism attached to the submission. After de-anonymization, the maintained
public artifact must list the project mailbox or issue tracker used for the
same reports.
