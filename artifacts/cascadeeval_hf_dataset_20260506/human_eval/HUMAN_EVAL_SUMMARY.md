# Human-Evaluation Summary

This file is the anonymized aggregate human-evaluation material included in the
reviewer artifact. It deliberately excludes raw annotation sheets, annotator
names, prompt text, and model responses.

## Scope

- Source sample: 240 stratified records.
- Majority-defined records: 239 of 240.
- Annotation task: harmfulness calibration for the automatic harmfulness judge.
- Public/reviewer-safe release mode: aggregate metrics and protocol only.
- Raw sheets: excluded from the public archive; gated release requires
  additional anonymization and removal of raw harmful prompt/response text.

## Aggregate Calibration Metrics

The paper uses the calibrated threshold view, treating the judge as a
measurement instrument rather than an oracle.

| Metric | Value |
|---|---:|
| Raw agreement with human majority | 59.83% |
| Raw Cohen's kappa | 0.194 |
| Calibrated threshold tau* | 0.709 |
| Calibrated agreement with human majority | 87.9% |
| Calibrated Cohen's kappa | 0.498 |
| Fleiss' kappa, three-way labels | 0.270 |
| Fleiss' kappa, binary labels | 0.305 |
| False-positive rate for original automatic label | 44.29% |
| False-negative rate for original automatic label | 10.34% |

## Release Boundary

This summary is safe for the public or reviewer-accessible archive because it
contains only aggregate statistics. Raw annotation CSV files remain outside the
public artifact because they include raw harmful prompts/responses and
annotator-identifying metadata.
