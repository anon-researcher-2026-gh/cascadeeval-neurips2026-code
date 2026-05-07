#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ARTIFACT_DIR="$ROOT/artifacts/cascadeeval_hf_dataset_20260506"
DEST="$ARTIFACT_DIR/data"
DIST="$ARTIFACT_DIR/dist"
WORK="$ARTIFACT_DIR/.package"
RAW_C3A_SRC="$ROOT/paper/data/gpu_results_20260505/raw/results/c3a"
RAW_C3A_DEST="$DEST/raw/results/c3a"

mkdir -p "$DEST" \
  "$RAW_C3A_DEST/part2_bc_full_20260503" \
  "$RAW_C3A_DEST/kb_holdout_config16_full" \
  "$RAW_C3A_DEST/kb_holdout_config16_reduced"
rm -f "$DEST/bc_full_100_stage_distribution.csv"

cp "$ROOT/paper/data/gpu_results_20260505/raw/results/c3a/paper_ready/verified/part1/raw/en_v2_results_7010.csv" "$DEST/en_v2_results_7010.csv"
cp "$ROOT/paper/data/gpu_results_20260505/raw/results/c3a/paper_ready/verified/part1/raw/knowledge_base.json" "$DEST/knowledge_base.json"
cp "$ROOT/paper/data/gpu_results_20260505/c3a_stage_distribution.csv" "$DEST/c3a_stage_distribution.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/audit_configuration_asr.csv" "$DEST/audit_configuration_asr.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/audit_summary_ci.csv" "$DEST/audit_summary_ci.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/c3a_asr_ci.csv" "$DEST/c3a_asr_ci.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/c3a_delta_ci.csv" "$DEST/c3a_delta_ci.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/c3a_cost_table.csv" "$DEST/c3a_cost_table.csv"
cp "$ROOT/paper/analysis/c3a_ci_cost_20260506/c3a_guidance_distribution.csv" "$DEST/c3a_guidance_distribution.csv"
cp "$RAW_C3A_SRC/part2_bc_full_20260503/aggregated_results.json" "$RAW_C3A_DEST/part2_bc_full_20260503/aggregated_results.json"
cp "$RAW_C3A_SRC/kb_holdout_config16_full/aggregated_results.json" "$RAW_C3A_DEST/kb_holdout_config16_full/aggregated_results.json"
cp "$RAW_C3A_SRC/kb_holdout_config16_reduced/aggregated_results.json" "$RAW_C3A_DEST/kb_holdout_config16_reduced/aggregated_results.json"

python3 - "$RAW_C3A_DEST/kb_holdout_config16_reduced/aggregated_results.json" "$DEST/c3a_jbb_goals.json" <<'PY'
import json
import sys

source_path, output_path = sys.argv[1:3]
with open(source_path, "r", encoding="utf-8") as f:
    source = json.load(f)

goals_by_id = {}

def walk(value):
    if isinstance(value, dict):
        if {"goal_id", "goal", "category"} <= value.keys():
            goal_id = int(value["goal_id"])
            record = {
                "id": goal_id,
                "goal": value["goal"],
                "category": value["category"],
            }
            existing = goals_by_id.get(goal_id)
            if existing is not None and existing != record:
                raise ValueError(f"Conflicting goal record for id={goal_id}")
            goals_by_id[goal_id] = record
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(source)
goals = [goals_by_id[k] for k in sorted(goals_by_id)]
payload = {
    "name": "JailbreakBench JBB-Behaviors goals used for C3A online evaluation",
    "source": "JailbreakBench/JBB-Behaviors",
    "derived_from": "data/raw/results/c3a/kb_holdout_config16_reduced/aggregated_results.json",
    "n_goals": len(goals),
    "goals": goals,
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

(
  cd "$ARTIFACT_DIR"
  shasum -a 256 \
    data/en_v2_results_7010.csv \
    data/knowledge_base.json \
    data/c3a_jbb_goals.json \
    data/c3a_stage_distribution.csv \
    data/audit_configuration_asr.csv \
    data/audit_summary_ci.csv \
    data/c3a_asr_ci.csv \
    data/c3a_delta_ci.csv \
    data/c3a_cost_table.csv \
    data/c3a_guidance_distribution.csv \
    data/raw/results/c3a/part2_bc_full_20260503/aggregated_results.json \
    data/raw/results/c3a/kb_holdout_config16_full/aggregated_results.json \
    data/raw/results/c3a/kb_holdout_config16_reduced/aggregated_results.json \
    human_eval/HUMAN_EVAL_SUMMARY.md > CHECKSUMS.sha256.new
  mv CHECKSUMS.sha256.new CHECKSUMS.sha256
)

rm -rf "$DIST" "$WORK"
mkdir -p "$DIST" "$WORK/public/data" "$WORK/public/metadata" "$WORK/public/scripts" "$WORK/public/human_eval" "$WORK/gated/data" "$WORK/gated/metadata" "$WORK/gated/human_eval"

cp "$ARTIFACT_DIR"/README.md "$WORK/public/"
cp "$ARTIFACT_DIR"/DATASET_CARD.md "$WORK/public/"
cp "$ARTIFACT_DIR"/DATASHEET.md "$WORK/public/"
cp "$ARTIFACT_DIR"/ETHICS.md "$WORK/public/"
cp "$ARTIFACT_DIR"/REPRODUCIBILITY.md "$WORK/public/"
cp "$ARTIFACT_DIR"/LICENSE "$WORK/public/"
cp "$ARTIFACT_DIR"/LICENSE_CODE "$WORK/public/"
cp "$ARTIFACT_DIR"/metadata/MANIFEST.csv "$WORK/public/metadata/"
cp "$ARTIFACT_DIR"/metadata/ACCESS_LEVELS.md "$WORK/public/metadata/"
cp "$ARTIFACT_DIR"/metadata/croissant.json "$WORK/public/metadata/"
cp "$ARTIFACT_DIR"/scripts/prepare_hf_dataset.sh "$WORK/public/scripts/"
cp "$ARTIFACT_DIR"/CHECKSUMS.sha256 "$WORK/public/"
cp "$ARTIFACT_DIR"/human_eval/HUMAN_EVAL_SUMMARY.md "$WORK/public/human_eval/"

cp "$DEST"/c3a_stage_distribution.csv "$WORK/public/data/"
cp "$DEST"/audit_configuration_asr.csv "$WORK/public/data/"
cp "$DEST"/audit_summary_ci.csv "$WORK/public/data/"
cp "$DEST"/c3a_asr_ci.csv "$WORK/public/data/"
cp "$DEST"/c3a_delta_ci.csv "$WORK/public/data/"
cp "$DEST"/c3a_cost_table.csv "$WORK/public/data/"
cp "$DEST"/c3a_guidance_distribution.csv "$WORK/public/data/"

cp "$ARTIFACT_DIR"/README.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/DATASET_CARD.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/DATASHEET.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/ETHICS.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/REPRODUCIBILITY.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/LICENSE_GATED.md "$WORK/gated/"
cp "$ARTIFACT_DIR"/metadata/MANIFEST.csv "$WORK/gated/metadata/"
cp "$ARTIFACT_DIR"/metadata/ACCESS_LEVELS.md "$WORK/gated/metadata/"
cp "$ARTIFACT_DIR"/metadata/croissant.json "$WORK/gated/metadata/"
cp "$ARTIFACT_DIR"/CHECKSUMS.sha256 "$WORK/gated/"
cp "$ARTIFACT_DIR"/human_eval/HUMAN_EVAL_SUMMARY.md "$WORK/gated/human_eval/"
cp "$DEST"/en_v2_results_7010.csv "$WORK/gated/data/"
cp "$DEST"/knowledge_base.json "$WORK/gated/data/"
cp "$DEST"/c3a_jbb_goals.json "$WORK/gated/data/"
cp -R "$DEST"/raw "$WORK/gated/data/"

GIT_REV="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
{
  echo "CascadeEval reviewer artifact"
  echo "version: 1.0.0"
  echo "git_revision: $GIT_REV"
  echo "generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$WORK/public/VERSION" > "$WORK/gated/VERSION"

(
  cd "$WORK/public"
  zip -qr "$DIST/cascadeeval_public_2026.zip" .
)

(
  cd "$WORK/gated"
  zip -qr "$DIST/cascadeeval_gated_2026.zip" .
)

(
  cd "$DIST"
  shasum -a 256 cascadeeval_public_2026.zip cascadeeval_gated_2026.zip > submission_sha256.txt
)

rm -rf "$WORK"

echo "Prepared HF dataset files in $DEST"
echo "Prepared archives in $DIST"
