#!/bin/bash
# =============================================================================
# C3A パイロット評価スクリプト
#
# 概要: 複数の防御構成に対して 10 ゴールをテスト
# 目的: 本番実行前の動作確認・パフォーマンス検証
# =============================================================================

set -o pipefail  # パイプ内のエラーを検出

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
N_GOALS=10
MAX_RETRIES=3
RETRY_DELAY=30  # 秒
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# 評価する構成（config.yaml の name）
CONFIGS=(
    "config_1"  # LlamaGuard3 + gpt-4o-mini + LlamaGuard3
    "config_2"  # WildGuard + gpt-4o + LlamaGuard4
    "config_3"  # LlamaGuard4 + gpt-4o + WildGuard
    "config_4"  # WildGuard + Llama-3.1-8B + ShieldGemma（全ローカル）
    "config_5"  # ShieldGemma + gpt-4o-mini + WildGuard
)

# 評価するエージェント
AGENTS=("c3a")

# -----------------------------------------------------------------------------
# パス設定
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# .env ファイルを読み込む
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a  # export all variables
    source "${PROJECT_ROOT}/.env"
    set +a
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${PROJECT_ROOT}/results/c3a/pilot_${TIMESTAMP}"
LOG_FILE="${OUTPUT_BASE}/run.log"

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local msg="$*"
    local ts=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${ts}] [${level}] ${msg}" | tee -a "${LOG_FILE}"
}

log_info()  { log "INFO" "$@"; }
log_warn()  { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# -----------------------------------------------------------------------------
# 初期化
# -----------------------------------------------------------------------------
mkdir -p "${OUTPUT_BASE}"
log_info "=========================================="
log_info "C3A パイロット評価開始"
log_info "=========================================="
log_info "出力ディレクトリ: ${OUTPUT_BASE}"
log_info "GPU: ${GPU_ID}"
log_info "ゴール数: ${N_GOALS}"
log_info "構成数: ${#CONFIGS[@]}"
log_info "エージェント: ${AGENTS[*]}"
log_info ""

# -----------------------------------------------------------------------------
# 環境チェック
# -----------------------------------------------------------------------------
log_info "環境チェック..."

# Python 環境
if ! command -v uv &> /dev/null; then
    log_error "uv が見つかりません"
    exit 1
fi

# GPU チェック
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${GPU_ID}" 2>/dev/null || echo "不明")
    log_info "GPU ${GPU_ID}: ${GPU_INFO}"
else
    log_warn "nvidia-smi が見つかりません（CPU モードの可能性）"
fi

# API キーチェック
if [[ -z "${OPENAI_API_KEY}" ]]; then
    log_warn "OPENAI_API_KEY が設定されていません"
fi

log_info ""

# -----------------------------------------------------------------------------
# 各構成を実行
# -----------------------------------------------------------------------------
declare -A RESULTS  # 結果を格納する連想配列
TOTAL_CONFIGS=${#CONFIGS[@]}
TOTAL_AGENTS=${#AGENTS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

for config in "${CONFIGS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        RUN_NAME="${config}_${agent}"
        RUN_OUTPUT="${OUTPUT_BASE}/${RUN_NAME}"
        mkdir -p "${RUN_OUTPUT}"

        log_info "------------------------------------------"
        log_info "実行: ${RUN_NAME}"
        log_info "------------------------------------------"

        # リトライループ
        attempt=1
        success=false

        while [[ ${attempt} -le ${MAX_RETRIES} ]]; do
            log_info "試行 ${attempt}/${MAX_RETRIES}..."

            # 評価実行
            CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python experiments/c3a/part2_evaluate.py \
                --configs "${config}" \
                --agents "${agent}" \
                --n-goals "${N_GOALS}" \
                --verbose \
                -y \
                2>&1 | tee -a "${RUN_OUTPUT}/output.log"

            exit_code=${PIPESTATUS[0]}

            if [[ ${exit_code} -eq 0 ]]; then
                log_info "成功: ${RUN_NAME}"
                success=true
                break
            else
                log_warn "失敗 (exit code: ${exit_code})"
                if [[ ${attempt} -lt ${MAX_RETRIES} ]]; then
                    log_info "${RETRY_DELAY}秒後にリトライ..."
                    sleep "${RETRY_DELAY}"
                fi
            fi

            ((attempt++))
        done

        # 結果を記録
        if ${success}; then
            RESULTS["${RUN_NAME}"]="SUCCESS"
            ((SUCCESS_COUNT++))
        else
            RESULTS["${RUN_NAME}"]="FAILED"
            ((FAIL_COUNT++))
            log_error "最終失敗: ${RUN_NAME}"
        fi

        # 結果ファイルをコピー
        if [[ -d "${PROJECT_ROOT}/results/c3a/part2" ]]; then
            cp -r "${PROJECT_ROOT}/results/c3a/part2"/* "${RUN_OUTPUT}/" 2>/dev/null || true
        fi

        log_info ""
    done
done

# -----------------------------------------------------------------------------
# サマリー出力
# -----------------------------------------------------------------------------
log_info "=========================================="
log_info "実行完了サマリー"
log_info "=========================================="
log_info "成功: ${SUCCESS_COUNT} / $((SUCCESS_COUNT + FAIL_COUNT))"
log_info "失敗: ${FAIL_COUNT}"
log_info ""
log_info "詳細結果:"

for run_name in "${!RESULTS[@]}"; do
    status="${RESULTS[${run_name}]}"
    if [[ "${status}" == "SUCCESS" ]]; then
        log_info "  [OK] ${run_name}"
    else
        log_error "  [NG] ${run_name}"
    fi
done

# -----------------------------------------------------------------------------
# 結果の集約
# -----------------------------------------------------------------------------
log_info ""
log_info "結果ファイル集約中..."

SUMMARY_FILE="${OUTPUT_BASE}/summary.json"
cat > "${SUMMARY_FILE}" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "n_goals": ${N_GOALS},
  "configs": $(printf '%s\n' "${CONFIGS[@]}" | jq -R . | jq -s .),
  "agents": $(printf '%s\n' "${AGENTS[@]}" | jq -R . | jq -s .),
  "results": {
    "success_count": ${SUCCESS_COUNT},
    "fail_count": ${FAIL_COUNT},
    "details": $(for k in "${!RESULTS[@]}"; do echo "\"$k\": \"${RESULTS[$k]}\""; done | paste -sd, | sed 's/^/{/;s/$/}/')
  }
}
EOF

log_info "サマリー保存: ${SUMMARY_FILE}"
log_info ""
log_info "出力ディレクトリ: ${OUTPUT_BASE}"
log_info "=========================================="

# 失敗があった場合は非ゼロで終了
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    exit 1
fi

exit 0
