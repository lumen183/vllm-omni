#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
HOST="127.0.0.1"
PORT="8091"
ENDPOINT="/v1/chat/completions"
BACKEND="openai-chat-omni"
DATASET_NAME="random"
NUM_PROMPTS="3"
NUM_WARMUPS="1"
MAX_CONCURRENCY="1"
REQUEST_RATE="inf"
RANDOM_INPUT_LEN="2500"
RANDOM_OUTPUT_LEN="900"
EXTRA_BODY='{"modalities":["text","audio"]}'
PERCENTILE_METRICS="ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,ttfc,tpoc,icl"
METRIC_PERCENTILES="50,90,99"
RESULT_DIR="/tmp/qwen3_omni_yuanrong_bench"
RESULT_FILENAME=""
SAVE_RESULT="true"
SAVE_DETAILED="false"
PRINT_STAGE="true"
IGNORE_EOS="true"
DISABLE_TQDM="true"
READY_CHECK_TIMEOUT_SEC="60"
VENV_PATH="${VLLM_OMNI_VENV:-/app/vllm_omni/.venv}"
DRY_RUN="false"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  run_yuanrong_cpu_rdma_two_node_bench.sh [options] [-- extra vllm bench serve args...]

This is a benchmark client wrapper. It does not start vLLM, does not pass
--stage-id, and does not create or register Omni stages. Start the Yuanrong
two-node service separately, then run this script against node-a's API port.

Common options:
  --host HOST                         API server host. Default: 127.0.0.1
  --port PORT                         API server port. Default: 8091
  --model MODEL                       Default: Qwen/Qwen3-Omni-30B-A3B-Instruct
  --endpoint PATH                     Default: /v1/chat/completions
  --backend BACKEND                   Default: openai-chat-omni
  --dataset-name NAME                 Default: random
  --num-prompts N                     Default: 3
  --num-warmups N                     Default: 1
  --max-concurrency N                 Default: 1
  --request-rate RATE                 Default: inf
  --random-input-len N                Default: 2500
  --random-output-len N               Default: 900
  --extra-body JSON                   Default: {"modalities":["text","audio"]}
  --percentile-metrics LIST           Default: ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,ttfc,tpoc,icl
  --metric-percentiles LIST           Default: 50,90,99
  --result-dir DIR                    Default: /tmp/qwen3_omni_yuanrong_bench
  --result-filename NAME              Optional explicit result JSON filename.
  --ready-check-timeout-sec SECONDS   Default: 60
  --venv PATH                         Python venv to source if present. Default: $VLLM_OMNI_VENV or /app/vllm_omni/.venv
  --save-detailed                     Include per-request details in result JSON.
  --no-save-result                    Do not write benchmark JSON.
  --no-print-stage                    Do not request/print per-stage metrics.
  --no-ignore-eos                     Do not pass --ignore-eos.
  --no-disable-tqdm                   Keep tqdm output.
  --dry-run                           Print command only.

Examples:
  ./run_yuanrong_cpu_rdma_two_node_bench.sh \
    --host NODE_A_API_IP --port 8091 --num-prompts 10 --max-concurrency 1

  ./run_yuanrong_cpu_rdma_two_node_bench.sh \
    --host NODE_A_API_IP --port 8091 --request-rate 2 --num-prompts 20 \
    --result-filename yuanrong-qwen3-omni-rdma.json
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

quote_cmd() {
  printf '%q ' "$@"
}

require_value() {
  local option_name="$1"
  local option_value="${2:-}"
  [[ -n "${option_value}" ]] || die "${option_name} requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) require_value "$1" "${2:-}"; HOST="$2"; shift 2 ;;
    --port) require_value "$1" "${2:-}"; PORT="$2"; shift 2 ;;
    --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --endpoint) require_value "$1" "${2:-}"; ENDPOINT="$2"; shift 2 ;;
    --backend) require_value "$1" "${2:-}"; BACKEND="$2"; shift 2 ;;
    --dataset-name) require_value "$1" "${2:-}"; DATASET_NAME="$2"; shift 2 ;;
    --num-prompts) require_value "$1" "${2:-}"; NUM_PROMPTS="$2"; shift 2 ;;
    --num-warmups) require_value "$1" "${2:-}"; NUM_WARMUPS="$2"; shift 2 ;;
    --max-concurrency) require_value "$1" "${2:-}"; MAX_CONCURRENCY="$2"; shift 2 ;;
    --request-rate) require_value "$1" "${2:-}"; REQUEST_RATE="$2"; shift 2 ;;
    --random-input-len) require_value "$1" "${2:-}"; RANDOM_INPUT_LEN="$2"; shift 2 ;;
    --random-output-len) require_value "$1" "${2:-}"; RANDOM_OUTPUT_LEN="$2"; shift 2 ;;
    --extra-body) require_value "$1" "${2:-}"; EXTRA_BODY="$2"; shift 2 ;;
    --percentile-metrics) require_value "$1" "${2:-}"; PERCENTILE_METRICS="$2"; shift 2 ;;
    --metric-percentiles) require_value "$1" "${2:-}"; METRIC_PERCENTILES="$2"; shift 2 ;;
    --result-dir) require_value "$1" "${2:-}"; RESULT_DIR="$2"; shift 2 ;;
    --result-filename) require_value "$1" "${2:-}"; RESULT_FILENAME="$2"; shift 2 ;;
    --ready-check-timeout-sec) require_value "$1" "${2:-}"; READY_CHECK_TIMEOUT_SEC="$2"; shift 2 ;;
    --venv) require_value "$1" "${2:-}"; VENV_PATH="$2"; shift 2 ;;
    --save-detailed) SAVE_DETAILED="true"; shift ;;
    --no-save-result) SAVE_RESULT="false"; shift ;;
    --no-print-stage) PRINT_STAGE="false"; shift ;;
    --no-ignore-eos) IGNORE_EOS="false"; shift ;;
    --no-disable-tqdm) DISABLE_TQDM="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "Unknown option: $1. Use --help." ;;
  esac
done

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
else
  echo "[WARN] venv not found at ${VENV_PATH}; using current PATH/python environment" >&2
fi

if [[ "${SAVE_RESULT}" == "true" ]]; then
  mkdir -p "${RESULT_DIR}"
fi

cmd=(vllm bench serve --omni
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --endpoint "${ENDPOINT}"
  --backend "${BACKEND}"
  --dataset-name "${DATASET_NAME}"
  --num-prompts "${NUM_PROMPTS}"
  --num-warmups "${NUM_WARMUPS}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --request-rate "${REQUEST_RATE}"
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --extra-body "${EXTRA_BODY}"
  --percentile-metrics "${PERCENTILE_METRICS}"
  --metric-percentiles "${METRIC_PERCENTILES}"
  --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SEC}"
)

if [[ "${IGNORE_EOS}" == "true" ]]; then
  cmd+=(--ignore-eos)
fi
if [[ "${DISABLE_TQDM}" == "true" ]]; then
  cmd+=(--disable-tqdm)
fi
if [[ "${PRINT_STAGE}" == "true" ]]; then
  cmd+=(--print-stage)
fi
if [[ "${SAVE_RESULT}" == "true" ]]; then
  cmd+=(--save-result --result-dir "${RESULT_DIR}")
  if [[ -n "${RESULT_FILENAME}" ]]; then
    cmd+=(--result-filename "${RESULT_FILENAME}")
  fi
fi
if [[ "${SAVE_DETAILED}" == "true" ]]; then
  cmd+=(--save-detailed)
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "[INFO] Benchmark target: http://${HOST}:${PORT}${ENDPOINT}"
echo "[INFO] This script only runs vllm bench serve; start the two-node service separately."
printf '[INFO] Command: '
quote_cmd "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

exec "${cmd[@]}"
