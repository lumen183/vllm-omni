#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL="ByteDance-Seed/BAGEL-7B-MoT"
ROLE=""
NODE_A_HOST=""
NODE_B_HOST=""
MASTER_HOST=""
MASTER_PORT="26000"
API_HOST="0.0.0.0"
API_PORT="8091"
CONNECTOR_HOST=""
CONNECTOR_BASE_PORT="50051"
RPC_PORT="auto"
POOL_SIZE="4294967296"
DEVICE_NAME="auto"
DEPLOY_CONFIG=""
OVERLAY_PATH=""
STAGE0_DEVICES="0"
STAGE1_DEVICES="0"
SSH_TARGET=""
SSH_PORT=""
SSH_OPTION=()
REMOTE_SCRIPT=""
REMOTE_WORKDIR=""
REMOTE_START_DELAY="5"
READY_TIMEOUT_SEC="900"
LOG_DIR=""
OUTPUT_DIR=""
VENV_PATH="${VLLM_OMNI_VENV:-/app/vllm_omni/.venv}"
RDMA_NETDEV=""
RDMA_DEVICE_NAME=""
RDMA_PORT=""
RDMA_GID_INDEX=""
NODE_A_RDMA_NETDEV=""
NODE_B_RDMA_NETDEV=""
NODE_A_RDMA_DEVICE_NAME=""
NODE_B_RDMA_DEVICE_NAME=""
NODE_A_RDMA_PORT=""
NODE_B_RDMA_PORT=""
NODE_A_RDMA_GID_INDEX=""
NODE_B_RDMA_GID_INDEX=""
DRY_RUN="false"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  run_yuanrong_transfer_engine_two_node_e2e.sh --role both|node-a|node-b \
      --node-a-host <stage0_api_and_rdma_ip> \
      --node-b-host <stage1_rdma_ip> \
      [options] [-- extra vllm serve args...]

Topology:
  node-a: BAGEL stage 0, OpenAI-compatible API server
  node-b: BAGEL stage 1, headless DiT worker
  stage0 -> stage1: YuanrongTransferEngineConnector, protocol=rdma, CPU memory pool

The "both" role starts node-a locally and node-b over ssh, waits for the API,
sends three concurrent text-to-image prompts, saves three images, then stops
both vLLM processes. If either vLLM process exits during startup, all processes
are killed and the script returns non-zero.

Required:
  --role ROLE                         both, node-a, or node-b
  --node-a-host HOST                  Routable host/IP for node-a
  --node-b-host HOST                  Routable host/IP for node-b

Common options:
  --model MODEL                       Default: ByteDance-Seed/BAGEL-7B-MoT
  --master-host HOST                  Default: node-a-host
  --master-port PORT                  Default: 26000
  --api-host HOST                     Default: 0.0.0.0
  --api-port PORT                     Default: 8091
  --connector-host HOST               Override this node's advertised connector host
  --connector-base-port PORT          Default: 50051
  --rpc-port PORT|auto                Default: auto
  --pool-size BYTES                   Default: 4294967296
  --device-name cpu:*|auto            Default: auto
  --deploy-config PATH                Default: vllm_omni/deploy/bagel.yaml
  --overlay-path PATH                 Default: /tmp/bagel_yuanrong_te_<role>.yaml
  --stage0-devices DEVICES            Stage 0 devices on node-a. Default: 0
  --stage1-devices DEVICES            Stage 1 devices on node-b. Default: 0
  --ssh-target TARGET                 SSH target for node-b in --role both. Default: node-b-host
  --ssh-port PORT                     SSH port for node-b
  --ssh-option OPTION                 Extra ssh option, repeatable
  --remote-script PATH                Default: same absolute path as local script
  --remote-workdir PATH               Default: dirname(remote-script)
  --remote-start-delay SECONDS        Delay after node-a before node-b. Default: 5
  --ready-timeout-sec SECONDS         API readiness timeout. Default: 900
  --log-dir DIR                       Default: /tmp/bagel_yuanrong_te_e2e_<timestamp>
  --output-dir DIR                    Default: <log-dir>/images
  --venv PATH                         Source PATH/bin/activate if present
  --rdma-netdev NETDEV                Export TRANSFER_ENGINE_CPU_RDMA_NETDEV
  --rdma-device-name HCA              Export TRANSFER_ENGINE_CPU_RDMA_DEVICE_NAME
  --rdma-port PORT                    Export TRANSFER_ENGINE_CPU_RDMA_PORT
  --rdma-gid-index INDEX              Export TRANSFER_ENGINE_CPU_RDMA_GID_INDEX
  --node-a-rdma-netdev NETDEV         Node-a override for --role both
  --node-b-rdma-netdev NETDEV         Node-b override for --role both
  --node-a-rdma-device-name HCA       Node-a override for --role both
  --node-b-rdma-device-name HCA       Node-b override for --role both
  --node-a-rdma-port PORT             Node-a override for --role both
  --node-b-rdma-port PORT             Node-b override for --role both
  --node-a-rdma-gid-index INDEX       Node-a override for --role both
  --node-b-rdma-gid-index INDEX       Node-b override for --role both
  --dry-run                           Print commands without starting vLLM

Example:
  ./examples/online_serving/bagel/run_yuanrong_transfer_engine_two_node_e2e.sh \
      --role both \
      --node-a-host 10.10.10.1 \
      --node-b-host 10.10.10.2 \
      --ssh-target node-b \
      --stage0-devices 0 \
      --stage1-devices 0 \
      --node-a-rdma-netdev ibp1142s0f1 \
      --node-b-rdma-netdev ibp1142s0f1
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_value() {
  local option_name="$1"
  local option_value="${2:-}"
  [[ -n "${option_value}" ]] || die "${option_name} requires a value"
}

quote_cmd() {
  printf '%q ' "$@"
}

append_if_set() {
  local -n out_ref="$1"
  local option_name="$2"
  local option_value="$3"
  if [[ -n "${option_value}" ]]; then
    out_ref+=("${option_name}" "${option_value}")
  fi
}

append_extra_args() {
  local -n out_ref="$1"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    out_ref+=(-- "${EXTRA_ARGS[@]}")
  fi
}

source_venv_if_present() {
  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
  else
    echo "[WARN] venv not found at ${VENV_PATH}; using current environment" >&2
  fi
}

write_overlay() {
  mkdir -p "$(dirname "${OVERLAY_PATH}")"
  cat > "${OVERLAY_PATH}" <<YAML
base_config: ${DEPLOY_CONFIG}

connectors:
  yuanrong_te_connector:
    name: YuanrongTransferEngineConnector
    extra:
      host: "${CONNECTOR_HOST}"
      zmq_port: ${CONNECTOR_BASE_PORT}
      rpc_port: "${RPC_PORT}"
      protocol: "rdma"
      device_name: "${DEVICE_NAME}"
      memory_pool_size: ${POOL_SIZE}
      memory_pool_device: "cpu"
      sender_host: "${NODE_A_HOST}"

stages:
  - stage_id: 0
    devices: "${STAGE0_DEVICES}"
    output_connectors:
      to_stage_1: yuanrong_te_connector
  - stage_id: 1
    devices: "${STAGE1_DEVICES}"
    input_connectors:
      from_stage_0: yuanrong_te_connector
YAML
}

export_rdma_env() {
  export TRANSFER_ENGINE_CPU_RDMA_LOCAL_IP="${CONNECTOR_HOST}"
  if [[ -n "${RDMA_NETDEV}" ]]; then
    export TRANSFER_ENGINE_CPU_RDMA_NETDEV="${RDMA_NETDEV}"
  fi
  if [[ -n "${RDMA_DEVICE_NAME}" ]]; then
    export TRANSFER_ENGINE_CPU_RDMA_DEVICE_NAME="${RDMA_DEVICE_NAME}"
  fi
  if [[ -n "${RDMA_PORT}" ]]; then
    export TRANSFER_ENGINE_CPU_RDMA_PORT="${RDMA_PORT}"
  fi
  if [[ -n "${RDMA_GID_INDEX}" ]]; then
    export TRANSFER_ENGINE_CPU_RDMA_GID_INDEX="${RDMA_GID_INDEX}"
  fi
}

build_child_args() {
  local child_role="$1"
  local child_rdma_netdev="$2"
  local child_rdma_device_name="$3"
  local child_rdma_port="$4"
  local child_rdma_gid_index="$5"
  local -n out_ref="$6"

  out_ref=(
    --role "${child_role}"
    --node-a-host "${NODE_A_HOST}"
    --node-b-host "${NODE_B_HOST}"
    --model "${MODEL}"
    --master-host "${MASTER_HOST}"
    --master-port "${MASTER_PORT}"
    --connector-base-port "${CONNECTOR_BASE_PORT}"
    --rpc-port "${RPC_PORT}"
    --pool-size "${POOL_SIZE}"
    --device-name "${DEVICE_NAME}"
    --deploy-config "${DEPLOY_CONFIG}"
    --stage0-devices "${STAGE0_DEVICES}"
    --stage1-devices "${STAGE1_DEVICES}"
    --venv "${VENV_PATH}"
  )

  append_if_set "$6" --rdma-netdev "${child_rdma_netdev}"
  append_if_set "$6" --rdma-device-name "${child_rdma_device_name}"
  append_if_set "$6" --rdma-port "${child_rdma_port}"
  append_if_set "$6" --rdma-gid-index "${child_rdma_gid_index}"

  if [[ "${child_role}" == "node-a" ]]; then
    out_ref+=(--api-host "${API_HOST}" --api-port "${API_PORT}")
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    out_ref+=(--dry-run)
  fi
  append_extra_args "$6"
}

wait_for_api_or_exit() {
  local url="http://${NODE_A_HOST}:${API_PORT}/v1/models"
  local deadline=$((SECONDS + READY_TIMEOUT_SEC))

  while (( SECONDS < deadline )); do
    if ! kill -0 "${NODE_A_PID}" 2>/dev/null; then
      echo "[ERROR] node-a vLLM exited during startup" >&2
      wait "${NODE_A_PID}" || true
      return 1
    fi
    if ! kill -0 "${NODE_B_PID}" 2>/dev/null; then
      echo "[ERROR] node-b ssh/vLLM exited during startup" >&2
      wait "${NODE_B_PID}" || true
      return 1
    fi
    if python3 - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "[INFO] API is ready: ${url}"
      return 0
    fi
    sleep 2
  done

  echo "[ERROR] API did not become ready within ${READY_TIMEOUT_SEC}s: ${url}" >&2
  return 1
}

run_image_bench() {
  mkdir -p "${OUTPUT_DIR}"
  python3 - "$NODE_A_HOST" "$API_PORT" "$MODEL" "$OUTPUT_DIR" <<'PY'
import base64
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

host, port, model, out_dir = sys.argv[1:5]
out_path = Path(out_dir)
url = f"http://{host}:{port}/v1/chat/completions"

prompts = [
    "A cute cat sitting on a wooden chair, warm window light.",
    "A futuristic city skyline at twilight, cyberpunk style, ultra detailed.",
    "A small red cube on a white table, studio lighting, sharp focus.",
]

def image_candidates(obj):
    if isinstance(obj, dict):
        value = obj.get("b64_json")
        if isinstance(value, str) and value:
            yield value
        image_url = obj.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url")
            if isinstance(value, str) and value:
                yield value
        value = obj.get("url")
        if isinstance(value, str) and value:
            yield value
        for child in obj.values():
            yield from image_candidates(child)
    elif isinstance(obj, list):
        for child in obj:
            yield from image_candidates(child)

def decode_image(candidate):
    if candidate.startswith("data:image"):
        match = re.match(r"^data:image/[^;]+;base64,(.*)$", candidate, re.S)
        if not match:
            raise ValueError("malformed data image URL")
        return base64.b64decode(match.group(1)), "png"
    return base64.b64decode(candidate), "png"

def request_one(index, prompt):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}
                ],
            }
        ],
        "modalities": ["image"],
        "height": 512,
        "width": 512,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": 42 + index,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"request {index} failed with HTTP {exc.code}: {body[:1000]}") from exc

    response_file = out_path / f"bagel_yuanrong_response_{index}.json"
    response_file.write_bytes(raw)
    payload = json.loads(raw.decode("utf-8"))
    for candidate in image_candidates(payload):
        image_bytes, ext = decode_image(candidate)
        image_file = out_path / f"bagel_yuanrong_{index}.{ext}"
        image_file.write_bytes(image_bytes)
        return {
            "index": index,
            "prompt": prompt,
            "image": str(image_file),
            "response": str(response_file),
            "seconds": round(time.time() - started, 3),
            "bytes": len(image_bytes),
        }
    raise RuntimeError(f"request {index} completed but no image payload was found")

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(request_one, i + 1, prompt) for i, prompt in enumerate(prompts)]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

results.sort(key=lambda item: item["index"])
summary_file = out_path / "summary.json"
summary_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"summary": str(summary_file), "results": results}, ensure_ascii=False, indent=2))
PY
}

run_both_nodes() {
  [[ -z "${CONNECTOR_HOST}" ]] || die "--connector-host is per-node; do not use it with --role both"
  [[ -z "${OVERLAY_PATH}" ]] || die "--overlay-path is per-node; do not use it with --role both"

  if [[ -z "${SSH_TARGET}" ]]; then
    SSH_TARGET="${NODE_B_HOST}"
  fi
  if [[ -z "${REMOTE_SCRIPT}" ]]; then
    REMOTE_SCRIPT="${SCRIPT_DIR}/run_yuanrong_transfer_engine_two_node_e2e.sh"
  fi
  if [[ -z "${REMOTE_WORKDIR}" ]]; then
    REMOTE_WORKDIR="$(dirname "${REMOTE_SCRIPT}")"
  fi
  if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="/tmp/bagel_yuanrong_te_e2e_$(date +%Y%m%d_%H%M%S)"
  fi
  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${LOG_DIR}/images"
  fi
  mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
  exec > >(tee -a "${LOG_DIR}/controller.log") 2>&1

  local node_a_args=()
  local node_b_args=()
  build_child_args node-a "${NODE_A_RDMA_NETDEV:-${RDMA_NETDEV}}" \
    "${NODE_A_RDMA_DEVICE_NAME:-${RDMA_DEVICE_NAME}}" \
    "${NODE_A_RDMA_PORT:-${RDMA_PORT}}" \
    "${NODE_A_RDMA_GID_INDEX:-${RDMA_GID_INDEX}}" node_a_args
  build_child_args node-b "${NODE_B_RDMA_NETDEV:-${RDMA_NETDEV}}" \
    "${NODE_B_RDMA_DEVICE_NAME:-${RDMA_DEVICE_NAME}}" \
    "${NODE_B_RDMA_PORT:-${RDMA_PORT}}" \
    "${NODE_B_RDMA_GID_INDEX:-${RDMA_GID_INDEX}}" node_b_args

  local local_cmd=("${SCRIPT_DIR}/run_yuanrong_transfer_engine_two_node_e2e.sh" "${node_a_args[@]}")
  local remote_cmd=("${REMOTE_SCRIPT}" "${node_b_args[@]}")
  local remote_line="cd $(printf '%q' "${REMOTE_WORKDIR}") && $(quote_cmd "${remote_cmd[@]}")"
  local ssh_cmd=(ssh)
  if [[ -n "${SSH_PORT}" ]]; then
    ssh_cmd+=(-p "${SSH_PORT}")
  fi
  ssh_cmd+=("${SSH_OPTION[@]}" "${SSH_TARGET}" "${remote_line}")

  echo "[INFO] Log dir: ${LOG_DIR}"
  echo "[INFO] Image output dir: ${OUTPUT_DIR}"
  echo "[INFO] Node-a command: $(quote_cmd "${local_cmd[@]}")"
  echo "[INFO] Node-b SSH command: $(quote_cmd "${ssh_cmd[@]}")"

  if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
  fi

  NODE_A_PID=""
  NODE_B_PID=""
  cleanup() {
    local status=$?
    if [[ -n "${NODE_A_PID}" ]]; then
      kill "${NODE_A_PID}" 2>/dev/null || true
    fi
    if [[ -n "${NODE_B_PID}" ]]; then
      kill "${NODE_B_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "${status}"
  }
  trap cleanup INT TERM EXIT

  "${local_cmd[@]}" >"${LOG_DIR}/node-a.log" 2>&1 &
  NODE_A_PID=$!

  sleep "${REMOTE_START_DELAY}"

  "${ssh_cmd[@]}" >"${LOG_DIR}/node-b.log" 2>&1 &
  NODE_B_PID=$!

  if ! wait_for_api_or_exit; then
    return 1
  fi

  if ! run_image_bench >"${LOG_DIR}/bench.log" 2>&1; then
    echo "[ERROR] image bench failed; see ${LOG_DIR}/bench.log" >&2
    return 1
  fi

  if ! kill -0 "${NODE_A_PID}" 2>/dev/null || ! kill -0 "${NODE_B_PID}" 2>/dev/null; then
    echo "[ERROR] a vLLM process exited before cleanup after bench" >&2
    return 1
  fi

  echo "[INFO] Bench succeeded. Images and responses saved under ${OUTPUT_DIR}"
  echo "[INFO] Logs saved under ${LOG_DIR}"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) require_value "$1" "${2:-}"; ROLE="$2"; shift 2 ;;
    --node-a-host) require_value "$1" "${2:-}"; NODE_A_HOST="$2"; shift 2 ;;
    --node-b-host) require_value "$1" "${2:-}"; NODE_B_HOST="$2"; shift 2 ;;
    --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --master-host) require_value "$1" "${2:-}"; MASTER_HOST="$2"; shift 2 ;;
    --master-port) require_value "$1" "${2:-}"; MASTER_PORT="$2"; shift 2 ;;
    --api-host) require_value "$1" "${2:-}"; API_HOST="$2"; shift 2 ;;
    --api-port) require_value "$1" "${2:-}"; API_PORT="$2"; shift 2 ;;
    --connector-host) require_value "$1" "${2:-}"; CONNECTOR_HOST="$2"; shift 2 ;;
    --connector-base-port) require_value "$1" "${2:-}"; CONNECTOR_BASE_PORT="$2"; shift 2 ;;
    --rpc-port) require_value "$1" "${2:-}"; RPC_PORT="$2"; shift 2 ;;
    --pool-size) require_value "$1" "${2:-}"; POOL_SIZE="$2"; shift 2 ;;
    --device-name) require_value "$1" "${2:-}"; DEVICE_NAME="$2"; shift 2 ;;
    --deploy-config) require_value "$1" "${2:-}"; DEPLOY_CONFIG="$2"; shift 2 ;;
    --overlay-path) require_value "$1" "${2:-}"; OVERLAY_PATH="$2"; shift 2 ;;
    --stage0-devices) require_value "$1" "${2:-}"; STAGE0_DEVICES="$2"; shift 2 ;;
    --stage1-devices) require_value "$1" "${2:-}"; STAGE1_DEVICES="$2"; shift 2 ;;
    --ssh-target) require_value "$1" "${2:-}"; SSH_TARGET="$2"; shift 2 ;;
    --ssh-port) require_value "$1" "${2:-}"; SSH_PORT="$2"; shift 2 ;;
    --ssh-option) require_value "$1" "${2:-}"; SSH_OPTION+=("$2"); shift 2 ;;
    --remote-script) require_value "$1" "${2:-}"; REMOTE_SCRIPT="$2"; shift 2 ;;
    --remote-workdir) require_value "$1" "${2:-}"; REMOTE_WORKDIR="$2"; shift 2 ;;
    --remote-start-delay) require_value "$1" "${2:-}"; REMOTE_START_DELAY="$2"; shift 2 ;;
    --ready-timeout-sec) require_value "$1" "${2:-}"; READY_TIMEOUT_SEC="$2"; shift 2 ;;
    --log-dir) require_value "$1" "${2:-}"; LOG_DIR="$2"; shift 2 ;;
    --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --venv) require_value "$1" "${2:-}"; VENV_PATH="$2"; shift 2 ;;
    --rdma-netdev) require_value "$1" "${2:-}"; RDMA_NETDEV="$2"; shift 2 ;;
    --rdma-device-name) require_value "$1" "${2:-}"; RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --rdma-port) require_value "$1" "${2:-}"; RDMA_PORT="$2"; shift 2 ;;
    --rdma-gid-index) require_value "$1" "${2:-}"; RDMA_GID_INDEX="$2"; shift 2 ;;
    --node-a-rdma-netdev) require_value "$1" "${2:-}"; NODE_A_RDMA_NETDEV="$2"; shift 2 ;;
    --node-b-rdma-netdev) require_value "$1" "${2:-}"; NODE_B_RDMA_NETDEV="$2"; shift 2 ;;
    --node-a-rdma-device-name) require_value "$1" "${2:-}"; NODE_A_RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --node-b-rdma-device-name) require_value "$1" "${2:-}"; NODE_B_RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --node-a-rdma-port) require_value "$1" "${2:-}"; NODE_A_RDMA_PORT="$2"; shift 2 ;;
    --node-b-rdma-port) require_value "$1" "${2:-}"; NODE_B_RDMA_PORT="$2"; shift 2 ;;
    --node-a-rdma-gid-index) require_value "$1" "${2:-}"; NODE_A_RDMA_GID_INDEX="$2"; shift 2 ;;
    --node-b-rdma-gid-index) require_value "$1" "${2:-}"; NODE_B_RDMA_GID_INDEX="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "Unknown option: $1. Use --help." ;;
  esac
done

[[ "${ROLE}" == "both" || "${ROLE}" == "node-a" || "${ROLE}" == "node-b" ]] || die "--role must be both, node-a, or node-b"
[[ -n "${NODE_A_HOST}" ]] || die "--node-a-host is required"
[[ -n "${NODE_B_HOST}" ]] || die "--node-b-host is required"

if [[ -z "${DEPLOY_CONFIG}" ]]; then
  DEPLOY_CONFIG="${REPO_ROOT}/vllm_omni/deploy/bagel.yaml"
fi
if [[ -z "${MASTER_HOST}" ]]; then
  MASTER_HOST="${NODE_A_HOST}"
fi
[[ -f "${DEPLOY_CONFIG}" ]] || die "Deploy config not found: ${DEPLOY_CONFIG}"

if [[ "${ROLE}" == "both" ]]; then
  run_both_nodes
  exit $?
fi

if [[ -z "${CONNECTOR_HOST}" ]]; then
  if [[ "${ROLE}" == "node-a" ]]; then
    CONNECTOR_HOST="${NODE_A_HOST}"
  else
    CONNECTOR_HOST="${NODE_B_HOST}"
  fi
fi
if [[ -z "${OVERLAY_PATH}" ]]; then
  OVERLAY_PATH="/tmp/bagel_yuanrong_te_${ROLE}.yaml"
fi

source_venv_if_present
write_overlay
export_rdma_env

echo "[INFO] Role: ${ROLE}"
echo "[INFO] Overlay: ${OVERLAY_PATH}"
echo "[INFO] Connector host: ${CONNECTOR_HOST}"
echo "[INFO] TransferEngine CPU RDMA local IP: ${TRANSFER_ENGINE_CPU_RDMA_LOCAL_IP}"
echo "[INFO] Master: ${MASTER_HOST}:${MASTER_PORT}"

if [[ "${ROLE}" == "node-a" ]]; then
  cmd=(vllm serve "${MODEL}" --omni
    --host "${API_HOST}"
    --port "${API_PORT}"
    --deploy-config "${OVERLAY_PATH}"
    --stage-id 0
    --omni-master-address "${MASTER_HOST}"
    --omni-master-port "${MASTER_PORT}"
    "${EXTRA_ARGS[@]}")
else
  cmd=(vllm serve "${MODEL}" --omni
    --deploy-config "${OVERLAY_PATH}"
    --stage-id 1
    --headless
    --omni-master-address "${MASTER_HOST}"
    --omni-master-port "${MASTER_PORT}"
    "${EXTRA_ARGS[@]}")
fi

printf '[INFO] Command: '
quote_cmd "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

exec "${cmd[@]}"
