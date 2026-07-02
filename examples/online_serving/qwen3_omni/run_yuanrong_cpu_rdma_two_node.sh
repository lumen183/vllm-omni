#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
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
RDMA_NETDEV=""
RDMA_DEVICE_NAME=""
RDMA_PORT=""
RDMA_GID_INDEX=""
DEPLOY_CONFIG=""
OVERLAY_PATH=""
STAGE0_DEVICES="0"
STAGE1_DEVICES="0"
STAGE2_DEVICES="0"
DRY_RUN="false"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  run_yuanrong_cpu_rdma_two_node.sh --role node-a|node-b \
      --node-a-host <stage0_api_and_rdma_ip> \
      --node-b-host <stage1_stage2_rdma_ip> \
      [options] [-- extra vllm serve args...]

Topology:
  node-a: stage 0 (Thinker + OpenAI API server)
  node-b: stage 1 (Talker) and stage 2 (Code2Wav)

The script generates a deploy overlay that uses:
  stage0 -> stage1: YuanrongTransferEngineConnector, protocol=rdma, CPU host pool
  stage1 -> stage2: SharedMemoryConnector, same-node handoff on node-b

Required on both nodes:
  --node-a-host IP      Routable IP of node-a. Also used as the API bind host by default.
  --node-b-host IP      Routable IP of node-b.
  --role ROLE           node-a or node-b.

Common options:
  --model MODEL                         Default: Qwen/Qwen3-Omni-30B-A3B-Instruct
  --master-host HOST                    Default: node-a-host
  --master-port PORT                    Default: 26000
  --api-host HOST                       Default: 0.0.0.0
  --api-port PORT                       Default: 8091
  --connector-host HOST                 Override local advertised RDMA/ZMQ host for this node.
                                        Defaults to node-a-host on node-a and node-b-host on node-b.
  --connector-base-port PORT            Base ZMQ port. Actual KV port for stage0->1 is base+100.
                                        Default: 50051
  --rpc-port PORT|auto                  TransferEngine RPC port. Default: auto
  --pool-size BYTES                     CPU staging pool size per stage worker. Default: 4294967296
  --device-name cpu:*|auto              TransferEngine device_name. Default: auto
  --rdma-netdev NETDEV                  Export TRANSFER_ENGINE_CPU_RDMA_NETDEV.
  --rdma-device-name HCA                Export TRANSFER_ENGINE_CPU_RDMA_DEVICE_NAME.
  --rdma-port PORT                      Export TRANSFER_ENGINE_CPU_RDMA_PORT.
  --rdma-gid-index INDEX                Export TRANSFER_ENGINE_CPU_RDMA_GID_INDEX.
  --deploy-config PATH                  Base deploy YAML. Default: vllm_omni/deploy/qwen3_omni_moe.yaml
  --overlay-path PATH                   Where to write generated overlay YAML.
                                        Default: /tmp/qwen3_omni_yuanrong_cpu_rdma_<role>.yaml
  --stage0-devices DEVICES             Devices for stage 0 on node-a. Default: 0
  --stage1-devices DEVICES             Devices for stage 1 on node-b. Default: 0
  --stage2-devices DEVICES             Devices for stage 2 on node-b. Default: 0
  --dry-run                            Generate overlay and print commands without starting vLLM.

Examples:
  # Node A, start first:
  ./run_yuanrong_cpu_rdma_two_node.sh \
      --role node-a --node-a-host 10.10.10.1 --node-b-host 10.10.10.2 \
      --rdma-netdev ibp1142s0f1

  # Node B:
  ./run_yuanrong_cpu_rdma_two_node.sh \
      --role node-b --node-a-host 10.10.10.1 --node-b-host 10.10.10.2 \
      --rdma-netdev ibp1142s0f1

Notes:
  - This is CPU staging RDMA: GPU KV -> CPU pool -> RDMA -> CPU pool -> consumer.
  - Run with host networking and /dev/infiniband access in containers.
  - Ensure memlock is sufficient, for example: ulimit -l unlimited
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --node-a-host) NODE_A_HOST="$2"; shift 2 ;;
    --node-b-host) NODE_B_HOST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --master-host) MASTER_HOST="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --api-host) API_HOST="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --connector-host) CONNECTOR_HOST="$2"; shift 2 ;;
    --connector-base-port) CONNECTOR_BASE_PORT="$2"; shift 2 ;;
    --rpc-port) RPC_PORT="$2"; shift 2 ;;
    --pool-size) POOL_SIZE="$2"; shift 2 ;;
    --device-name) DEVICE_NAME="$2"; shift 2 ;;
    --rdma-netdev) RDMA_NETDEV="$2"; shift 2 ;;
    --rdma-device-name) RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --rdma-port) RDMA_PORT="$2"; shift 2 ;;
    --rdma-gid-index) RDMA_GID_INDEX="$2"; shift 2 ;;
    --deploy-config) DEPLOY_CONFIG="$2"; shift 2 ;;
    --overlay-path) OVERLAY_PATH="$2"; shift 2 ;;
    --stage0-devices) STAGE0_DEVICES="$2"; shift 2 ;;
    --stage1-devices) STAGE1_DEVICES="$2"; shift 2 ;;
    --stage2-devices) STAGE2_DEVICES="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "Unknown option: $1. Use --help." ;;
  esac
done

[[ "${ROLE}" == "node-a" || "${ROLE}" == "node-b" ]] || die "--role must be node-a or node-b"
[[ -n "${NODE_A_HOST}" ]] || die "--node-a-host is required"
[[ -n "${NODE_B_HOST}" ]] || die "--node-b-host is required"

if [[ -z "${DEPLOY_CONFIG}" ]]; then
  DEPLOY_CONFIG="${REPO_ROOT}/vllm_omni/deploy/qwen3_omni_moe.yaml"
fi
if [[ -z "${MASTER_HOST}" ]]; then
  MASTER_HOST="${NODE_A_HOST}"
fi
if [[ -z "${CONNECTOR_HOST}" ]]; then
  if [[ "${ROLE}" == "node-a" ]]; then
    CONNECTOR_HOST="${NODE_A_HOST}"
  else
    CONNECTOR_HOST="${NODE_B_HOST}"
  fi
fi
if [[ -z "${OVERLAY_PATH}" ]]; then
  OVERLAY_PATH="/tmp/qwen3_omni_yuanrong_cpu_rdma_${ROLE}.yaml"
fi

[[ -f "${DEPLOY_CONFIG}" ]] || die "Deploy config not found: ${DEPLOY_CONFIG}"
mkdir -p "$(dirname "${OVERLAY_PATH}")"

cat > "${OVERLAY_PATH}" <<YAML
base_config: ${DEPLOY_CONFIG}

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      initial_codec_chunk_frames: 4
      codec_chunk_frames: 25
      codec_left_context_frames: 25
  yuanrong_cpu_rdma_connector:
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
      to_stage_1: yuanrong_cpu_rdma_connector
  - stage_id: 1
    devices: "${STAGE1_DEVICES}"
    input_connectors:
      from_stage_0: yuanrong_cpu_rdma_connector
    output_connectors:
      to_stage_2: connector_of_shared_memory
  - stage_id: 2
    devices: "${STAGE2_DEVICES}"
    input_connectors:
      from_stage_1: connector_of_shared_memory
YAML

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

echo "[INFO] Role: ${ROLE}"
echo "[INFO] Overlay: ${OVERLAY_PATH}"
echo "[INFO] Connector host: ${CONNECTOR_HOST}"
echo "[INFO] Yuanrong CPU RDMA base ZMQ port: ${CONNECTOR_BASE_PORT} (stage0->1 KV port uses base+100)"
echo "[INFO] TransferEngine CPU RDMA local IP: ${TRANSFER_ENGINE_CPU_RDMA_LOCAL_IP}"
echo "[INFO] Master: ${MASTER_HOST}:${MASTER_PORT}"

if [[ "${ROLE}" == "node-a" ]]; then
  cmd=(vllm serve "${MODEL}" --omni \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --deploy-config "${OVERLAY_PATH}" \
    --stage-id 0 \
    --omni-master-address "${MASTER_HOST}" \
    --omni-master-port "${MASTER_PORT}" \
    "${EXTRA_ARGS[@]}")
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
  fi
  exec "${cmd[@]}"
else
  echo "[INFO] Starting node-b stage 1 and stage 2. Stop with Ctrl+C."
  cmd_stage1=(vllm serve "${MODEL}" --omni \
    --deploy-config "${OVERLAY_PATH}" \
    --stage-id 1 \
    --headless \
    --omni-master-address "${MASTER_HOST}" \
    --omni-master-port "${MASTER_PORT}" \
    "${EXTRA_ARGS[@]}")
  cmd_stage2=(vllm serve "${MODEL}" --omni \
    --deploy-config "${OVERLAY_PATH}" \
    --stage-id 2 \
    --headless \
    --omni-master-address "${MASTER_HOST}" \
    --omni-master-port "${MASTER_PORT}" \
    "${EXTRA_ARGS[@]}")
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[DRY-RUN stage1] '
    printf '%q ' "${cmd_stage1[@]}"
    printf '\n'
    printf '[DRY-RUN stage2] '
    printf '%q ' "${cmd_stage2[@]}"
    printf '\n'
    exit 0
  fi

  pids=()
  cleanup() {
    for pid in "${pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap cleanup INT TERM EXIT

  "${cmd_stage1[@]}" &
  pids+=("$!")

  "${cmd_stage2[@]}" &
  pids+=("$!")

  wait -n "${pids[@]}"
  exit $?
fi
