#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source /app/vllm_omni/.venv/bin/activate
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
NODE_A_RDMA_NETDEV=""
NODE_B_RDMA_NETDEV=""
NODE_A_RDMA_DEVICE_NAME=""
NODE_B_RDMA_DEVICE_NAME=""
NODE_A_RDMA_PORT=""
NODE_B_RDMA_PORT=""
NODE_A_RDMA_GID_INDEX=""
NODE_B_RDMA_GID_INDEX=""
DEPLOY_CONFIG=""
OVERLAY_PATH=""
STAGE0_DEVICES="0"
STAGE1_DEVICES="0"
STAGE2_DEVICES="0"
SSH_TARGET=""
SSH_PORT=""
SSH_OPTION=()
REMOTE_SCRIPT=""
REMOTE_WORKDIR=""
REMOTE_START_DELAY="5"
DRY_RUN="false"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  run_yuanrong_cpu_rdma_two_node.sh --role both|node-a|node-b \
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
  --role ROLE           both, node-a, or node-b. Use both to launch node-b over ssh
                        and node-a locally from one machine.

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
  --node-a-rdma-netdev NETDEV           Node-a specific RDMA netdev for --role both.
  --node-b-rdma-netdev NETDEV           Node-b specific RDMA netdev for --role both.
  --node-a-rdma-device-name HCA         Node-a specific HCA for --role both.
  --node-b-rdma-device-name HCA         Node-b specific HCA for --role both.
  --node-a-rdma-port PORT               Node-a specific RDMA port for --role both.
  --node-b-rdma-port PORT               Node-b specific RDMA port for --role both.
  --node-a-rdma-gid-index INDEX         Node-a specific GID index for --role both.
  --node-b-rdma-gid-index INDEX         Node-b specific GID index for --role both.
  --deploy-config PATH                  Base deploy YAML. Default: vllm_omni/deploy/qwen3_omni_moe.yaml
  --overlay-path PATH                   Where to write generated overlay YAML.
                                        Default: /tmp/qwen3_omni_yuanrong_cpu_rdma_<role>.yaml
  --stage0-devices DEVICES             Devices for stage 0 on node-a. Default: 0
  --stage1-devices DEVICES             Devices for stage 1 on node-b. Default: 0
  --stage2-devices DEVICES             Devices for stage 2 on node-b. Default: 0
  --ssh-target TARGET                   SSH target for node-b in --role both. Default: node-b-host.
                                        This is independent from node-b-host, which is the RDMA/ZMQ host.
                                        Examples: root@10.90.67.90, peer.
  --ssh-port PORT                       SSH port for node-b in --role both.
  --ssh-option OPTION                   Extra ssh option. Repeat as needed, for example
                                        --ssh-option -p --ssh-option 2222.
  --remote-script PATH                  Remote script path. Default: same absolute path as local script.
  --remote-workdir PATH                 Remote workdir. Default: dirname(remote-script).
  --remote-start-delay SECONDS          Delay after launching node-b before node-a. Default: 5.
  --dry-run                            Generate overlay and print commands without starting vLLM.

Examples:
  # One-command two-node launch from node-a, using cards 6,7 on both nodes:
  ./run_yuanrong_cpu_rdma_two_node.sh \
      --role both --node-a-host 10.10.10.1 --node-b-host 10.10.10.2 \
      --stage0-devices "6,7" --stage1-devices "6" --stage2-devices "7" \
      --node-a-rdma-netdev ibp1142s0f1 --node-b-rdma-netdev ibp1142s0f1 \
      --ssh-target root@10.90.67.90 --ssh-port 2222

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
    --stage2-devices "${STAGE2_DEVICES}"
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

run_both_nodes() {
  [[ -z "${CONNECTOR_HOST}" ]] || die "--connector-host is per-node; do not use it with --role both"
  [[ -z "${OVERLAY_PATH}" ]] || die "--overlay-path is per-node; do not use it with --role both"

  local node_a_rdma_netdev="${NODE_A_RDMA_NETDEV:-${RDMA_NETDEV}}"
  local node_b_rdma_netdev="${NODE_B_RDMA_NETDEV:-${RDMA_NETDEV}}"
  local node_a_rdma_device_name="${NODE_A_RDMA_DEVICE_NAME:-${RDMA_DEVICE_NAME}}"
  local node_b_rdma_device_name="${NODE_B_RDMA_DEVICE_NAME:-${RDMA_DEVICE_NAME}}"
  local node_a_rdma_port="${NODE_A_RDMA_PORT:-${RDMA_PORT}}"
  local node_b_rdma_port="${NODE_B_RDMA_PORT:-${RDMA_PORT}}"
  local node_a_rdma_gid_index="${NODE_A_RDMA_GID_INDEX:-${RDMA_GID_INDEX}}"
  local node_b_rdma_gid_index="${NODE_B_RDMA_GID_INDEX:-${RDMA_GID_INDEX}}"

  if [[ -z "${SSH_TARGET}" ]]; then
    SSH_TARGET="${NODE_B_HOST}"
  fi
  if [[ -z "${REMOTE_SCRIPT}" ]]; then
    REMOTE_SCRIPT="${SCRIPT_DIR}/run_yuanrong_cpu_rdma_two_node.sh"
  fi
  if [[ -z "${REMOTE_WORKDIR}" ]]; then
    REMOTE_WORKDIR="$(dirname "${REMOTE_SCRIPT}")"
  fi

  local node_a_args=()
  local node_b_args=()
  build_child_args node-a "${node_a_rdma_netdev}" "${node_a_rdma_device_name}" \
    "${node_a_rdma_port}" "${node_a_rdma_gid_index}" node_a_args
  build_child_args node-b "${node_b_rdma_netdev}" "${node_b_rdma_device_name}" \
    "${node_b_rdma_port}" "${node_b_rdma_gid_index}" node_b_args

  local local_cmd=("${SCRIPT_DIR}/run_yuanrong_cpu_rdma_two_node.sh" "${node_a_args[@]}")
  local remote_cmd=("${REMOTE_SCRIPT}" "${node_b_args[@]}")
  local remote_line
  remote_line="cd $(printf '%q' "${REMOTE_WORKDIR}") && $(quote_cmd "${remote_cmd[@]}")"
  local ssh_cmd=(ssh)
  if [[ -n "${SSH_PORT}" ]]; then
    ssh_cmd+=(-p "${SSH_PORT}")
  fi
  ssh_cmd+=("${SSH_OPTION[@]}" "${SSH_TARGET}" "${remote_line}")

  echo "[INFO] Role: both"
  echo "[INFO] SSH target: ${SSH_TARGET}"
  echo "[INFO] Remote script: ${REMOTE_SCRIPT}"
  echo "[INFO] Node-a command:"
  printf '  '
  quote_cmd "${local_cmd[@]}"
  printf '\n'
  echo "[INFO] Node-b SSH command:"
  printf '  '
  quote_cmd "${ssh_cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "true" ]]; then
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

  "${ssh_cmd[@]}" &
  pids+=("$!")

  sleep "${REMOTE_START_DELAY}"
  if ! kill -0 "${pids[0]}" 2>/dev/null; then
    echo "[ERROR] Remote node-b ssh command exited before node-a startup" >&2
    wait "${pids[0]}" || true
    exit 1
  fi

  "${local_cmd[@]}" &
  pids+=("$!")

  wait -n "${pids[@]}"
  exit $?
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
    --node-a-rdma-netdev) NODE_A_RDMA_NETDEV="$2"; shift 2 ;;
    --node-b-rdma-netdev) NODE_B_RDMA_NETDEV="$2"; shift 2 ;;
    --node-a-rdma-device-name) NODE_A_RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --node-b-rdma-device-name) NODE_B_RDMA_DEVICE_NAME="$2"; shift 2 ;;
    --node-a-rdma-port) NODE_A_RDMA_PORT="$2"; shift 2 ;;
    --node-b-rdma-port) NODE_B_RDMA_PORT="$2"; shift 2 ;;
    --node-a-rdma-gid-index) NODE_A_RDMA_GID_INDEX="$2"; shift 2 ;;
    --node-b-rdma-gid-index) NODE_B_RDMA_GID_INDEX="$2"; shift 2 ;;
    --deploy-config) DEPLOY_CONFIG="$2"; shift 2 ;;
    --overlay-path) OVERLAY_PATH="$2"; shift 2 ;;
    --stage0-devices) STAGE0_DEVICES="$2"; shift 2 ;;
    --stage1-devices) STAGE1_DEVICES="$2"; shift 2 ;;
    --stage2-devices) STAGE2_DEVICES="$2"; shift 2 ;;
    --ssh-target) SSH_TARGET="$2"; shift 2 ;;
    --ssh-port) SSH_PORT="$2"; shift 2 ;;
    --ssh-option) SSH_OPTION+=("$2"); shift 2 ;;
    --remote-script) REMOTE_SCRIPT="$2"; shift 2 ;;
    --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
    --remote-start-delay) REMOTE_START_DELAY="$2"; shift 2 ;;
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
  DEPLOY_CONFIG="${REPO_ROOT}/vllm_omni/deploy/qwen3_omni_moe.yaml"
fi
if [[ -z "${MASTER_HOST}" ]]; then
  MASTER_HOST="${NODE_A_HOST}"
fi
[[ -f "${DEPLOY_CONFIG}" ]] || die "Deploy config not found: ${DEPLOY_CONFIG}"

if [[ "${ROLE}" == "both" ]]; then
  run_both_nodes
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
