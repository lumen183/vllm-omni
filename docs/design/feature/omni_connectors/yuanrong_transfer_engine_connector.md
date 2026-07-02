# YuanrongTransferEngineConnector

## When to Use

Use `YuanrongTransferEngineConnector` for high-performance peer-to-peer KV cache
transfer through Yuanrong TransferEngine. It is different from
`YuanrongConnector`:

- `YuanrongConnector` uses Yuanrong Datasystem as a distributed KV store and
  requires Datasystem workers plus etcd.
- `YuanrongTransferEngineConnector` uses Yuanrong TransferEngine directly. It
  registers local memory in each stage worker and lets the receiver pull data
  from the sender with TransferEngine.

The implementation currently lives under
`vllm_omni/platforms/npu/omni_connectors/` for compatibility with the original
Ascend integration. The connector is still registered with the generic
OmniConnector factory under the name `YuanrongTransferEngineConnector`.

For Ascend P2P transfer, the connector should use NPU memory as its transfer
pool. For CPU RDMA transfer, it uses a CPU host memory pool as a staging area:
GPU KV payloads are copied into the CPU pool before RDMA and copied/restored to
the target device after receive. This is the same staging model used by
Mooncake's CPU RDMA connector, not GPUDirect RDMA.

## Prerequisites

Install Yuanrong Datasystem/TransferEngine Python bindings in the runtime
environment. The packaged install is:

```bash
pip install openyuanrong-datasystem
```

To build from source, follow the upstream project instructions:

- Source repository: [openeuler/yuanrong-datasystem](https://atomgit.com/openeuler/yuanrong-datasystem)
- Installation guide: [Yuanrong Datasystem Linux installation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/installation/installation_linux.html)

The Ascend driver, CANN runtime, HCCL/HCCP, and NPU device network must also be
configured on every machine participating in the transfer.

## Check NPU Device IPv4

Yuanrong TransferEngine with `protocol: "ascend"` uses the NPU device network,
not only the host management IP. Each participating NPU must have an IPv4 HCCN
device IP.

Check the device IP, link, and health state:

```bash
for i in {0..7}; do
  echo "===== NPU $i ====="
  hccn_tool -i "$i" -ip -g
  hccn_tool -i "$i" -link -g
  hccn_tool -i "$i" -net_health -g
done
```

Expected output for each NPU should include an IPv4 address and healthy link:

```text
ipaddr:10.86.1.80
netmask:255.255.255.0
link status: UP
net health status: Success
```

If the output says `Get ipconf failed, because no ip was preset there!`, the NPU
device IP is not configured and Ascend P2P will fail. If the address is IPv6,
configure IPv4 for this connector path or validate that the underlying Yuanrong
TransferEngine build supports the IPv6 HCCN path before using it.

## Check Device-to-Device Connectivity

After collecting the target-stage NPU IPv4 addresses, ping them from each
source-stage NPU:

```bash
# Example: AR uses NPU 0..3 and DiT uses NPU 4..7.
# Replace the destination IPs with the values reported by hccn_tool -i 4..7 -ip -g.
for src in 0 1 2 3; do
  for dst_ip in 10.86.1.78 10.86.1.79 10.86.1.80 10.86.1.81; do
    echo "===== NPU $src -> $dst_ip ====="
    hccn_tool -i "$src" -ping -g address "$dst_ip" || true
  done
done
```

The expected result is `0.00% packet loss` for every required source/target pair.
This verifies basic HCCN IPv4 connectivity. TransferEngine can still fail later
if RDMA/QP/HCCL setup is broken, but failed ping means the device network must be
fixed first.

## YAML Configuration

### Ascend P2P

For Ascend P2P, configure the connector with `protocol: "ascend"` and an NPU
memory pool:

```yaml
runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
  connectors:
    yuanrong_te_connector:
      name: YuanrongTransferEngineConnector
      extra:
        host: "auto"
        zmq_port: 50051
        rpc_port: "auto"
        protocol: "ascend"
        device_name: "auto"
        memory_pool_size: 1073741824
        memory_pool_device: "npu"
  edges:
    - from: 0
      to: 1
      window_size: -1
```

Important fields:

| Parameter | Recommended Value | Notes |
|---|---|---|
| `host` | `"auto"` | Advertises a routable local host IP for ZMQ metadata exchange. |
| `zmq_port` | `50051` | Base ZMQ port. The runtime applies stage/rank offsets. |
| `rpc_port` | `"auto"` | Lets TransferEngine choose an available RPC port. |
| `protocol` | `"ascend"` | Uses the Ascend TransferEngine backend. |
| `device_name` | `"auto"` | Each TP worker resolves to its local logical NPU, for example `npu:0`, `npu:1`, etc. |
| `memory_pool_device` | `"npu"` | Required for the Ascend fast path. This registers NPU memory with TransferEngine. |
| `memory_pool_size` | `1073741824` | 1 GiB per worker is a practical starting point for KV-cache transfer validation. Increase if the transfer pool is exhausted. |

`memory_pool_device: "cpu"` is rejected for
`YuanrongTransferEngineConnector` with `protocol: "ascend"`. CPU pool registration
can lead to Ascend P2P failures such as HCCL RA init errors, QP timeouts, or
FFTS/SDMA runtime errors. `memory_pool_device: "cpu"` is common for
`MooncakeTransferEngineConnector`, but Yuanrong TE on Ascend only supports
`"npu"`.

### CPU RDMA

For CPU RDMA, configure the connector with `protocol: "rdma"` and a CPU memory
pool:

```yaml
runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
  connectors:
    yuanrong_cpu_rdma_connector:
      name: YuanrongTransferEngineConnector
      extra:
        host: "auto"
        zmq_port: 50051
        rpc_port: "auto"
        protocol: "rdma"
        device_name: "auto"
        memory_pool_size: 4294967296
        memory_pool_device: "cpu"
  edges:
    - from: 0
      to: 1
      window_size: -1
```

Important CPU RDMA fields:

| Parameter | Recommended Value | Notes |
|---|---|---|
| `protocol` | `"rdma"` | Uses Yuanrong TransferEngine CPU RDMA backend. |
| `device_name` | `"auto"` or `"cpu:*"` | `"auto"` resolves to `cpu:*`. |
| `memory_pool_device` | `"cpu"` | Required. The registered RDMA memory is host memory. |
| `memory_pool_size` | `4294967296` | 4 GiB per worker is a practical starting point for CPU staging. Increase if KV payloads exceed the pool. |

CPU RDMA data flow:

```text
sender GPU KV -> CPU pool -> RDMA -> receiver CPU pool -> receiver GPU/use site
```

The CPU pool is explicitly allocated by this connector and registered with
TransferEngine via `register_memory(base_ptr, memory_pool_size)`. TransferEngine
then pins/registers the host range with the CPU RDMA backend. The connector
returns `ManagedBuffer` objects for fast-path payloads; vLLM Omni consumes those
buffers and moves reconstructed tensors to the target device when required.

## 4-Card AR-to-DiT TP Example

For AR 4-way TP sending KV cache to DiT 4-way TP, set the stage TP topology in
both stage `omni_kv_config` blocks:

```yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0,1,2,3"
    engine_args:
      tensor_parallel_size: 4
      omni_kv_config:
        need_send_cache: true
        rank_mapping:
          from_tp: 4
          to_tp: 4
    output_connectors:
      to_stage_1: yuanrong_te_connector

  - stage_id: 1
    runtime:
      devices: "4,5,6,7"
    engine_args:
      parallel_config:
        tensor_parallel_size: 4
      omni_kv_config:
        need_recv_cache: true
        rank_mapping:
          from_tp: 4
          to_tp: 4
    input_connectors:
      from_stage_0: yuanrong_te_connector
```

In this topology each DiT rank receives the corresponding AR rank shard:

```text
DiT rank 0 <- AR rank 0
DiT rank 1 <- AR rank 1
DiT rank 2 <- AR rank 2
DiT rank 3 <- AR rank 3
```

## Validate a Successful Run

Search the inference log for Yuanrong TE receive-side messages:

```bash
grep -E "\[YR GET\]|Successfully received KV cache|transfer_engine get failed" inference.log
```

A successful Ascend fast-path transfer contains lines like:

```text
[YR GET] ... fast_path, zero-copy
Successfully received KV cache ... across 1 key(s)
Applied CFG KV caches
[KV Reuse] ...
```

Sender-side `KV transfer OK` only means that the sender registered the payload
and metadata. The actual P2P data transfer happens on the receiver side during
`[YR GET]`.

## Troubleshooting

| Symptom | Likely Cause | Action |
|---|---|---|
| `IPv4 device IP not found` | NPU HCCN IPv4 is missing. | Run `hccn_tool -i <id> -ip -g` and configure IPv4 for every participating NPU. |
| `P2PCommInitRootInfo failed`, `ra init failed`, `RA_QP_STATUS_TIMEOUT` | Ascend P2P/RDMA/QP setup failed. | First verify device IPv4, link health, and device-to-device ping. Then test a minimal Yuanrong TE pair outside vLLM. |
| `fftsplus sdma error`, `context is abort` | Runtime SDMA/FFTS failure, often after an invalid or failed P2P transfer. | Use `memory_pool_device: "npu"` for Ascend TE, reduce `memory_pool_size` if OOM occurs, and isolate the failing NPU pair with a Yuanrong TE demo. |
| Sender logs `KV transfer OK` but receiver never logs `[YR GET]` | Sender `put()` succeeded, receiver P2P read failed or timed out. | Check receiver logs and search for `transfer_engine get failed`. |
| Pool allocation fails or OOM | Transfer pool is too small or too large for available NPU memory. | Start with 1 GiB (`1073741824`), reduce to 512 MiB for validation, or increase for larger KV payloads. |
