# YuanrongTransferEngineConnector Observability Design

Status: Draft

## Motivation

`YuanrongTransferEngineConnector` currently reports initialization failures and
receiver-side transfer summaries, but it does not expose a complete view of a
cross-node transfer. In particular, a successful sender `put()`, the metadata
query over ZMQ, the TransferEngine read, and the sender-side cleanup are not
connected by one observable event sequence.

The goal of this design is to make a failed or slow transfer diagnosable from
the stage logs collected on both nodes, without logging payload contents or
adding per-request labels to production metrics.

## Transfer lifecycle

The connector has two paths that should be observable independently:

```text
Sender                                  Receiver
------                                  --------
put()                                   get()
  serialize object                        query metadata over ZMQ
  allocate sender pool                    allocate receiver pool
  copy tensor/bytes into pool              batch_transfer_sync_read()
  synchronize device                      synchronize receive device
  publish metadata                        cleanup sender buffer over ZMQ
                                          deserialize or return ManagedBuffer
```

The log event names should identify the phase that completed, rather than
implying that `put()` itself performed the network transfer. The actual
TransferEngine data movement occurs during receiver-side `get()`.

## Log levels

| Level | Intended content |
| --- | --- |
| `INFO` | Connector lifecycle, configuration summary, and one final summary per completed transfer when detailed transfer logging is explicitly enabled. |
| `DEBUG` | Per-phase timings, metadata query/cleanup exchanges, allocation details, and control-plane state transitions. |
| `WARNING` | Recoverable conditions such as metadata misses, cleanup not acknowledged, stale buffers, and pool pressure. |
| `ERROR` | Initialization, memory registration, device synchronization, or TransferEngine failures with the exception and phase. |

Per-transfer logs should not be emitted at `INFO` by default for high request
rates. The existing vLLM logger and `VLLM_LOGGING_LEVEL=DEBUG` should be used
for diagnostic runs rather than introducing a connector-specific environment
variable in the first version.

## Common event fields

Every transfer-related event should use the same fields where available:

```text
event=<event_name>
role=sender|receiver
from_stage=<stage>
to_stage=<stage>
key=<short stable request identifier>
protocol=rdma|ascend
device=<TransferEngine device>
pool_device=<memory pool device>
size_bytes=<payload size>
fast_path=true|false
```

The key should be a short, stable representation suitable for correlating the
two stage logs. Full payloads, serialized objects, and raw memory addresses
must not be logged at `INFO`. If pointer values are needed for a specific
debugging session, they should be restricted to `DEBUG`.

## Proposed events

### Sender events

```text
event=put_complete role=sender key=... size_bytes=... fast_path=...
serialize_ms=... alloc_ms=... copy_ms=... sync_ms=... total_ms=...
```

The sender event means that the payload was copied into the registered pool
and metadata was published locally. It must not be described as a completed
network transfer.

### Receiver events

```text
event=get_complete role=receiver key=... size_bytes=... fast_path=...
query_ms=... alloc_ms=... transfer_ms=... sync_ms=...
cleanup_ms=... copy_ms=... deserialize_ms=... total_ms=... throughput_mbps=...
```

`transfer_ms` must cover the `batch_transfer_sync_read()` call only. Device
synchronization and object deserialization should be measured separately.

### Control-plane events

```text
event=metadata_query_complete key=... found=true data_size=... query_ms=...
event=metadata_query_complete key=... found=false query_ms=...
event=sender_cleanup_complete key=... acknowledged=true cleanup_ms=...
```

These events are important because a sender-side `put()` success does not prove
that the receiver was able to query or read the payload.

### Failure events

Failures should include a stable `phase` or `error_type`, for example:

```text
event=transfer_failed key=... phase=metadata_query error_type=timeout
event=transfer_failed key=... phase=receive_pool error_type=exhausted
event=transfer_failed key=... phase=transfer_engine_read error_type=runtime_error
event=transfer_failed key=... phase=device_sync error_type=runtime_error
event=transfer_failed key=... phase=deserialize error_type=decode_error
```

The `timeouts` counter should only increase for actual timeout failures. Other
TransferEngine or deserialization exceptions should be counted separately.

## Metrics

Metrics should remain aggregated and compatible with the existing transfer
metrics model. Do not add `request_id`, full key, or pointer values as metric
labels because their cardinality is unbounded.

The first implementation should expose or reuse the following dimensions:

- payload size histogram;
- sender-side submit time;
- receiver-side total time;
- receiver-side TransferEngine read time;
- receiver-side device synchronization time;
- transfer count, bytes, and error count;
- error counters partitioned by phase;
- current and peak memory-pool usage.

Stage and replica are appropriate labels when they are already available from
the runtime. Protocol, device kind, and fast-path mode should be bounded label
values.

## Community comparison

The current vLLM-Omni connectors provide a useful local convention for this
design:

- `MooncakeTransferEngineConnector` logs initialization, memory-pool setup,
  and configuration at `INFO`, while its completed receive path reports
  `query`, `alloc`, `rdma`, `sync`, `copy`, `deserialize`, `total`, and
  throughput in one summary event.
- `MoriTransferEngineConnector` uses the same style for completed receives and
  keeps lower-level remote-engine registration at `DEBUG`.
- Connector `health()` methods expose aggregate counters and configuration
  fields rather than per-request state.

This is consistent with the proposed split: concise, phase-aware completion
summaries for a deliberate diagnostic run; lower-level control-plane details at
`DEBUG`; and bounded aggregate metrics for production monitoring. The Yuanrong
implementation should follow the existing `get_connector_logger()` path and
avoid introducing a connector-specific logging framework.

The repository's Prometheus metrics discussion also explicitly favors
pipeline/stage-level metrics and rejects per-request labels such as request ID.
It proposes histograms for latency and payload size, with bounded labels such as
`model_name` and `stage_id`. This supports keeping request correlation in logs
while keeping metrics cardinality bounded. See the
[`Prometheus metrics RFC`](https://github.com/vllm-project/vllm-omni/issues/3228).

Recent community PRs also make the validation environment part of the review
artifact: they state the exact test commands, commit/version, hardware,
workload, and before/after results instead of relying on log snippets alone.
The observability PR should follow that pattern; for example, see the
[`MiniCPM-o performance PR`](https://github.com/vllm-project/vllm-omni/pull/5385)
and the [`Mooncake fallback PR`](https://github.com/vllm-project/vllm-omni/pull/4142).

Reference implementations:

- [`MooncakeTransferEngineConnector`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/distributed/omni_connectors/connectors/mooncake_transfer_engine_connector.py)
- [`MoriTransferEngineConnector`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py)
- [`OmniTransferMetrics`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/metrics/transfer.py)

The main adjustment from the initial draft is therefore not to add more
always-on logs. It is to bring Yuanrong's sender path and control plane to the
same level of phase visibility already present in the other TransferEngine
connectors, while preserving the repository's bounded-metrics convention.

## Validation artifacts

A diagnostic run should collect, for both nodes:

1. the exact vLLM-Omni commit and Yuanrong binding version;
2. connector configuration with model paths and credentials redacted;
3. stage logs with timestamps and host/role information;
4. connector health snapshots before and after the run;
5. a successful transfer and at least one controlled failure;
6. payload size, transfer mode, hardware topology, and throughput summary.

The minimal command-line diagnostic mode is expected to use:

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

The setting must be applied on both stage processes. `--log-stats` can remain
enabled for vLLM-Omni request and stage statistics, but it is not a substitute
for connector debug logging.

The two-node BAGEL diagnostic runner now produces a local bundle containing:

- `run-metadata.json`, the exact arguments, commit, interpreter, and final
  status;
- `stage0.log`, the fetched `stage1.log`, both generated deploy YAML files, and
  the exact stage/benchmark commands;
- best-effort GPU, RDMA, network, process, Python-package, and selected
  environment snapshots from both nodes;
- API snapshots for `/metrics`, `/health`, and `/v1/models` before and after
  the benchmark; and
- the raw benchmark responses, generated images, and `bench-result.json`.

Use `--run-dir` to give each experiment a unique bundle path. Use
`--keep-artifacts` when the remote directory is also needed after a successful
run; the local bundle is retained in either case. System probing is
best-effort and can be disabled with `--skip-system-info` when a host's device
utilities are known to block or are unavailable.

## Proposed implementation sequence

1. Add consistent sender, receiver, metadata, cleanup, and failure events.
2. Split receiver timing into query, allocation, TransferEngine read,
   synchronization, cleanup, copy, and deserialization phases.
3. Correct error-phase counters and add bounded pool-usage statistics.
4. Add focused unit tests for event fields and failure classification using the
   existing fake TransferEngine.
5. Document one two-node BAGEL collection procedure and include sanitized
   success/failure results in the PR description.

The first PR should avoid changing transport semantics, adding a new logging
framework, or enabling verbose per-transfer logging unconditionally.
