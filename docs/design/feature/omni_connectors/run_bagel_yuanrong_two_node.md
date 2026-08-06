# Two-node BAGEL Yuanrong TransferEngine diagnostic runner

The runner at
[`tests/distributed/omni_connectors/run_bagel_yuanrong_two_node.py`](../../../../tests/distributed/omni_connectors/run_bagel_yuanrong_two_node.py)
starts a two-stage BAGEL deployment, runs three concurrent image requests,
and collects logs and runtime information for Yuanrong
`TransferEngineConnector` debugging.

It is intended for a validation machine, not as a production serving
launcher. Run it on the stage-0 machine.

## Prerequisites

Both nodes must have:

- the same vLLM-Omni checkout at the path resolved by the script;
- the BAGEL model available at the same path, or a path supplied with
  `--model`;
- a Python environment containing vLLM-Omni, PyYAML, and the Yuanrong binding;
- SSH connectivity from stage 0 to stage 1 without an interactive password;
- the stage-1 virtual environment available at
  `VLLM_OMNI_REMOTE_VENV` (default: `/app/vllm-omni/.venv`); and
- network reachability for the API/master ports and the configured RDMA
  interfaces.

The script uses `ssh -o BatchMode=yes`, so verify the connection first:

```bash
ssh -o BatchMode=yes <stage1-host> true
```

The remote checkout must contain this script at the same absolute path. If
the CUDA installation is not `/usr/local/cuda-13.0`, set
`VLLM_OMNI_REMOTE_CUDA_HOME` before running.

## Basic command

Replace the host, IP, model, and environment values with the validation
machine's values:

```bash
export VLLM_OMNI_REMOTE_VENV=/app/vllm-omni/.venv
export VLLM_OMNI_REMOTE_CUDA_HOME=/usr/local/cuda-13.0

RUN_ID="$(date +%Y%m%d-%H%M%S)"
python3 tests/distributed/omni_connectors/run_bagel_yuanrong_two_node.py \
  --connector yuanrong \
  --model /path/to/BAGEL-7B-MoT \
  --stage0-host stage0-hostname \
  --stage1-host stage1-hostname \
  --stage0-ip 192.168.1.10 \
  --stage1-ip 192.168.1.11 \
  --memory-pool-device cpu \
  --memory-pool-size 4294967296 \
  --vllm-logging-level DEBUG \
  --keep-artifacts \
  --run-dir "/tmp/vllm-omni-bagel-yuanrong-${RUN_ID}"
```

`--stage0-ip` and `--stage1-ip` are the addresses used for the transfer
connection. They are not necessarily the management or SSH addresses.
`--stage0-host` is used for recording the run; `--stage1-host` is the SSH
destination.

The default API port is `8000` and the default Omni master port is `8091`.
Change them with `--api-port` and `--omni-master-port` if either is occupied.
The connector configuration uses ZMQ port `50051`; Yuanrong RPC port remains
`auto`.

## Important options

| Option | Purpose |
| --- | --- |
| `--connector yuanrong` | Select `YuanrongTransferEngineConnector`. |
| `--rdma-device NAME` | Set the Mooncake-compatible RDMA HCA name; Yuanrong keeps its device endpoint on `auto`. |
| `--memory-pool-device cpu` | Use CPU memory for the Yuanrong pool. CUDA pools can be selected with `cuda` or `cuda:<id>` when supported by the binding. |
| `--memory-pool-size BYTES` | Size of the registered transfer pool on each node. |
| `--vllm-logging-level DEBUG` | Enable detailed vLLM-Omni logs on both stages. This is the recommended diagnostic setting. |
| `--run-dir PATH` | Use a unique local/remote artifact directory for each run. |
| `--keep-artifacts` | Keep the remote artifact directory after completion. Local copies are kept regardless. Failed runs are always retained. |
| `--skip-system-info` | Skip best-effort GPU, RDMA, network, process, and package snapshots. |
| `--startup-timeout SECONDS` | Maximum time to wait for the stage-0 API to become ready. |
| `--request-timeout SECONDS` | Timeout passed to the three image requests. |

For a quick smoke run, omit `--keep-artifacts`. For a PR investigation, use a
unique `--run-dir` and keep the remote artifacts.

## What the runner collects

The local run directory contains the following groups of artifacts:

- `run-metadata.json`: command-line arguments, commit, interpreter, status,
  and failure information;
- `bagel-stage0.yaml` and `remote-bagel-stage1.yaml`: generated deploy
  configurations;
- `stage0-command.txt`, `remote-stage1-command.txt`, and
  `bench-command.txt`: exact commands used to launch the stages and benchmark;
- `stage0.log`, `stage1.log`, `remote-controller.log`, and `controller.log`;
- `stage0-system-info.txt` and `remote-stage1-system-info.txt`: best-effort
  host, Python/package, GPU, RDMA, network, route, and process snapshots;
- `api-startup-*`, `api-after-bench-*`, and `api-final-*`: raw responses from
  `/metrics`, `/health`, and `/v1/models`; and
- `images/`: raw request responses, generated PNGs, and `bench-result.json`.

Before sharing the bundle, redact model paths, hostnames, IP addresses,
environment details, and any credentials that may be present in application
logs. Do not share raw logs blindly with a public issue or PR.

## Reading the transfer evidence

Start with these files:

1. `run-metadata.json` confirms the exact source and options used.
2. `stage0.log` and `stage1.log` contain the connector and stage lifecycle.
3. Search both logs for `[YR GET]`, `transfer_engine`, `cleanup`, `Pool`, and
   `ERROR`.
4. Compare the `[YR GET]` `query`, `alloc`, `read`, `copy`, `total`, and
   `MB/s` fields with `bench-result.json`.
5. Compare `api-startup-metrics.txt` with
   `api-after-bench-metrics.txt` for aggregate transfer counters.
6. Use the two system-info files to correlate the result with RDMA device,
   link, route, driver, and package versions.

The current connector's successful receive log is emitted after the
TransferEngine read and device synchronization. Therefore `read` is closer to
the actual read/synchronization portion, while `total` includes metadata and
post-read handling. A successful sender `put()` is not proof that the receiver
completed its read; correlate the sender and receiver logs by request key.

## Failure handling and cleanup

The controller kills only process groups recorded for this run. On failure it
fetches the remote log and artifacts before leaving the remote directory in
place. On a successful run, remote artifacts are purged unless
`--keep-artifacts` is set; the local diagnostic bundle remains.

If a previous run left a process behind, rerun with the same `--run-dir` only
when that directory belongs to this test. Prefer a new directory for a new
experiment. Do not manually kill unrelated vLLM processes by matching a broad
process name.

## Logging overhead and performance use

There are two different overhead sources:

1. The runner's system snapshots and API snapshots are outside the transfer
   hot path. They add startup/teardown time and a few HTTP requests, but do
   not run once per transferred buffer.
2. `VLLM_LOGGING_LEVEL=DEBUG` applies to the complete vLLM-Omni process, not
   only Yuanrong. Under high request rates it can produce synchronous log
   formatting and file/stderr I/O, contend on logging locks, and perturb CPU
   scheduling. It can therefore change end-to-end latency and throughput even
   though it does not change the RDMA operation itself.

In the current Yuanrong connector, the successful `[YR GET]` summary is at
`INFO`, while the successful `put()` path has no per-transfer success log.
`DEBUG` mainly exposes lower-level failures and other vLLM-Omni debug logs.
The summary is emitted after `batch_transfer_sync_read()` and synchronization,
so it does not inflate the measured `read` interval, but it still runs before
`get()` returns and can affect request completion latency.

Recommended procedure:

1. Run one diagnostic pass with `--vllm-logging-level DEBUG` to obtain the
   transfer evidence.
2. Run the same workload again with `--vllm-logging-level WARNING
   --skip-system-info` for a lower-logging comparison.
3. Compare several repeated runs, not a single run, using
   `images/bench-result.json`, API metrics, and the connector log summaries.

Do not use the DEBUG result as a production throughput number. For a future
connector observability PR, prefer bounded aggregate metrics and sampled or
explicitly enabled per-transfer logs. Avoid request IDs, pointer values, and
unbounded keys as Prometheus labels.
