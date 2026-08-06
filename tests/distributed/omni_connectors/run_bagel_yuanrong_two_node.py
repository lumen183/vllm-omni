#!/usr/bin/env python3
"""Two-node BAGEL TransferEngineConnector end-to-end smoke benchmark.

Run this script on the stage-0 machine. It starts stage 1 through SSH, writes
per-node deploy YAML files under the diagnostic run directory, submits three
concurrent image requests, captures stage/API/system diagnostics, then stops
only the process groups created by this run.

See docs/design/feature/omni_connectors/run_bagel_yuanrong_two_node.md for
prerequisites, command examples, collected artifacts, and performance notes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


MODEL_PATH = "/path/to/BAGEL-7B-MoT"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
BENCH_SCRIPT = REPO_ROOT / "tests/distributed/omni_connectors/bagel_te_bench.py"
DEFAULT_RUN_DIR = Path("/tmp/vllm-omni-bagel-cross-node")
RUN_DIR = DEFAULT_RUN_DIR
# SSH runs a non-interactive shell, so the remote virtual environment must be
# activated explicitly. Override this when the remote checkout uses another
# location, for example: VLLM_OMNI_REMOTE_VENV=/opt/vllm-omni/.venv.
REMOTE_VENV = Path(os.environ.get("VLLM_OMNI_REMOTE_VENV", "/app/vllm-omni/.venv"))
REMOTE_CUDA_HOME = os.environ.get("VLLM_OMNI_REMOTE_CUDA_HOME", "/usr/local/cuda-13.0")
ERROR_PATTERN = re.compile(r"Traceback|\bERROR\b|\bException\b|\bRuntimeError\b", re.IGNORECASE)
DEFAULT_LOG_LEVEL = os.environ.get("VLLM_OMNI_DIAGNOSTIC_LOG_LEVEL", "DEBUG").upper()
YR_GET_PATTERN = re.compile(
    r"\[YR GET\]\s+(?P<key>[^:]+):\s+"
    r"(?:size_bytes=(?P<size_bytes>\d+),\s+)?"
    r"query=(?P<query_ms>[\d.]+)ms,\s+"
    r"alloc=(?P<alloc_ms>[\d.]+)ms,\s+"
    r"read=(?P<read_ms>[\d.]+)ms,\s+"
    r"(?:copy=(?P<copy_ms>[\d.]+)ms,\s+)?"
    r"total=(?P<total_ms>[\d.]+)ms,\s+"
    r"(?P<mbps>[\d.]+)\s+MB/s(?:\s+\((?P<flags>[^)]*)\))?"
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_command(path: Path, command: list[str]) -> None:
    path.write_text(shlex.join(command) + "\n", encoding="utf-8")


def _run_capture(command: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def _collect_system_info(path: Path, args: argparse.Namespace) -> None:
    """Collect best-effort host and runtime information without failing the test."""
    package_probe = (
        "import importlib.metadata as m, importlib.util, sys; "
        "print('python:', sys.version.replace('\\n', ' ')); "
        "names = ('vllm', 'vllm-omni', 'torch', 'pyzmq', 'PyYAML', 'openyuanrong-datasystem'); "
        "installed = {d.metadata.get('Name', '').lower(): d.version for d in m.distributions()}; "
        "[print(name + ':', installed.get(name.lower(), 'not-installed')) for name in names]; "
        "[print('module ' + name + ':', bool(importlib.util.find_spec(name))) for name in ('yr', 'torch', 'vllm', 'vllm_omni')]"
    )
    commands: list[tuple[str, list[str]]] = [
        ("timestamp", ["date", "--iso-8601=seconds"]),
        ("hostname", ["hostname", "--fqdn"]),
        ("uname", ["uname", "-a"]),
        ("python-and-packages", [sys.executable, "-c", package_probe]),
        ("git-commit", ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        ("git-status", ["git", "-C", str(REPO_ROOT), "status", "--short"]),
        ("gpu-list", ["nvidia-smi", "-L"]),
        (
            "gpu-summary",
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,pci.bus_id",
                "--format=csv",
            ],
        ),
        ("rdma-devices", ["ibdev2netdev"]),
        ("rdma-status", ["ibstat"]),
        ("network-addresses", ["ip", "-br", "addr"]),
        ("network-routes", ["ip", "route"]),
        ("npu-smi", ["npu-smi", "info"]),
        ("hccn-tool-ip", ["hccn_tool", "-i", "0", "-ip", "-g"]),
        ("hccn-tool-link", ["hccn_tool", "-i", "0", "-link", "-g"]),
        ("hccn-tool-health", ["hccn_tool", "-i", "0", "-net_health", "-g"]),
        ("processes", ["ps", "-eo", "pid,ppid,pgid,etime,cmd"]),
    ]
    selected_env = (
        "CUDA_HOME",
        "CUDA_PATH",
        "VLLM_LOGGING_LEVEL",
        "VLLM_OMNI_DIAGNOSTIC_LOG_LEVEL",
        "VLLM_OMNI_REMOTE_VENV",
        "VLLM_OMNI_REMOTE_CUDA_HOME",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as info_file:
        info_file.write("# Diagnostic system information (best effort)\n")
        info_file.write(f"# run_dir={RUN_DIR}\n")
        info_file.write(f"# connector={args.connector}\n")
        info_file.write(f"# logging_level={args.vllm_logging_level}\n\n")
        info_file.write("[selected-environment]\n")
        for name in selected_env:
            value = (
                args.vllm_logging_level
                if name == "VLLM_LOGGING_LEVEL"
                else os.environ.get(name, "<unset>")
            )
            info_file.write(f"{name}={value}\n")
        info_file.write("\n")
        for name, command in commands:
            info_file.write(f"[{name}]\n$ {shlex.join(command)}\n")
            try:
                result = _run_capture(command)
                info_file.write(f"exit_code={result.returncode}\n")
                if result.stdout:
                    info_file.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        info_file.write("\n")
                if result.stderr:
                    info_file.write("[stderr]\n")
                    info_file.write(result.stderr)
                    if not result.stderr.endswith("\n"):
                        info_file.write("\n")
            except (OSError, subprocess.TimeoutExpired) as exc:
                info_file.write(f"unavailable={exc}\n")
            info_file.write("\n")


def _runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = args.vllm_logging_level
    return env


def _write_run_metadata(args: argparse.Namespace, *, status: str, error: str | None = None) -> None:
    git_result = _run_capture(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    metadata = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "args": vars(args),
        "script": str(SCRIPT_PATH),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else "unknown",
        "local_python": sys.executable,
        "local_pid": os.getpid(),
        "error": error,
    }
    _write_json(RUN_DIR / "run-metadata.json", metadata)


def _parse_yr_get_log(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    for match in YR_GET_PATTERN.finditer(text):
        size_bytes = match.group("size_bytes")
        total_ms = float(match.group("total_ms"))
        mbps = float(match.group("mbps"))
        record: dict[str, Any] = {
            "key": match.group("key"),
            "size_bytes": int(size_bytes) if size_bytes is not None else None,
            "query_ms": float(match.group("query_ms")),
            "alloc_ms": float(match.group("alloc_ms")),
            "read_ms": float(match.group("read_ms")),
            "copy_ms": float(match.group("copy_ms")) if match.group("copy_ms") else None,
            "total_ms": total_ms,
            "mbps": mbps,
            "fast_path": "fast_path" in (match.group("flags") or ""),
        }
        # Old logs did not include size_bytes. Keep the result useful, but
        # explicitly mark the value as estimated because MB/s and total_ms
        # are rounded to one decimal place in the log.
        if record["size_bytes"] is None:
            record["size_bytes_estimate"] = mbps * (total_ms / 1000.0) * 1024**2
        records.append(record)
    return records


def _find_metric_value(text: str, metric_name: str, required_labels: tuple[str, ...] = ()) -> float | None:
    value_pattern = re.compile(r"\s([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
    for line in text.splitlines():
        if not line.startswith(metric_name):
            continue
        if any(label not in line for label in required_labels):
            continue
        match = value_pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _format_stat(values: list[float], unit: str = "ms") -> str:
    if not values:
        return "n/a"
    return (
        f"avg={statistics.fmean(values):.2f}{unit}, "
        f"min={min(values):.2f}{unit}, max={max(values):.2f}{unit}"
    )


def _write_transfer_summary(args: argparse.Namespace, *, success: bool, failure: str | None) -> None:
    """Write a concise PR-ready summary from logs and Prometheus snapshots."""
    records = _parse_yr_get_log(RUN_DIR / "stage1.log")
    bench = _load_json(RUN_DIR / "images" / "bench-result.json")
    metrics_path = RUN_DIR / "api-after-bench-metrics.txt"
    try:
        metrics_text = metrics_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        metrics_text = ""

    sizes_are_exact = bool(records) and all(record["size_bytes"] is not None for record in records)
    size_values = [
        float(record["size_bytes"])
        if record["size_bytes"] is not None
        else float(record["size_bytes_estimate"])
        for record in records
    ]
    total_bytes = sum(size_values)
    total_read_ms = sum(float(record["read_ms"]) for record in records)
    total_total_ms = sum(float(record["total_ms"]) for record in records)
    total_query_ms = [float(record["query_ms"]) for record in records]
    total_alloc_ms = [float(record["alloc_ms"]) for record in records]
    total_copy_ms = [float(record["copy_ms"]) for record in records if record["copy_ms"] is not None]
    total_overhead_ms = [float(record["total_ms"]) - float(record["read_ms"]) for record in records]
    fast_path_count = sum(1 for record in records if record["fast_path"])

    transfer_count = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_size_bytes_count",
        ('from_stage="0"', 'to_stage="1"'),
    )
    transfer_size_sum = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_size_bytes_sum",
        ('from_stage="0"', 'to_stage="1"'),
    )
    tx_count = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_tx_s_count",
        ('from_stage="0"', 'to_stage="1"'),
    )
    tx_sum_s = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_tx_s_sum",
        ('from_stage="0"', 'to_stage="1"'),
    )
    rx_count = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_rx_s_count",
        ('from_stage="0"', 'to_stage="1"'),
    )
    in_flight_count = _find_metric_value(
        metrics_text,
        "vllm_omni:transfer_in_flight_s_count",
        ('from_stage="0"', 'to_stage="1"'),
    )
    http_count = _find_metric_value(
        metrics_text,
        "http_requests_total",
        ('handler="/v1/chat/completions"', 'status="2xx"'),
    )
    http_duration_sum_s = _find_metric_value(
        metrics_text,
        "http_request_duration_seconds_sum",
        ('handler="/v1/chat/completions"', 'method="POST"'),
    )

    lines = [
        "# Yuanrong TransferEngine benchmark summary",
        "",
        f"- Result: **{'PASS' if success else 'FAIL'}**",
        f"- Connector: `{args.connector}`",
        f"- Model: `{args.model}`",
        f"- Edge: stage `{args.stage0_host}` (`{args.stage0_ip}`) -> stage `{args.stage1_host}` (`{args.stage1_ip}`)",
        f"- Source commit: see `run-metadata.json`",
    ]
    if failure:
        lines.append(f"- Failure: `{failure}`")

    lines.extend(
        [
            "",
            "## Request result",
            "",
            f"- HTTP chat requests: `{int(http_count) if http_count is not None else bench.get('successful_requests', 'n/a')}` succeeded",
            f"- Benchmark requests: `{bench.get('successful_requests', 'n/a')}/{bench.get('request_count', 'n/a')}`",
            f"- Benchmark wall time: `{bench.get('wall_time_seconds', 'n/a')} s`",
            f"- Image throughput: `{bench.get('image_throughput_per_second', 'n/a')} requests/s`",
            f"- HTTP chat duration: `{http_duration_sum_s:.3f} s` total" if http_duration_sum_s is not None else "- HTTP chat duration: `n/a`",
            "",
            "## Connector receive result",
            "",
            f"- Connector receives observed: `{len(records)}`",
            f"- Payload transferred: `{total_bytes / 1024**2:.3f} MiB` ({'exact' if sizes_are_exact else 'estimated from rounded log values'})",
            f"- Fast path / zero-copy: `{fast_path_count}/{len(records)}`",
            f"- Query: `{_format_stat(total_query_ms)}`",
            f"- Allocation: `{_format_stat(total_alloc_ms)}`",
            f"- Transfer read + device sync: `{_format_stat([float(record['read_ms']) for record in records])}`",
            f"- Copy to host/deserialize: `{_format_stat(total_copy_ms) if total_copy_ms else 'n/a (fast path)'}`",
            f"- Post-read overhead (`total - read`): `{_format_stat(total_overhead_ms)}`",
            f"- Total connector receive time: `{total_total_ms:.2f} ms`",
            f"- Effective end-to-end throughput: `{total_bytes / 1024**2 / (total_total_ms / 1000.0):.2f} MiB/s`" if total_total_ms > 0 else "- Effective end-to-end throughput: `n/a`",
            f"- Read-path throughput: `{total_bytes / 1024**2 / (total_read_ms / 1000.0):.2f} MiB/s`" if total_read_ms > 0 else "- Read-path throughput: `n/a`",
            "",
            "## Per-transfer details",
            "",
            "| # | Size | Query | Alloc | Read | Total | Throughput | Path |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for index, record in enumerate(records, start=1):
        size = record.get("size_bytes")
        size_text = f"{size / 1024**2:.3f} MiB" if size is not None else f"~{record['size_bytes_estimate'] / 1024**2:.3f} MiB"
        lines.append(
            f"| {index} | {size_text} | {record['query_ms']:.1f} ms | {record['alloc_ms']:.1f} ms | "
            f"{record['read_ms']:.1f} ms | {record['total_ms']:.1f} ms | {record['mbps']:.1f} MB/s | "
            f"{'fast/zero-copy' if record['fast_path'] else 'copy'} |"
        )

    lines.extend(
        [
            "",
            "## Prometheus transfer instrumentation",
            "",
            f"- `transfer_size_bytes`: count=`{transfer_count if transfer_count is not None else 'missing'}`, sum=`{transfer_size_sum if transfer_size_sum is not None else 'missing'}` bytes",
            f"- `transfer_tx_s`: count=`{tx_count if tx_count is not None else 'missing'}`, sum=`{tx_sum_s * 1000:.3f} ms`" if tx_sum_s is not None else "- `transfer_tx_s`: missing",
            f"- `transfer_rx_s`: count=`{rx_count if rx_count is not None else 'missing'}`",
            f"- `transfer_in_flight_s`: count=`{in_flight_count if in_flight_count is not None else 'missing'}`",
        ]
    )
    if records and transfer_size_sum == 0:
        lines.append("- Note: connector logs report non-zero payloads, but the generic `transfer_size_bytes` emitter recorded zero; do not use that Prometheus sum as the payload size.")
    if records and rx_count is None:
        lines.append("- Note: RX/in-flight histogram samples were not emitted for this run; use `[YR GET]` read timing until the receive-side metrics hook is fixed.")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "- Connector log: `stage1.log`",
            "- Request result: `images/bench-result.json`",
            "- Prometheus snapshot: `api-after-bench-metrics.txt`",
        ]
    )
    (RUN_DIR / "transfer-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _memory_pool_device_arg(value: str) -> str:
    value = value.strip().lower()
    if value == "cpu" or value == "cuda" or re.fullmatch(r"cuda:\d+", value):
        return value
    raise argparse.ArgumentTypeError("must be cpu, cuda, or cuda:<device_id>")


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if RUN_DIR.exists():
        with (RUN_DIR / "controller.log").open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")


def _record_path(stage: int) -> Path:
    return RUN_DIR / f"stage{stage}.process.json"


def _kill_process_group(record_path: Path) -> None:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        pgid = int(record["pgid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _write_record(stage: int, process: subprocess.Popen[Any]) -> None:
    _record_path(stage).write_text(
        json.dumps({"pid": process.pid, "pgid": os.getpgid(process.pid)}) + "\n",
        encoding="utf-8",
    )


def _load_base_config() -> dict[str, Any]:
    config_path = REPO_ROOT / "vllm_omni/deploy/bagel.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid deploy config: {config_path}")
    return config


def _connector_config(
    connector: str,
    host: str,
    rdma_device: str,
    memory_pool_device: str,
    memory_pool_size: int,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "host": host,
        "zmq_port": 50051,
        "protocol": "rdma",
        # Yuanrong's RDMA device_name is a CPU/CUDA endpoint (not an HCA
        # name).  Leave it on auto so it follows memory_pool_device.  The
        # --rdma-device argument is retained for Mooncake compatibility.
        "device_name": rdma_device if connector == "mooncake" else "auto",
        "memory_pool_size": memory_pool_size,
        "memory_pool_device": memory_pool_device,
    }
    if connector == "mooncake":
        return {"name": "MooncakeTransferEngineConnector", "extra": extra}
    extra["rpc_port"] = "auto"
    return {"name": "YuanrongTransferEngineConnector", "extra": extra}


def _write_deploy_yaml(
    stage: int,
    connector: str,
    local_ip: str,
    rdma_device: str,
    memory_pool_device: str,
    memory_pool_size: int,
) -> Path:
    config = copy.deepcopy(_load_base_config())
    stages = config.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("bagel.yaml does not contain a stages list")
    stage_by_id = {item.get("stage_id"): item for item in stages if isinstance(item, dict)}
    if 0 not in stage_by_id or 1 not in stage_by_id:
        raise RuntimeError("bagel.yaml must contain stages 0 and 1")

    connector_name = "transfer_engine_connector"
    connectors = config.setdefault("connectors", {})
    if not isinstance(connectors, dict):
        raise RuntimeError("bagel.yaml connectors must be a mapping")
    connectors[connector_name] = _connector_config(
        connector, local_ip, rdma_device, memory_pool_device, memory_pool_size
    )
    stage_by_id[0]["output_connectors"] = {"to_stage_1": connector_name}
    stage_by_id[1]["input_connectors"] = {"from_stage_0": connector_name}

    path = RUN_DIR / f"bagel-stage{stage}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(f"Required local port {port} is unavailable: {exc}") from exc


def _append_command_output(log_path: Path, result: subprocess.CompletedProcess[str]) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        if result.stdout:
            log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)


def _ssh(host: str, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    # `ssh host command` does not source the user's interactive shell setup.
    # Use bash explicitly and set CUDA variables so that activation also
    # affects subprocesses started by the remote Python process.
    cuda_home = shlex.quote(REMOTE_CUDA_HOME)
    venv_activate = shlex.quote(str(REMOTE_VENV / "bin/activate"))
    remote_command = (
        # CUDA is often configured only in ~/.bashrc on compute nodes.
        'source ~/.bashrc >/dev/null 2>&1 || true; '
        f"cuda_home=\"${{CUDA_HOME:-{cuda_home}}}\"; "
        'if [ ! -d "$cuda_home" ] && [ -d /usr/local/cuda ]; then '
        'cuda_home=/usr/local/cuda; fi; '
        'export CUDA_HOME="$cuda_home" CUDA_PATH="$cuda_home"; '
        'export PATH="$cuda_home/bin:$PATH"; '
        'export LD_LIBRARY_PATH="$cuda_home/lib64:$cuda_home/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
        f"source {venv_activate} && exec {shlex.join(command)}"
    )
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, shlex.join(["bash", "-lc", remote_command])],
        text=True,
        capture_output=True,
        check=check,
    )


def _remote_script_command(mode: str, args: argparse.Namespace) -> list[str]:
    return [
        # `_ssh()` activates the remote venv before this command runs.
        "env",
        f"VLLM_LOGGING_LEVEL={args.vllm_logging_level}",
        "python",
        str(SCRIPT_PATH),
        mode,
        "--stage0-host",
        args.stage0_host,
        "--stage1-host",
        args.stage1_host,
        "--stage0-ip",
        args.stage0_ip,
        "--stage1-ip",
        args.stage1_ip,
        "--model",
        args.model,
        "--connector",
        args.connector,
        "--api-port",
        str(args.api_port),
        "--omni-master-port",
        str(args.omni_master_port),
        "--rdma-device",
        args.rdma_device,
        "--memory-pool-device",
        args.memory_pool_device,
        "--memory-pool-size",
        str(args.memory_pool_size),
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--run-dir",
        str(args.run_dir),
        "--vllm-logging-level",
        args.vllm_logging_level,
    ]


def _remote_alive(args: argparse.Namespace) -> bool:
    result = _ssh(args.stage1_host, _remote_script_command("--remote-status", args), check=False)
    return result.returncode == 0


def _remote_has_startup_error(args: argparse.Namespace) -> bool:
    command = ["sh", "-lc", f"test -f {shlex.quote(str(RUN_DIR / 'stage1.log'))} && grep -Eqi {shlex.quote(ERROR_PATTERN.pattern)} {shlex.quote(str(RUN_DIR / 'stage1.log'))}"]
    result = _ssh(args.stage1_host, command, check=False)
    return result.returncode == 0


def _has_startup_error(log_path: Path) -> bool:
    try:
        return ERROR_PATTERN.search(log_path.read_text(encoding="utf-8", errors="replace")) is not None
    except OSError:
        return False


def _api_ready(host: str, port: int) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{host}:{port}/health", timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _capture_api_snapshot(args: argparse.Namespace, label: str) -> None:
    """Save observability endpoints without making endpoint availability fatal."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoints = {
        "metrics": f"http://{args.stage0_ip}:{args.api_port}/metrics",
        "health": f"http://{args.stage0_ip}:{args.api_port}/health",
        "models": f"http://{args.stage0_ip}:{args.api_port}/v1/models",
    }
    for name, url in endpoints.items():
        path = RUN_DIR / f"api-{label}-{name}.txt"
        try:
            with opener.open(url, timeout=10) as response:
                body = response.read().decode("utf-8", errors="replace")
                path.write_text(
                    f"url={url}\nstatus={response.status}\n"
                    f"content_type={response.headers.get('Content-Type', '')}\n\n{body}",
                    encoding="utf-8",
                )
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            path.write_text(f"url={url}\nunavailable={exc}\n", encoding="utf-8")


def _start_stage0(args: argparse.Namespace) -> subprocess.Popen[Any]:
    deploy_yaml = _write_deploy_yaml(
        0,
        args.connector,
        args.stage0_ip,
        args.rdma_device,
        args.memory_pool_device,
        args.memory_pool_size,
    )
    command = [
        "vllm",
        "serve",
        args.model,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.api_port),
        "--stage-id",
        "0",
        "--omni-master-address",
        args.stage0_ip,
        "--omni-master-port",
        str(args.omni_master_port),
        "--deploy-config",
        str(deploy_yaml),
        "--log-stats",
    ]
    _write_command(RUN_DIR / "stage0-command.txt", command)
    log_file = (RUN_DIR / "stage0.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=_runtime_env(args),
    )
    log_file.close()
    _write_record(0, process)
    return process


def _start_stage1_remote(args: argparse.Namespace) -> None:
    command = _remote_script_command("--remote-stage1", args)
    result = _ssh(args.stage1_host, command, check=False)
    _append_command_output(RUN_DIR / "controller.log", result)
    if result.returncode:
        raise RuntimeError(f"Failed to launch stage 1 over SSH (exit {result.returncode})")


def _wait_for_startup(stage0: subprocess.Popen[Any], args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.startup_timeout
    stage0_log = RUN_DIR / "stage0.log"
    startup_log_warning_reported = False
    while time.monotonic() < deadline:
        if stage0.poll() is not None:
            raise RuntimeError(f"Stage 0 exited during startup with status {stage0.returncode}")
        if not _remote_alive(args):
            raise RuntimeError("Stage 1 exited during startup")
        # Log scanners are only diagnostic: vLLM and CUDA dependencies can
        # emit transient ERROR/Traceback text while a process is still
        # starting. Process liveness and the health endpoint are authoritative.
        if not startup_log_warning_reported and (
            _has_startup_error(stage0_log) or _remote_has_startup_error(args)
        ):
            _log("Startup log contains an error-like line; continuing to wait because both stages are alive")
            startup_log_warning_reported = True
        if _api_ready(args.stage0_ip, args.api_port):
            _log(f"API is ready on {args.stage0_ip}:{args.api_port}")
            return
        time.sleep(1)
    raise RuntimeError(f"API did not become ready within {args.startup_timeout}s")


def _run_bench(args: argparse.Namespace) -> None:
    output_dir = RUN_DIR / "images"
    command = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--host",
        args.stage0_ip,
        "--port",
        str(args.api_port),
        "--model",
        args.model,
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(args.request_timeout),
    ]
    with (RUN_DIR / "bench.log").open("w", encoding="utf-8") as log_file:
        _write_command(RUN_DIR / "bench-command.txt", command)
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Bench wrapper failed with status {result.returncode}")


def _fetch_remote_log(args: argparse.Namespace) -> None:
    result = _ssh(args.stage1_host, ["cat", str(RUN_DIR / "stage1.log")], check=False)
    _append_command_output(RUN_DIR / "stage1.log", result)


def _fetch_remote_artifact(args: argparse.Namespace, filename: str) -> None:
    remote_path = RUN_DIR / filename
    local_name = f"remote-{filename}"
    result = _ssh(args.stage1_host, ["cat", str(remote_path)], check=False)
    if result.returncode == 0:
        (RUN_DIR / local_name).write_text(result.stdout, encoding="utf-8")
    else:
        _append_command_output(RUN_DIR / "controller.log", result)
        _log(f"Could not fetch remote artifact {remote_path}")


def _fetch_remote_artifacts(args: argparse.Namespace) -> None:
    # Keep local copies even when the remote run directory is purged after a
    # successful run. These files make the diagnostic bundle self-contained.
    for filename in (
        "bagel-stage1.yaml",
        "stage1-system-info.txt",
        "stage1.process.json",
        "controller.log",
        "stage1-command.txt",
    ):
        _fetch_remote_artifact(args, filename)


def _cleanup_remote(args: argparse.Namespace, *, purge: bool) -> None:
    result = _ssh(args.stage1_host, _remote_script_command("--remote-cleanup", args), check=False)
    _append_command_output(RUN_DIR / "controller.log", result)
    if purge:
        result = _ssh(args.stage1_host, ["rm", "-rf", str(RUN_DIR)], check=False)
        _append_command_output(RUN_DIR / "controller.log", result)


def _remote_stage1(args: argparse.Namespace) -> int:
    _kill_process_group(_record_path(1))
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    deploy_yaml = _write_deploy_yaml(
        1,
        args.connector,
        args.stage1_ip,
        args.rdma_device,
        args.memory_pool_device,
        args.memory_pool_size,
    )
    command = [
        "vllm",
        "serve",
        args.model,
        "--omni",
        "--stage-id",
        "1",
        "--headless",
        "--omni-master-address",
        args.stage0_ip,
        "--omni-master-port",
        str(args.omni_master_port),
        "--deploy-config",
        str(deploy_yaml),
        "--log-stats",
    ]
    _write_command(RUN_DIR / "stage1-command.txt", command)
    log_file = (RUN_DIR / "stage1.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=_runtime_env(args),
    )
    log_file.close()
    _write_record(1, process)
    _log(f"Started stage 1 with pid {process.pid}")
    return 0


def _remote_status() -> int:
    try:
        record = json.loads(_record_path(1).read_text(encoding="utf-8"))
        os.kill(int(record["pid"]), 0)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return 1
    return 0


def _remote_system_info(args: argparse.Namespace) -> int:
    _collect_system_info(RUN_DIR / "stage1-system-info.txt", args)
    return 0


def _prepare_local_run() -> None:
    _kill_process_group(_record_path(0))
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _run_controller(args: argparse.Namespace) -> int:
    _prepare_local_run()
    controller_log = RUN_DIR / "controller.log"
    controller_log.touch()
    _write_run_metadata(args, status="running")
    stage0: subprocess.Popen[Any] | None = None
    success = False
    failure: str | None = None
    try:
        _assert_port_available(args.api_port)
        _assert_port_available(args.omni_master_port)
        _cleanup_remote(args, purge=False)
        _log(f"Artifacts: {RUN_DIR}")
        _log(f"Stage 0 host: {args.stage0_host}; Stage 1 SSH host: {args.stage1_host}")
        _log(f"Connector: {args.connector}; VLLM_LOGGING_LEVEL={args.vllm_logging_level}")
        _log(f"Remote artifacts will be {'kept' if args.keep_artifacts else 'purged after local fetch'}")
        if not args.skip_system_info:
            _collect_system_info(RUN_DIR / "stage0-system-info.txt", args)
            _log("Collecting remote system information")
            result = _ssh(
                args.stage1_host,
                _remote_script_command("--remote-system-info", args),
                check=False,
            )
            _append_command_output(controller_log, result)
            if result.returncode:
                _log("Remote system information collection failed; continuing")
        stage0 = _start_stage0(args)
        _start_stage1_remote(args)
        _wait_for_startup(stage0, args)
        _capture_api_snapshot(args, "startup")
        _run_bench(args)
        _capture_api_snapshot(args, "after-bench")
        if stage0.poll() is not None or not _remote_alive(args):
            raise RuntimeError("A vLLM stage exited before benchmark completion")
        success = True
        _log("BAGEL TransferEngineConnector benchmark completed successfully")
        return 0
    except Exception as exc:
        failure = str(exc)
        _log(f"FAILED: {exc}")
        with controller_log.open("a", encoding="utf-8") as log_file:
            log_file.write(traceback.format_exc())
        return 1
    finally:
        _capture_api_snapshot(args, "final")
        _kill_process_group(_record_path(0))
        _fetch_remote_log(args)
        _fetch_remote_artifacts(args)
        try:
            _write_transfer_summary(args, success=success, failure=failure)
            _log(f"Transfer summary: {RUN_DIR / 'transfer-summary.md'}")
        except Exception as summary_exc:
            _log(f"Could not write transfer summary: {summary_exc}")
        # A failed run is always retained for debugging. A successful run is
        # retained when requested; otherwise only the fetched local bundle is
        # kept and the remote process directory is removed.
        _cleanup_remote(args, purge=success and not args.keep_artifacts)
        if success:
            if args.keep_artifacts:
                _log(f"Artifacts retained locally and remotely at {RUN_DIR}")
            else:
                _log(f"Local diagnostic artifacts retained at {RUN_DIR}")
        else:
            _log(f"Failure artifacts retained locally at {RUN_DIR}")
        _write_run_metadata(args, status="passed" if success else "failed", error=failure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH, help="BAGEL model path, visible on both nodes")
    parser.add_argument("--stage0-host", required=True, help="Stage-0 local host identity")
    parser.add_argument("--stage1-host", required=True, help="SSH host name for stage 1")
    parser.add_argument("--stage0-ip", required=True, help="Stage-0 API and RDMA IP")
    parser.add_argument("--stage1-ip", required=True, help="Stage-1 RDMA IP")
    parser.add_argument("--connector", choices=("mooncake", "yuanrong"), default="mooncake")
    parser.add_argument("--rdma-device", default="", help="Optional RDMA HCA name")
    parser.add_argument(
        "--memory-pool-device",
        type=_memory_pool_device_arg,
        default="cpu",
        help="TransferEngine memory pool device; CUDA values require GPU RDMA support",
    )
    parser.add_argument("--memory-pool-size", type=int, default=4294967296, help="Memory pool size in bytes")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--omni-master-port", type=int, default=8091)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Local and remote diagnostic directory; use a unique path to retain multiple runs",
    )
    parser.add_argument(
        "--vllm-logging-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=DEFAULT_LOG_LEVEL if DEFAULT_LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR"} else "DEBUG",
        help="VLLM_LOGGING_LEVEL for both vLLM stages (DEBUG captures connector details)",
    )
    parser.add_argument(
        "--skip-system-info",
        action="store_true",
        help="Skip best-effort GPU/RDMA/network/runtime snapshots",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the remote diagnostic directory after the run; local copies are always kept",
    )
    parser.add_argument("--remote-stage1", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-status", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-cleanup", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-system-info", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    global RUN_DIR
    args = parse_args()
    RUN_DIR = Path(args.run_dir).expanduser()
    if not RUN_DIR.is_absolute():
        RUN_DIR = Path.cwd() / RUN_DIR
    RUN_DIR = RUN_DIR.resolve()
    args.run_dir = str(RUN_DIR)
    if args.remote_stage1:
        return _remote_stage1(args)
    if args.remote_status:
        return _remote_status()
    if args.remote_cleanup:
        _kill_process_group(_record_path(1))
        return 0
    if args.remote_system_info:
        return _remote_system_info(args)
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
