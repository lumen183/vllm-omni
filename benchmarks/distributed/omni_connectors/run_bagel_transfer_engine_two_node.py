#!/usr/bin/env python3
"""Two-node BAGEL TransferEngineConnector end-to-end smoke benchmark.

Run this script on the stage-0 machine. It starts stage 1 through SSH, writes
per-node deploy YAML files under /tmp, submits three concurrent image requests,
then stops only the process groups created by this run.
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
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


MODEL_PATH = "/path/to/BAGEL-7B-MoT"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
BENCH_SCRIPT = SCRIPT_PATH.with_name("bagel_te_bench.py")
RUN_DIR = Path("/tmp/vllm-omni-bagel-cross-node")
ERROR_PATTERN = re.compile(r"Traceback|\bERROR\b|\bException\b|\bRuntimeError\b", re.IGNORECASE)


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


def _connector_config(connector: str, host: str, rdma_device: str) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "host": host,
        "zmq_port": 50051,
        "protocol": "rdma",
        "device_name": rdma_device,
        "memory_pool_size": 4294967296,
        "memory_pool_device": "cpu",
    }
    if connector == "mooncake":
        return {"name": "MooncakeTransferEngineConnector", "extra": extra}
    extra["rpc_port"] = "auto"
    return {"name": "YuanrongTransferEngineConnector", "extra": extra}


def _write_deploy_yaml(stage: int, connector: str, local_ip: str, rdma_device: str) -> Path:
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
    connectors[connector_name] = _connector_config(connector, local_ip, rdma_device)
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
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, shlex.join(command)],
        text=True,
        capture_output=True,
        check=check,
    )


def _remote_script_command(mode: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
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
        "--connector",
        args.connector,
        "--api-port",
        str(args.api_port),
        "--omni-master-port",
        str(args.omni_master_port),
        "--rdma-device",
        args.rdma_device,
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
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


def _start_stage0(args: argparse.Namespace) -> subprocess.Popen[Any]:
    deploy_yaml = _write_deploy_yaml(0, args.connector, args.stage0_ip, args.rdma_device)
    command = [
        "vllm",
        "serve",
        MODEL_PATH,
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
    log_file = (RUN_DIR / "stage0.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
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
    while time.monotonic() < deadline:
        if stage0.poll() is not None:
            raise RuntimeError(f"Stage 0 exited during startup with status {stage0.returncode}")
        if not _remote_alive(args):
            raise RuntimeError("Stage 1 exited during startup")
        if _has_startup_error(stage0_log) or _remote_has_startup_error(args):
            raise RuntimeError("Detected a startup error in a vLLM service log")
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
        MODEL_PATH,
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(args.request_timeout),
    ]
    with (RUN_DIR / "bench.log").open("w", encoding="utf-8") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Bench wrapper failed with status {result.returncode}")


def _fetch_remote_log(args: argparse.Namespace) -> None:
    result = _ssh(args.stage1_host, ["cat", str(RUN_DIR / "stage1.log")], check=False)
    _append_command_output(RUN_DIR / "stage1.log", result)


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
    deploy_yaml = _write_deploy_yaml(1, args.connector, args.stage1_ip, args.rdma_device)
    command = [
        "vllm",
        "serve",
        MODEL_PATH,
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
    log_file = (RUN_DIR / "stage1.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
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


def _prepare_local_run() -> None:
    _kill_process_group(_record_path(0))
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _run_controller(args: argparse.Namespace) -> int:
    _prepare_local_run()
    controller_log = RUN_DIR / "controller.log"
    controller_log.touch()
    stage0: subprocess.Popen[Any] | None = None
    success = False
    try:
        _assert_port_available(args.api_port)
        _assert_port_available(args.omni_master_port)
        _cleanup_remote(args, purge=False)
        _log(f"Artifacts: {RUN_DIR}")
        _log(f"Stage 0 host: {args.stage0_host}; Stage 1 SSH host: {args.stage1_host}")
        _log(f"Connector: {args.connector}")
        stage0 = _start_stage0(args)
        _start_stage1_remote(args)
        _wait_for_startup(stage0, args)
        _run_bench(args)
        if stage0.poll() is not None or not _remote_alive(args):
            raise RuntimeError("A vLLM stage exited before benchmark completion")
        success = True
        _log("BAGEL TransferEngineConnector benchmark completed successfully")
        return 0
    except Exception as exc:
        _log(f"FAILED: {exc}")
        with controller_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"FAILED: {exc}\n")
        return 1
    finally:
        _kill_process_group(_record_path(0))
        _fetch_remote_log(args)
        _cleanup_remote(args, purge=True)
        if not success:
            _log(f"Failure artifacts retained at {RUN_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-host", required=True, help="Stage-0 local host identity")
    parser.add_argument("--stage1-host", required=True, help="SSH host name for stage 1")
    parser.add_argument("--stage0-ip", required=True, help="Stage-0 API and RDMA IP")
    parser.add_argument("--stage1-ip", required=True, help="Stage-1 RDMA IP")
    parser.add_argument("--connector", choices=("mooncake", "yuanrong"), default="mooncake")
    parser.add_argument("--rdma-device", default="", help="Optional RDMA HCA name")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--omni-master-port", type=int, default=8091)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--remote-stage1", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-status", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-cleanup", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.remote_stage1:
        return _remote_stage1(args)
    if args.remote_status:
        return _remote_status()
    if args.remote_cleanup:
        _kill_process_group(_record_path(1))
        return 0
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
