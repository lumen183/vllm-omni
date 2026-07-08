# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-node Yuanrong multi-HCA stress benchmark.

This benchmark exercises YuanrongTransferEngineConnector in a multi-process,
multi-lane CPU RDMA shape:

* each node starts N worker processes;
* each worker owns one YuanrongTransferEngineConnector;
* every round prepares L local buffers, then concurrently pulls L remote
  buffers from the peer worker with the same rank.

Run the same command on both nodes with different --node-id values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))


_add_repo_root_to_path()

torch = None
zmq = None
ManagedBuffer = None
TransferEngine = None
YuanrongTransferEngineConnector = None


def _load_runtime_deps() -> None:
    global torch, zmq, ManagedBuffer, TransferEngine, YuanrongTransferEngineConnector

    try:
        import torch as torch_module
        import zmq as zmq_module
        from vllm_omni.distributed.omni_connectors.utils.memory_pool import (
            ManagedBuffer as managed_buffer_cls,
        )
        from vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector import (
            TransferEngine as transfer_engine_cls,
            YuanrongTransferEngineConnector as yuanrong_connector_cls,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Yuanrong multi-HCA benchmark dependencies are not available. "
            "Install vLLM-Omni runtime dependencies plus openyuanrong-datasystem; "
            f"missing import: {exc}"
        ) from exc

    torch = torch_module
    zmq = zmq_module
    ManagedBuffer = managed_buffer_cls
    TransferEngine = transfer_engine_cls
    YuanrongTransferEngineConnector = yuanrong_connector_cls


@dataclass
class BenchConfig:
    node_id: int
    local_host: str
    remote_host: str
    base_zmq_port: int
    base_rpc_port: int
    base_ctrl_port: int
    processes: int
    lanes: int
    rounds: int
    sizes_mb: list[int]
    pool_size_mb: int
    rdma_devices: str
    verify: str
    warmup_rounds: int
    barrier_timeout_s: float
    log_dir: str
    expect_hca_count: int


@dataclass
class LaneResult:
    node_id: int
    rank: int
    round_index: int
    size_mb: int
    lane: int
    request_id: str
    success: bool
    bytes_transferred: int
    start_ns: int
    end_ns: int
    duration_ms: float
    throughput_mib_s: float
    error: str = ""


@dataclass
class RoundResult:
    node_id: int
    rank: int
    round_index: int
    size_mb: int
    success: bool
    overlap: bool
    bytes_transferred: int
    start_ns: int
    end_ns: int
    duration_ms: float
    throughput_mib_s: float
    lane_results: list[LaneResult]
    error: str = ""


@dataclass
class WorkerSummary:
    node_id: int
    rank: int
    success: bool
    rounds: int
    failed_rounds: int
    lane_success: int
    lane_failed: int
    total_bytes: int
    measured_seconds: float
    throughput_mib_s: float
    error: str = ""


class WorkerLog:
    def __init__(self, config: BenchConfig, rank: int):
        self.config = config
        self.rank = rank
        self.file = None
        if config.log_dir:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            path = Path(config.log_dir) / f"node{config.node_id}_rank{rank}.log"
            self.file = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def emit(self, tag: str, **fields: Any) -> None:
        fields.setdefault("node", self.config.node_id)
        fields.setdefault("rank", self.rank)
        fields.setdefault("pid", os.getpid())
        line = f"[{tag}] " + " ".join(f"{key}={_format_field(value)}" for key, value in fields.items())
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()


def _format_field(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    if any(ch.isspace() for ch in text):
        return json.dumps(text, ensure_ascii=True)
    return text


def _request_id(node_id: int, rank: int, size_mb: int, round_index: int, lane: int) -> str:
    return f"yr_multi_hca_node{node_id}_rank{rank}_size{size_mb}_round{round_index}_lane{lane}"


def _make_tensor(byte_count: int, pattern: int, verify: str) -> Any:
    assert torch is not None
    tensor = torch.empty(byte_count, dtype=torch.uint8)
    if verify in {"pattern", "md5"}:
        tensor.fill_(pattern)
    return tensor


def _buffer_to_cpu_uint8(value: Any) -> Any:
    assert torch is not None
    assert ManagedBuffer is not None
    if isinstance(value, ManagedBuffer):
        value = value.tensor
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(bytearray(value), dtype=torch.uint8)
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.contiguous().view(torch.uint8)


def _md5_tensor(value: Any) -> str:
    tensor = _buffer_to_cpu_uint8(value)
    return hashlib.md5(tensor.numpy().tobytes()).hexdigest()


def _verify_payload(value: Any, expected_pattern: int, expected_md5: str, verify: str) -> None:
    if verify == "none":
        return
    tensor = _buffer_to_cpu_uint8(value)
    if verify == "pattern":
        # Full pattern validation is intentional for correctness mode. Use
        # --verify none for pure throughput measurements.
        mismatches = torch.count_nonzero(tensor != expected_pattern).item()
        if mismatches:
            raise RuntimeError(f"pattern mismatch, expected={expected_pattern}, mismatches={mismatches}")
        return
    if verify == "md5":
        actual = hashlib.md5(tensor.numpy().tobytes()).hexdigest()
        if actual != expected_md5:
            raise RuntimeError(f"md5 mismatch, expected={expected_md5}, actual={actual}")
        return
    raise RuntimeError(f"unsupported verify mode: {verify}")


def _connector_config(config: BenchConfig, rank: int) -> dict[str, Any]:
    return {
        "host": config.local_host,
        "zmq_port": config.base_zmq_port + rank,
        "rpc_port": str(config.base_rpc_port + rank),
        "protocol": "rdma",
        "device_name": "cpu:*",
        "memory_pool_size": config.pool_size_mb * 1024 * 1024,
        "memory_pool_device": "cpu",
        "role": "sender",
        "sender_host": config.remote_host,
        "sender_zmq_port": config.base_zmq_port + rank,
    }


def _setup_ctrl_socket(config: BenchConfig, rank: int, log: WorkerLog) -> tuple[Any, Any]:
    assert zmq is not None
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP if config.node_id == 0 else zmq.REQ)
    timeout_ms = int(config.barrier_timeout_s * 1000)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    ctrl_port = config.base_ctrl_port + rank
    if config.node_id == 0:
        addr = f"tcp://*:{ctrl_port}"
        sock.bind(addr)
        log.emit("YR_MULTI_HCA_CTRL_BIND", addr=addr)
    else:
        addr = f"tcp://{config.remote_host}:{ctrl_port}"
        sock.connect(addr)
        log.emit("YR_MULTI_HCA_CTRL_CONNECT", addr=addr)
    return ctx, sock


def _sync_round(sock: Any, config: BenchConfig, rank: int, payload: dict[str, Any], log: WorkerLog) -> None:
    if config.node_id == 0:
        remote = sock.recv_json()
        if remote.get("msg_type") != "READY":
            raise RuntimeError(f"unexpected peer barrier message: {remote}")
        sock.send_json({"msg_type": "START", **payload})
    else:
        sock.send_json({"msg_type": "READY", **payload})
        remote = sock.recv_json()
        if remote.get("msg_type") != "START":
            raise RuntimeError(f"unexpected peer barrier response: {remote}")
    log.emit("YR_MULTI_HCA_BARRIER_READY", size_mb=payload["size_mb"], round=payload["round_index"])


def _sync_done(sock: Any, config: BenchConfig, payload: dict[str, Any]) -> None:
    if config.node_id == 0:
        remote = sock.recv_json()
        if remote.get("msg_type") != "DONE":
            raise RuntimeError(f"unexpected peer done message: {remote}")
        sock.send_json({"msg_type": "ACK", **payload})
    else:
        sock.send_json({"msg_type": "DONE", **payload})
        remote = sock.recv_json()
        if remote.get("msg_type") != "ACK":
            raise RuntimeError(f"unexpected peer done response: {remote}")


def _release_if_managed(value: Any) -> None:
    if ManagedBuffer is not None and isinstance(value, ManagedBuffer):
        value.release()


def _do_get_lane(
    connector: Any,
    config: BenchConfig,
    log: WorkerLog,
    rank: int,
    size_mb: int,
    round_index: int,
    lane: int,
    expected_pattern: int,
    expected_md5: str,
) -> LaneResult:
    remote_node = 1 - config.node_id
    request_id = _request_id(remote_node, rank, size_mb, round_index, lane)
    byte_count = size_mb * 1024 * 1024
    start_ns = time.monotonic_ns()
    log.emit(
        "YR_MULTI_HCA_LANE_START",
        size_mb=size_mb,
        round=round_index,
        lane=lane,
        request_id=request_id,
        bytes=byte_count,
    )
    try:
        result = connector.get("producer", "consumer", request_id, metadata=None)
        if result is None:
            raise RuntimeError("connector.get returned None")
        recv_buffer, recv_size = result
        try:
            if recv_size != byte_count:
                raise RuntimeError(f"size mismatch, expected={byte_count}, actual={recv_size}")
            _verify_payload(recv_buffer, expected_pattern, expected_md5, config.verify)
        finally:
            _release_if_managed(recv_buffer)
        end_ns = time.monotonic_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000
        throughput = (byte_count / 1024 / 1024) / max(duration_ms / 1000, 1e-9)
        log.emit(
            "YR_MULTI_HCA_LANE_DONE",
            size_mb=size_mb,
            round=round_index,
            lane=lane,
            request_id=request_id,
            duration_ms=duration_ms,
            throughput_mib_s=throughput,
        )
        return LaneResult(
            node_id=config.node_id,
            rank=rank,
            round_index=round_index,
            size_mb=size_mb,
            lane=lane,
            request_id=request_id,
            success=True,
            bytes_transferred=byte_count,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ms=duration_ms,
            throughput_mib_s=throughput,
        )
    except Exception as exc:
        end_ns = time.monotonic_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000
        log.emit(
            "YR_MULTI_HCA_LANE_FAIL",
            size_mb=size_mb,
            round=round_index,
            lane=lane,
            request_id=request_id,
            duration_ms=duration_ms,
            error=str(exc),
        )
        return LaneResult(
            node_id=config.node_id,
            rank=rank,
            round_index=round_index,
            size_mb=size_mb,
            lane=lane,
            request_id=request_id,
            success=False,
            bytes_transferred=0,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ms=duration_ms,
            throughput_mib_s=0.0,
            error=str(exc),
        )


def _run_round(
    connector: Any,
    ctrl_sock: Any,
    config: BenchConfig,
    log: WorkerLog,
    rank: int,
    size_mb: int,
    round_index: int,
) -> RoundResult:
    byte_count = size_mb * 1024 * 1024
    local_node = config.node_id
    remote_node = 1 - config.node_id
    holders: list[tuple[str, Any]] = []
    expected_remote: dict[int, tuple[int, str]] = {}

    log.emit("YR_MULTI_HCA_ROUND_START", size_mb=size_mb, round=round_index, lanes=config.lanes, bytes=byte_count)
    for lane in range(config.lanes):
        request_id = _request_id(local_node, rank, size_mb, round_index, lane)
        pattern = (local_node * 97 + rank * 17 + round_index * 7 + lane * 31) % 256
        tensor = _make_tensor(byte_count, pattern, config.verify)
        expected_md5 = _md5_tensor(tensor) if config.verify == "md5" else ""
        success, size, _metadata = connector.put("producer", "consumer", request_id, tensor)
        if not success:
            raise RuntimeError(f"put failed for {request_id}")
        if size != byte_count:
            raise RuntimeError(f"put size mismatch for {request_id}: expected={byte_count}, actual={size}")
        holders.append((request_id, tensor))

    for lane in range(config.lanes):
        pattern = (remote_node * 97 + rank * 17 + round_index * 7 + lane * 31) % 256
        expected_md5 = ""
        if config.verify == "md5":
            probe = _make_tensor(byte_count, pattern, config.verify)
            expected_md5 = _md5_tensor(probe)
        expected_remote[lane] = (pattern, expected_md5)

    _sync_round(
        ctrl_sock,
        config,
        rank,
        {"size_mb": size_mb, "round_index": round_index, "lanes": config.lanes},
        log,
    )

    round_start_ns = time.monotonic_ns()
    lane_results: list[LaneResult] = []
    with ThreadPoolExecutor(max_workers=config.lanes, thread_name_prefix=f"yr-rank{rank}-lane") as executor:
        futures = [
            executor.submit(
                _do_get_lane,
                connector,
                config,
                log,
                rank,
                size_mb,
                round_index,
                lane,
                expected_remote[lane][0],
                expected_remote[lane][1],
            )
            for lane in range(config.lanes)
        ]
        for future in as_completed(futures):
            lane_results.append(future.result())
    round_end_ns = time.monotonic_ns()

    for request_id, _tensor in holders:
        connector.cleanup(request_id, "producer", "consumer")

    lane_results.sort(key=lambda item: item.lane)
    all_success = all(item.success for item in lane_results)
    overlap = False
    if lane_results:
        overlap = max(item.start_ns for item in lane_results) < min(item.end_ns for item in lane_results)
    transferred = sum(item.bytes_transferred for item in lane_results)
    duration_ms = (round_end_ns - round_start_ns) / 1_000_000
    throughput = (transferred / 1024 / 1024) / max(duration_ms / 1000, 1e-9)
    success = all_success and overlap
    tag = "YR_MULTI_HCA_ROUND_PASS" if success else "YR_MULTI_HCA_ROUND_FAIL"
    error = ""
    if not all_success:
        error = "one_or_more_lanes_failed"
    elif not overlap:
        error = "lane_tasks_did_not_overlap"
    log.emit(
        tag,
        size_mb=size_mb,
        round=round_index,
        lanes=config.lanes,
        overlap=str(overlap).lower(),
        duration_ms=duration_ms,
        throughput_mib_s=throughput,
        bytes=transferred,
        error=error,
    )
    _sync_done(
        ctrl_sock,
        config,
        {
            "size_mb": size_mb,
            "round_index": round_index,
            "success": success,
            "bytes": transferred,
        },
    )
    return RoundResult(
        node_id=config.node_id,
        rank=rank,
        round_index=round_index,
        size_mb=size_mb,
        success=success,
        overlap=overlap,
        bytes_transferred=transferred,
        start_ns=round_start_ns,
        end_ns=round_end_ns,
        duration_ms=duration_ms,
        throughput_mib_s=throughput,
        lane_results=lane_results,
        error=error,
    )


def _worker_main(config: BenchConfig, rank: int, result_queue: Any) -> None:
    log = WorkerLog(config, rank)
    connector = None
    ctrl_ctx = None
    ctrl_sock = None
    round_results: list[RoundResult] = []
    measured_start_ns = 0
    measured_end_ns = 0
    try:
        _load_runtime_deps()
        if TransferEngine is None:
            raise RuntimeError("Yuanrong TransferEngine Python binding is not available")

        if config.rdma_devices:
            os.environ["TRANSFER_ENGINE_CPU_RDMA_DEVICE_NAME"] = config.rdma_devices
        os.environ.setdefault("TRANSFER_ENGINE_CPU_RDMA_EXCLUSIVE_HCA", "0")

        log.emit(
            "YR_MULTI_HCA_WORKER_START",
            local_host=config.local_host,
            remote_host=config.remote_host,
            zmq_port=config.base_zmq_port + rank,
            rpc_port=config.base_rpc_port + rank,
            ctrl_port=config.base_ctrl_port + rank,
            lanes=config.lanes,
            rdma_devices=config.rdma_devices or "auto",
        )

        assert YuanrongTransferEngineConnector is not None
        connector = YuanrongTransferEngineConnector(_connector_config(config, rank))
        info = connector.get_connection_info()
        log.emit(
            "YR_MULTI_HCA_WORKER_READY",
            host=info["host"],
            zmq_port=info["zmq_port"],
            rpc_port=info["rpc_port"],
            can_put=info["can_put"],
        )

        ctrl_ctx, ctrl_sock = _setup_ctrl_socket(config, rank, log)

        for size_mb in config.sizes_mb:
            total_rounds = config.warmup_rounds + config.rounds
            for round_index in range(total_rounds):
                is_measured = round_index >= config.warmup_rounds
                if is_measured and measured_start_ns == 0:
                    measured_start_ns = time.monotonic_ns()
                result = _run_round(connector, ctrl_sock, config, log, rank, size_mb, round_index)
                if is_measured:
                    round_results.append(result)
                    measured_end_ns = time.monotonic_ns()

        failed_rounds = sum(1 for item in round_results if not item.success)
        lane_success = sum(1 for item in round_results for lane in item.lane_results if lane.success)
        lane_failed = sum(1 for item in round_results for lane in item.lane_results if not lane.success)
        total_bytes = sum(item.bytes_transferred for item in round_results)
        measured_seconds = (measured_end_ns - measured_start_ns) / 1_000_000_000 if measured_start_ns else 0.0
        throughput = (total_bytes / 1024 / 1024) / max(measured_seconds, 1e-9)
        summary = WorkerSummary(
            node_id=config.node_id,
            rank=rank,
            success=failed_rounds == 0 and lane_failed == 0,
            rounds=len(round_results),
            failed_rounds=failed_rounds,
            lane_success=lane_success,
            lane_failed=lane_failed,
            total_bytes=total_bytes,
            measured_seconds=measured_seconds,
            throughput_mib_s=throughput,
        )
        log.emit(
            "YR_MULTI_HCA_WORKER_SUMMARY",
            success=str(summary.success).lower(),
            rounds=summary.rounds,
            failed_rounds=summary.failed_rounds,
            lane_success=summary.lane_success,
            lane_failed=summary.lane_failed,
            total_mib=summary.total_bytes / 1024 / 1024,
            measured_seconds=summary.measured_seconds,
            throughput_mib_s=summary.throughput_mib_s,
        )
        result_queue.put({"summary": asdict(summary), "rounds": [_round_to_dict(item) for item in round_results]})
    except Exception as exc:
        log.emit("YR_MULTI_HCA_WORKER_FAIL", error=str(exc), traceback=traceback.format_exc().replace("\n", "\\n"))
        result_queue.put(
            {
                "summary": asdict(
                    WorkerSummary(
                        node_id=config.node_id,
                        rank=rank,
                        success=False,
                        rounds=len(round_results),
                        failed_rounds=max(1, sum(1 for item in round_results if not item.success)),
                        lane_success=sum(1 for item in round_results for lane in item.lane_results if lane.success),
                        lane_failed=max(1, sum(1 for item in round_results for lane in item.lane_results if not lane.success)),
                        total_bytes=sum(item.bytes_transferred for item in round_results),
                        measured_seconds=0.0,
                        throughput_mib_s=0.0,
                        error=str(exc),
                    )
                ),
                "rounds": [_round_to_dict(item) for item in round_results],
            }
        )
    finally:
        if ctrl_sock is not None:
            ctrl_sock.close(linger=0)
        if ctrl_ctx is not None:
            ctrl_ctx.term()
        if connector is not None:
            connector.close()
        log.emit("YR_MULTI_HCA_WORKER_CLOSED")
        log.close()


def _round_to_dict(item: RoundResult) -> dict[str, Any]:
    data = asdict(item)
    data["lane_results"] = [asdict(lane) for lane in item.lane_results]
    return data


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _print_node_summary(config: BenchConfig, results: list[dict[str, Any]], elapsed_s: float) -> int:
    summaries = [item["summary"] for item in results]
    rounds = [round_item for item in results for round_item in item["rounds"]]
    failed_workers = [item for item in summaries if not item["success"]]
    failed_rounds = [item for item in rounds if not item["success"]]
    durations = [float(item["duration_ms"]) for item in rounds if item["success"]]
    total_bytes = sum(int(item["total_bytes"]) for item in summaries)
    throughput = (total_bytes / 1024 / 1024) / max(elapsed_s, 1e-9)
    avg_ms = statistics.mean(durations) if durations else 0.0
    print(
        "[YR_MULTI_HCA_NODE_SUMMARY] "
        f"node={config.node_id} processes={config.processes} lanes={config.lanes} "
        f"sizes_mb={','.join(str(v) for v in config.sizes_mb)} rounds={config.rounds} "
        f"warmup_rounds={config.warmup_rounds} rdma_devices={config.rdma_devices or 'auto'} "
        f"success_workers={len(summaries) - len(failed_workers)}/{len(summaries)} "
        f"failed_rounds={len(failed_rounds)} total_mib={total_bytes / 1024 / 1024:.3f} "
        f"elapsed_s={elapsed_s:.3f} throughput_mib_s={throughput:.3f} "
        f"avg_round_ms={avg_ms:.3f} p50_round_ms={_percentile(durations, 0.50):.3f} "
        f"p95_round_ms={_percentile(durations, 0.95):.3f} max_round_ms={(max(durations) if durations else 0.0):.3f}",
        flush=True,
    )
    for size_mb in config.sizes_mb:
        size_rounds = [item for item in rounds if int(item["size_mb"]) == size_mb]
        size_success = [item for item in size_rounds if item["success"]]
        size_durations = [float(item["duration_ms"]) for item in size_success]
        size_bytes = sum(int(item["bytes_transferred"]) for item in size_success)
        first_start = min((int(item["start_ns"]) for item in size_success), default=0)
        last_end = max((int(item["end_ns"]) for item in size_success), default=0)
        size_seconds = (last_end - first_start) / 1_000_000_000 if first_start else 0.0
        size_throughput = (size_bytes / 1024 / 1024) / max(size_seconds, 1e-9)
        print(
            "[YR_MULTI_HCA_SIZE_SUMMARY] "
            f"node={config.node_id} size_mb={size_mb} rounds={len(size_rounds)} "
            f"success={len(size_success)} failed={len(size_rounds) - len(size_success)} "
            f"avg_ms={(statistics.mean(size_durations) if size_durations else 0.0):.3f} "
            f"p50_ms={_percentile(size_durations, 0.50):.3f} "
            f"p95_ms={_percentile(size_durations, 0.95):.3f} "
            f"throughput_mib_s={size_throughput:.3f}",
            flush=True,
        )
    if config.expect_hca_count > 0:
        print(
            "[YR_MULTI_HCA_HCA_CHECK] "
            f"node={config.node_id} expected_hca_count={config.expect_hca_count} "
            "status=requires_backend_lane_select_logs",
            flush=True,
        )
    print(
        "[YR_MULTI_HCA_BENCHMARK_SUMMARY] "
        f"node={config.node_id} success={not failed_workers and not failed_rounds} "
        f"workers={len(summaries)} failed_workers={len(failed_workers)} failed_rounds={len(failed_rounds)}",
        flush=True,
    )
    return 0 if not failed_workers and not failed_rounds else 1


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(
        description="Cross-node YuanrongTransferEngineConnector multi-HCA stress benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Node A
  python cross_node_yuanrong_multi_hca_stress.py \\
      --node-id 0 --local-host <A_IP> --remote-host <B_IP> \\
      --rdma-devices mlx5_0,mlx5_1,mlx5_2

  # Node B
  python cross_node_yuanrong_multi_hca_stress.py \\
      --node-id 1 --local-host <B_IP> --remote-host <A_IP> \\
      --rdma-devices mlx5_0,mlx5_1,mlx5_2
        """,
    )
    parser.add_argument("--node-id", type=int, required=True, choices=[0, 1])
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--base-zmq-port", type=int, default=15500)
    parser.add_argument("--base-rpc-port", type=int, default=16500)
    parser.add_argument("--base-ctrl-port", type=int, default=17500)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--sizes-mb", default="64,256,1024")
    parser.add_argument("--pool-size-mb", type=int, default=4096)
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--verify", choices=["none", "pattern", "md5"], default="pattern")
    parser.add_argument("--barrier-timeout-s", type=float, default=120.0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--expect-hca-count", type=int, default=0)
    args = parser.parse_args()

    sizes_mb = []
    for raw in str(args.sizes_mb).split(","):
        raw = raw.strip()
        if raw:
            sizes_mb.append(int(raw))
    if not sizes_mb:
        parser.error("--sizes-mb must contain at least one positive size")
    if any(size <= 0 for size in sizes_mb):
        parser.error("--sizes-mb values must be positive")
    if args.processes <= 0:
        parser.error("--processes must be positive")
    if args.lanes <= 0:
        parser.error("--lanes must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be non-negative")
    max_size = max(sizes_mb)
    min_pool = max_size * args.lanes * 2
    if args.pool_size_mb < min_pool:
        parser.error(f"--pool-size-mb should be at least {min_pool} for max size and lane count")
    return BenchConfig(
        node_id=args.node_id,
        local_host=args.local_host,
        remote_host=args.remote_host,
        base_zmq_port=args.base_zmq_port,
        base_rpc_port=args.base_rpc_port,
        base_ctrl_port=args.base_ctrl_port,
        processes=args.processes,
        lanes=args.lanes,
        rounds=args.rounds,
        sizes_mb=sizes_mb,
        pool_size_mb=args.pool_size_mb,
        rdma_devices=args.rdma_devices,
        verify=args.verify,
        warmup_rounds=args.warmup_rounds,
        barrier_timeout_s=args.barrier_timeout_s,
        log_dir=args.log_dir,
        expect_hca_count=args.expect_hca_count,
    )


def main() -> int:
    config = parse_args()
    print(
        "[YR_MULTI_HCA_SCRIPT_START] "
        f"node={config.node_id} local_host={config.local_host} remote_host={config.remote_host} "
        f"processes={config.processes} lanes={config.lanes} sizes_mb={','.join(str(v) for v in config.sizes_mb)} "
        f"rounds={config.rounds} warmup_rounds={config.warmup_rounds} "
        f"rdma_devices={config.rdma_devices or 'auto'} verify={config.verify}",
        flush=True,
    )

    start = time.monotonic()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=_worker_main, args=(config, rank, result_queue), name=f"yr-multi-hca-rank{rank}")
        for rank in range(config.processes)
    ]
    for worker in workers:
        worker.start()

    results: list[dict[str, Any]] = []
    while len(results) < len(workers):
        try:
            results.append(result_queue.get(timeout=1.0))
            continue
        except queue.Empty:
            pass
        if not any(worker.is_alive() for worker in workers):
            break

    exit_failed = False
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            exit_failed = True
            print(
                "[YR_MULTI_HCA_PROCESS_FAIL] "
                f"node={config.node_id} process={worker.name} exitcode={worker.exitcode}",
                flush=True,
            )
    if len(results) < len(workers):
        exit_failed = True
        print(
            "[YR_MULTI_HCA_RESULT_MISSING] "
            f"node={config.node_id} expected={len(workers)} actual={len(results)}",
            flush=True,
        )

    elapsed = time.monotonic() - start
    rc = _print_node_summary(config, results, elapsed)
    return 1 if exit_failed else rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
