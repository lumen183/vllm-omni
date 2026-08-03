# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-node Yuanrong TransferEngineConnector RDMA benchmark.

This script mirrors cross_node_mooncake_transfer_engine.py but exercises
YuanrongTransferEngineConnector directly. The connector data plane is
requester-pull: producer calls put(), consumer calls get(), and the consumer
pulls from producer with Yuanrong TransferEngine.

CPU RDMA uses a host memory pool. CUDA RDMA uses a CUDA memory pool for
GPUDirect RDMA. Ascend/NPU mode uses an NPU memory pool and requires a working
Ascend TransferEngine runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
            "Yuanrong connector benchmark dependencies are not available. "
            "Install vLLM-Omni runtime dependencies plus openyuanrong-datasystem; "
            f"missing import: {exc}"
        ) from exc

    torch = torch_module
    zmq = zmq_module
    ManagedBuffer = managed_buffer_cls
    TransferEngine = transfer_engine_cls
    YuanrongTransferEngineConnector = yuanrong_connector_cls


def compute_md5(data: torch.Tensor | ManagedBuffer | bytes) -> str:
    assert torch is not None
    assert ManagedBuffer is not None
    if isinstance(data, ManagedBuffer):
        data = data.tensor
    if isinstance(data, bytes):
        payload = data
    else:
        tensor = data.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.md5(payload).hexdigest()


@dataclass
class CtrlMsg:
    msg_type: str
    request_id: str = ""
    md5: str = ""
    data_size: int = 0
    error: str = ""


def _send_ctrl(socket: Any, msg: CtrlMsg) -> None:
    socket.send_json(
        {
            "msg_type": msg.msg_type,
            "request_id": msg.request_id,
            "md5": msg.md5,
            "data_size": msg.data_size,
            "error": msg.error,
        }
    )


def _recv_ctrl(socket: Any) -> CtrlMsg:
    payload = socket.recv_json()
    return CtrlMsg(
        msg_type=str(payload.get("msg_type", "")),
        request_id=str(payload.get("request_id", "")),
        md5=str(payload.get("md5", "")),
        data_size=int(payload.get("data_size", 0)),
        error=str(payload.get("error", "")),
    )


@dataclass
class TransferConfig:
    role: str
    local_host: str
    remote_host: str
    local_port: int
    remote_port: int
    local_rpc_port: str
    ctrl_port: int
    num_transfers: int
    tensor_size_mb: int
    mode: str
    benchmark: bool
    pool_size_mb: int
    protocol: str
    device_name: str
    gpu_id: int
    npu_id: int
    memory_pool_device: str

    @property
    def data_size(self) -> int:
        return self.tensor_size_mb * 1024 * 1024

    @property
    def pool_size(self) -> int:
        return self.pool_size_mb * 1024 * 1024

    @property
    def resolved_pool_device(self) -> str:
        """Resolve the connector pool device for this benchmark mode."""
        if self.protocol != "rdma":
            return f"npu:{self.npu_id}"

        configured = self.memory_pool_device.strip().lower()
        if configured in {"", "auto"}:
            return f"cuda:{self.gpu_id}" if self.mode == "gpu" else "cpu"
        if configured == "cuda":
            return f"cuda:{self.gpu_id}"
        return configured

    @property
    def requires_cuda(self) -> bool:
        return self.mode == "gpu" or self.resolved_pool_device.startswith("cuda:")

    @property
    def cuda_device_id(self) -> int:
        pool_device = self.resolved_pool_device
        if pool_device.startswith("cuda:"):
            return int(pool_device.split(":", 1)[1])
        return self.gpu_id


@dataclass
class TransferStats:
    success_count: int = 0
    fail_count: int = 0
    total_bytes: int = 0
    elapsed_time: float = 0.0
    timing_samples_ms: dict[str, list[float]] = field(default_factory=dict)

    def record_timing(self, name: str, elapsed_ms: float) -> None:
        self.timing_samples_ms.setdefault(name, []).append(elapsed_ms)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile)))
        return ordered[index]

    @property
    def throughput_mbps(self) -> float:
        if self.elapsed_time <= 0:
            return 0.0
        return (self.total_bytes / (1024 * 1024)) / self.elapsed_time

    def print_summary(self, role: str) -> None:
        total = self.success_count + self.fail_count
        print(f"\n{'=' * 60}")
        print(f" {role.upper()} SUMMARY")
        print(f"  Successful: {self.success_count}/{total}")
        print(f"  Failed:     {self.fail_count}/{total}")
        print(f"  Total:      {self.total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Time:       {self.elapsed_time:.2f} s")
        print(f"  Throughput: {self.throughput_mbps:.2f} MB/s")
        for name, samples in self.timing_samples_ms.items():
            if not samples:
                continue
            avg_ms = sum(samples) / len(samples)
            p50_ms = self._percentile(samples, 0.50)
            p95_ms = self._percentile(samples, 0.95)
            print(f"  {name}: avg={avg_ms:.1f} ms, p50={p50_ms:.1f} ms, p95={p95_ms:.1f} ms")
        print(f"{'=' * 60}")


class CrossNodeTester(ABC):
    def __init__(self, config: TransferConfig):
        self.config = config
        self.connector: YuanrongTransferEngineConnector | None = None
        self.zmq_ctx: zmq.Context | None = None
        self.ctrl_socket: zmq.Socket | None = None
        self.stats = TransferStats()

    def get_connector_config(self) -> dict[str, Any]:
        conn_config: dict[str, Any] = {
            "host": self.config.local_host,
            "zmq_port": self.config.local_port,
            "rpc_port": self.config.local_rpc_port,
            "protocol": self.config.protocol,
            "device_name": self.config.device_name,
            "memory_pool_size": self.config.pool_size,
            "role": "sender" if self.config.role == "producer" else "receiver",
        }

        conn_config["memory_pool_device"] = self.config.resolved_pool_device

        if self.config.role == "consumer":
            conn_config["sender_host"] = self.config.remote_host
            conn_config["sender_zmq_port"] = self.config.remote_port

        return conn_config

    def synchronize_cuda(self, timing_name: str, device_id: int | None = None) -> None:
        """Synchronize the benchmark's CUDA device and record its cost."""
        if not self.config.requires_cuda:
            return
        assert torch is not None
        started = time.perf_counter()
        torch.cuda.synchronize(self.config.cuda_device_id if device_id is None else device_id)
        self.stats.record_timing(timing_name, (time.perf_counter() - started) * 1000)

    def initialize(self) -> None:
        assert zmq is not None
        assert YuanrongTransferEngineConnector is not None
        print(f"[{self.role}] Initializing YuanrongTransferEngineConnector...")
        started = time.perf_counter()
        self.connector = YuanrongTransferEngineConnector(self.get_connector_config())
        self.zmq_ctx = zmq.Context()
        self.stats.record_timing("Connector initialize", (time.perf_counter() - started) * 1000)
        info = self.connector.get_connection_info()
        print(
            f"[{self.role}] Ready: host={info['host']} "
            f"zmq_port={info['zmq_port']} rpc_port={info['rpc_port']}"
        )

    def cleanup(self) -> None:
        if self.ctrl_socket is not None:
            self.ctrl_socket.close(linger=0)
        if self.zmq_ctx is not None:
            self.zmq_ctx.term()
        if self.connector is not None:
            self.connector.close()
        print(f"[{self.role}] Closed.")

    @property
    @abstractmethod
    def role(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class Producer(CrossNodeTester):
    @property
    def role(self) -> str:
        return "PRODUCER"

    def print_header(self) -> None:
        print(f"\n{'=' * 60}")
        print(f" PRODUCER MODE ({self.config.mode.upper()}, protocol={self.config.protocol})")
        print(f" Local:        {self.config.local_host}:{self.config.local_port}")
        print(f" Local RPC:    {self.config.local_rpc_port}")
        print(f" Remote:       {self.config.remote_host}:{self.config.remote_port}")
        print(f" Control Port: {self.config.ctrl_port}")
        print(f" Pool Size:    {self.config.pool_size_mb} MB")
        print(f" Pool Device:  {self.config.resolved_pool_device}")
        print(f" Device Name:  {self.config.device_name}")
        print(f"{'=' * 60}\n")

    def setup_control_channel(self) -> None:
        assert zmq is not None
        assert self.zmq_ctx is not None
        self.ctrl_socket = self.zmq_ctx.socket(zmq.REP)
        self.ctrl_socket.bind(f"tcp://*:{self.config.ctrl_port}")
        print(f"[PRODUCER] Control channel listening on port {self.config.ctrl_port}")

    def wait_for_consumer(self) -> bool:
        assert self.ctrl_socket is not None
        msg = _recv_ctrl(self.ctrl_socket)
        if msg.msg_type != "READY":
            print(f"[PRODUCER] Unexpected first message: {msg.msg_type}")
            return False
        _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="ACK"))
        print("[PRODUCER] Consumer connected.")
        return True

    def create_test_data(self, transfer_idx: int) -> tuple[Any, str, int]:
        assert self.connector is not None
        assert torch is not None
        assert ManagedBuffer is not None
        num_elements = self.config.data_size // 4
        data_size = num_elements * 4

        if self.config.mode == "zerocopy":
            offset = self.connector.allocator.alloc(data_size)
            managed_buf = ManagedBuffer(self.connector.allocator, offset, data_size, self.connector.pool)
            if not self.config.benchmark:
                tensor_view = managed_buf.as_tensor(dtype=torch.float32, shape=(num_elements,))
                random_data = torch.randn(num_elements, dtype=torch.float32)
                tensor_view.copy_(random_data.to(tensor_view.device))
                return managed_buf, compute_md5(tensor_view), data_size
            return managed_buf, "", data_size

        if self.config.mode == "gpu":
            device = f"cuda:{self.config.gpu_id}"
            tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            if not self.config.benchmark:
                cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
                tensor.copy_(cpu_tensor.to(device))
                return tensor, compute_md5(cpu_tensor), data_size
            return tensor, "", data_size

        tensor = torch.empty(num_elements, dtype=torch.float32)
        if not self.config.benchmark:
            tensor.normal_()
            return tensor, compute_md5(tensor), data_size
        return tensor, "", data_size

    def do_transfer(self, transfer_idx: int) -> bool:
        assert self.connector is not None
        assert self.ctrl_socket is not None
        req_id = f"cross_node_yuanrong_{transfer_idx}"

        if not self.config.benchmark:
            print(f"\n[PRODUCER] Transfer {transfer_idx + 1}/{self.config.num_transfers}")

        t0 = time.perf_counter()
        data, md5, data_size = self.create_test_data(transfer_idx)
        caller_owned_buffer = data if isinstance(data, ManagedBuffer) else None
        t_create = time.perf_counter() - t0
        self.stats.record_timing("Create payload", t_create * 1000)
        self.synchronize_cuda("Pre-put CUDA sync", self.config.gpu_id)

        t1 = time.perf_counter()
        success, size, _metadata = self.connector.put("producer", "consumer", req_id, data)
        t_put = time.perf_counter() - t1
        self.stats.record_timing("Connector put", t_put * 1000)
        self.synchronize_cuda("Post-put CUDA sync")
        if not success:
            self.stats.fail_count += 1
            if caller_owned_buffer is not None:
                caller_owned_buffer.release()
            print("  [FAIL] Put failed")
            return False

        if not self.config.benchmark:
            print(f"  Size: {data_size / (1024 * 1024):.2f} MB")
            if md5:
                print(f"  MD5:  {md5[:16]}...")
            print(f"  Create time: {t_create * 1000:.1f} ms")
            print(f"  [OK] Put registered, {size} bytes ({t_put * 1000:.1f} ms)")

        t_ready = time.perf_counter()
        msg = _recv_ctrl(self.ctrl_socket)
        self.stats.record_timing("Wait for consumer READY", (time.perf_counter() - t_ready) * 1000)
        if msg.msg_type != "READY":
            self.stats.fail_count += 1
            self.connector.cleanup(req_id, "producer", "consumer")
            if caller_owned_buffer is not None:
                caller_owned_buffer.release()
            print(f"  [ERROR] Unexpected message: {msg.msg_type}")
            return False

        _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="TRANSFER", request_id=req_id, md5=md5, data_size=data_size))

        t2 = time.perf_counter()
        response = _recv_ctrl(self.ctrl_socket)
        t_get = time.perf_counter() - t2
        self.stats.record_timing("Consumer round trip", t_get * 1000)
        ok = response.msg_type == "ACK"
        if ok:
            if not self.config.benchmark:
                print(f"  [OK] Consumer get complete ({t_get * 1000:.1f} ms)")
            self.stats.success_count += 1
            self.stats.total_bytes += size
        else:
            print(f"  [WARN] Consumer reported error: {response.error}")
            self.stats.fail_count += 1

        _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="ACK"))
        t_cleanup = time.perf_counter()
        self.connector.cleanup(req_id, "producer", "consumer")
        if caller_owned_buffer is not None:
            caller_owned_buffer.release()
        self.stats.record_timing("Producer cleanup", (time.perf_counter() - t_cleanup) * 1000)
        return ok

    def run(self) -> None:
        self.print_header()
        self.initialize()
        self.setup_control_channel()

        try:
            if not self.wait_for_consumer():
                return

            start_time = time.perf_counter()
            for idx in range(self.config.num_transfers):
                self.do_transfer(idx)
                if self.config.benchmark and (idx + 1) % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    mbps = (self.stats.total_bytes / (1024 * 1024)) / max(elapsed, 1e-9)
                    print(f"  Progress: {idx + 1}/{self.config.num_transfers}, Throughput: {mbps:.2f} MB/s")

            self.stats.elapsed_time = time.perf_counter() - start_time
            self.stats.print_summary("PRODUCER")

            assert self.ctrl_socket is not None
            self.ctrl_socket.recv()
            _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="DONE"))
        finally:
            self.cleanup()


class Consumer(CrossNodeTester):
    @property
    def role(self) -> str:
        return "CONSUMER"

    def print_header(self) -> None:
        print(f"\n{'=' * 60}")
        print(f" CONSUMER MODE ({self.config.mode.upper()}, protocol={self.config.protocol})")
        print(f" Local:        {self.config.local_host}:{self.config.local_port}")
        print(f" Local RPC:    {self.config.local_rpc_port}")
        print(f" Remote:       {self.config.remote_host}:{self.config.remote_port}")
        print(f" Control Port: {self.config.ctrl_port}")
        print(f" Pool Size:    {self.config.pool_size_mb} MB")
        print(f" Pool Device:  {self.config.resolved_pool_device}")
        print(f" Device Name:  {self.config.device_name}")
        print(f"{'=' * 60}\n")

    def setup_control_channel(self) -> None:
        assert zmq is not None
        assert self.zmq_ctx is not None
        self.ctrl_socket = self.zmq_ctx.socket(zmq.REQ)
        ctrl_addr = f"tcp://{self.config.remote_host}:{self.config.ctrl_port}"
        print(f"[CONSUMER] Connecting to producer control channel at {ctrl_addr}...")
        self.ctrl_socket.connect(ctrl_addr)

    def connect_to_producer(self) -> bool:
        assert self.ctrl_socket is not None
        _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="READY"))
        msg = _recv_ctrl(self.ctrl_socket)
        if msg.msg_type != "ACK":
            print(f"[CONSUMER] Unexpected response: {msg.msg_type}")
            return False
        print("[CONSUMER] Connected to producer.")
        return True

    def do_transfer(self, transfer_idx: int) -> bool:
        assert self.connector is not None
        assert self.ctrl_socket is not None

        if not self.config.benchmark:
            print(f"\n[CONSUMER] Transfer {transfer_idx + 1}/{self.config.num_transfers}")

        t_wait = time.perf_counter()
        _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="READY"))
        msg = _recv_ctrl(self.ctrl_socket)
        self.stats.record_timing("Wait for producer TRANSFER", (time.perf_counter() - t_wait) * 1000)
        if msg.msg_type == "DONE":
            print("[CONSUMER] Producer signaled completion")
            return False
        if msg.msg_type != "TRANSFER":
            print(f"[CONSUMER] Unexpected message: {msg.msg_type}")
            return False

        t0 = time.perf_counter()
        result = self.connector.get("producer", "consumer", msg.request_id, metadata=None)
        t_get = time.perf_counter() - t0
        t_get_ms = t_get * 1000
        self.stats.record_timing("Connector get", t_get_ms)
        if self.config.benchmark:
            print(f"[YR SCRIPT GET] {msg.request_id}: get={t_get_ms:.1f}ms")
        response = CtrlMsg(msg_type="ERROR", error="Get failed")

        if result is None:
            self.stats.fail_count += 1
            print("  [FAIL] Get failed")
        else:
            recv_buffer, recv_size = result
            if not self.config.benchmark:
                print(f"  [OK] Get successful, {recv_size} bytes ({t_get_ms:.1f} ms)")

            try:
                self.synchronize_cuda("Post-get CUDA sync")
                if self.config.benchmark or not msg.md5:
                    response = CtrlMsg(msg_type="ACK")
                    self.stats.success_count += 1
                    self.stats.total_bytes += recv_size
                else:
                    t_checksum = time.perf_counter()
                    recv_md5 = compute_md5(recv_buffer)
                    self.stats.record_timing("Checksum", (time.perf_counter() - t_checksum) * 1000)
                    print(f"  MD5: {recv_md5[:16]}...")
                    if recv_md5 == msg.md5:
                        print("  [PASS] MD5 checksum verified.")
                        response = CtrlMsg(msg_type="ACK")
                        self.stats.success_count += 1
                        self.stats.total_bytes += recv_size
                    else:
                        print("  [FAIL] MD5 mismatch.")
                        response = CtrlMsg(msg_type="ERROR", error="MD5 mismatch")
                        self.stats.fail_count += 1
            finally:
                if ManagedBuffer is not None and isinstance(recv_buffer, ManagedBuffer):
                    recv_buffer.release()

        _send_ctrl(self.ctrl_socket, response)
        t_ack = time.perf_counter()
        self.ctrl_socket.recv()
        self.stats.record_timing("Wait for producer ACK", (time.perf_counter() - t_ack) * 1000)
        return response.msg_type == "ACK"

    def run(self) -> None:
        self.print_header()
        self.initialize()
        self.setup_control_channel()

        try:
            if not self.connect_to_producer():
                return

            start_time = time.perf_counter()
            for idx in range(self.config.num_transfers):
                if not self.do_transfer(idx):
                    break
                if self.config.benchmark and (idx + 1) % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    mbps = (self.stats.total_bytes / (1024 * 1024)) / max(elapsed, 1e-9)
                    print(f"  Progress: {idx + 1}/{self.config.num_transfers}, Throughput: {mbps:.2f} MB/s")

            self.stats.elapsed_time = time.perf_counter() - start_time
            self.stats.print_summary("CONSUMER")

            assert self.ctrl_socket is not None
            _send_ctrl(self.ctrl_socket, CtrlMsg(msg_type="READY"))
            self.ctrl_socket.recv()
        finally:
            self.cleanup()


def parse_args() -> TransferConfig:
    parser = argparse.ArgumentParser(
        description="Cross-node YuanrongTransferEngineConnector benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPU RDMA copy mode:
  python cross_node_yuanrong_transfer_engine.py --role producer --local-host <PRODUCER_IP> --remote-host <CONSUMER_IP>

  python cross_node_yuanrong_transfer_engine.py --role consumer --local-host <CONSUMER_IP> --remote-host <PRODUCER_IP>

  # CPU RDMA zero-copy pool mode:
  python cross_node_yuanrong_transfer_engine.py --role producer ... --mode zerocopy

  # CUDA GPUDirect RDMA mode (use the same options on both nodes):
  python cross_node_yuanrong_transfer_engine.py --role producer ... \
      --protocol rdma --mode gpu --gpu-id 0 --memory-pool-device cuda

  # CUDA pool without an extra source-tensor copy:
  python cross_node_yuanrong_transfer_engine.py --role producer ... \
      --protocol rdma --mode zerocopy --memory-pool-device cuda:0

  # Ascend/NPU mode:
  python cross_node_yuanrong_transfer_engine.py --role producer ... \
      --protocol ascend --mode npu --npu-id 0 --device-name auto
        """,
    )
    parser.add_argument("--role", required=True, choices=["producer", "consumer"])
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--local-port", type=int, default=15500, help="local connector ZMQ metadata port")
    parser.add_argument("--remote-port", type=int, default=15500, help="remote connector ZMQ metadata port")
    parser.add_argument(
        "--local-rpc-port",
        default="auto",
        help="local Yuanrong TransferEngine RPC port, or 'auto'",
    )
    parser.add_argument("--ctrl-port", type=int, default=15501, help="script control channel port")
    parser.add_argument("--num-transfers", type=int, default=20)
    parser.add_argument("--tensor-size-mb", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["copy", "zerocopy", "gpu", "npu"],
        default="copy",
        help=(
            "copy/zerocopy use connector pool; gpu uses a CUDA source tensor "
            "and CUDA pool by default; npu uses Ascend pool"
        ),
    )
    parser.add_argument("--benchmark", action="store_true", help="skip random data generation and MD5 verification")
    parser.add_argument("--pool-size-mb", type=int, default=512)
    parser.add_argument("--protocol", choices=["rdma", "ascend"], default="rdma")
    parser.add_argument("--device-name", default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument(
        "--memory-pool-device",
        default="auto",
        help="RDMA pool device: auto, cpu, cuda, or cuda:N; auto selects cuda:<gpu-id> for gpu mode and cpu otherwise",
    )
    args = parser.parse_args()

    if args.num_transfers <= 0:
        parser.error("--num-transfers must be positive")
    if args.tensor_size_mb <= 0:
        parser.error("--tensor-size-mb must be positive")
    if args.pool_size_mb <= args.tensor_size_mb:
        parser.error("--pool-size-mb must be larger than --tensor-size-mb")
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    if args.npu_id < 0:
        parser.error("--npu-id must be non-negative")
    if args.protocol == "rdma" and args.mode == "npu":
        parser.error("--mode npu requires --protocol ascend")
    if args.protocol == "ascend" and args.mode in {"copy", "zerocopy"}:
        parser.error("--protocol ascend requires --mode npu")
    pool_device = str(args.memory_pool_device).strip().lower()
    if pool_device not in {"", "auto", "cpu", "cuda"} and not (
        pool_device.startswith("cuda:") and pool_device[5:].isdigit()
    ):
        parser.error("--memory-pool-device must be auto, cpu, cuda, or cuda:N")
    if args.protocol == "ascend" and pool_device not in {"", "auto", "npu"} and not pool_device.startswith("npu:"):
        parser.error("--memory-pool-device is only applicable to RDMA; Ascend mode uses the NPU pool")
    return TransferConfig(
        role=args.role,
        local_host=args.local_host,
        remote_host=args.remote_host,
        local_port=args.local_port,
        remote_port=args.remote_port,
        local_rpc_port=str(args.local_rpc_port),
        ctrl_port=args.ctrl_port,
        num_transfers=args.num_transfers,
        tensor_size_mb=args.tensor_size_mb,
        mode=args.mode,
        benchmark=args.benchmark,
        pool_size_mb=args.pool_size_mb,
        protocol=args.protocol,
        device_name=args.device_name,
        gpu_id=args.gpu_id,
        npu_id=args.npu_id,
        memory_pool_device=pool_device,
    )


def main() -> int:
    config = parse_args()

    try:
        _load_runtime_deps()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if TransferEngine is None:
        print("[ERROR] Yuanrong TransferEngine Python binding is not available.")
        print("Install openyuanrong-datasystem or build/export transfer_engine Python artifacts first.")
        return 1

    assert torch is not None
    if config.requires_cuda:
        if not torch.cuda.is_available():
            print("[ERROR] The selected source/pool configuration requires CUDA.")
            return 1
        required_gpu_ids = {config.cuda_device_id}
        if config.mode == "gpu":
            required_gpu_ids.add(config.gpu_id)
        device_count = torch.cuda.device_count()
        for gpu_id in sorted(required_gpu_ids):
            if gpu_id >= device_count:
                print(f"[ERROR] GPU {gpu_id} is not available; found {device_count} GPUs.")
                return 1
        print(f"[INFO] CUDA pool: {config.resolved_pool_device}")
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(config.cuda_device_id)}")

    runner: CrossNodeTester
    if config.role == "producer":
        runner = Producer(config)
    else:
        runner = Consumer(config)
    runner.run()
    return 0 if runner.stats.fail_count == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
