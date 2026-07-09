# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for manual cross-node Yuanrong connector scenario tests."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))


add_repo_root_to_path()

torch = None
zmq = None
ManagedBuffer = None
TransferEngine = None
YuanrongTransferEngineConnector = None


def load_runtime_deps() -> None:
    global torch, zmq, ManagedBuffer, TransferEngine, YuanrongTransferEngineConnector
    try:
        import torch as torch_module
        import zmq as zmq_module
        from vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector import (
            TransferEngine as transfer_engine_cls,
            YuanrongTransferEngineConnector as connector_cls,
        )
        from vllm_omni.distributed.omni_connectors.utils.memory_pool import (
            ManagedBuffer as managed_buffer_cls,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Yuanrong scenario dependencies are unavailable. Install torch, pyzmq, "
            "msgspec and openyuanrong-datasystem for CPU RDMA tests."
        ) from exc

    torch = torch_module
    zmq = zmq_module
    ManagedBuffer = managed_buffer_cls
    TransferEngine = transfer_engine_cls
    YuanrongTransferEngineConnector = connector_cls

    if TransferEngine is None:
        raise RuntimeError("Yuanrong TransferEngine Python binding is unavailable.")


@dataclass
class Ports:
    a_zmq: int = 15500
    a_rpc: int = 15600
    a_recv_rpc: int = 15620
    b_zmq: int = 15510
    b_rpc: int = 15610
    b_recv_rpc: int = 15630
    ctrl: int = 15700


@dataclass
class Endpoint:
    host: str
    zmq_port: int
    rpc_port: int


@dataclass
class CaseResult:
    name: str
    status: str
    expected_failure: bool = False
    unsupported: bool = False
    failure_phase: str = ""
    failure_reason: str = ""
    put_ms: float = 0.0
    get_ms: float = 0.0
    total_ms: float = 0.0
    bytes: int = 0
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def result_ok(name: str, started: float, **kwargs: Any) -> CaseResult:
    return CaseResult(name=name, status="PASS", total_ms=elapsed_ms(started), **kwargs)


def result_fail(
    name: str,
    started: float,
    phase: str,
    reason: str,
    *,
    expected: bool = False,
    unsupported: bool = False,
    **kwargs: Any,
) -> CaseResult:
    status = "EXPECTED_FAIL" if expected else "UNSUPPORTED" if unsupported else "FAIL"
    return CaseResult(
        name=name,
        status=status,
        expected_failure=expected,
        unsupported=unsupported,
        failure_phase=phase,
        failure_reason=reason,
        total_ms=elapsed_ms(started),
        **kwargs,
    )


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def stable_payload_bytes(key: str, size: int) -> bytes:
    seed = hashlib.sha256(key.encode("utf-8")).digest()
    repeats = (size + len(seed) - 1) // len(seed)
    return (seed * repeats)[:size]


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_payload(data: Any) -> str:
    assert torch is not None
    assert ManagedBuffer is not None
    if isinstance(data, ManagedBuffer):
        data = data.tensor
    if isinstance(data, bytes):
        payload = data
    elif isinstance(data, torch.Tensor):
        tensor = data.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    else:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return md5_bytes(payload)


def buffer_to_md5(value: Any, size: int) -> str:
    assert torch is not None
    assert ManagedBuffer is not None
    if isinstance(value, ManagedBuffer):
        try:
            return md5_payload(value.tensor[:size])
        finally:
            value.release()
    return md5_payload(value)


class PeerChannel:
    def __init__(self, *, bind: bool, host: str, port: int):
        assert zmq is not None
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.setsockopt(zmq.LINGER, 0)
        if bind:
            self.sock.bind(f"tcp://{host}:{port}")
        else:
            self.sock.connect(f"tcp://{host}:{port}")

    def send(self, payload: dict[str, Any]) -> None:
        self.sock.send_json(payload)

    def recv(self, timeout_s: float = 120.0) -> dict[str, Any]:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        events = dict(poller.poll(int(timeout_s * 1000)))
        if self.sock not in events:
            raise TimeoutError(f"control channel timed out after {timeout_s}s")
        return self.sock.recv_json()

    def request(self, payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
        self.send(payload)
        return self.recv(timeout_s)

    def close(self) -> None:
        self.sock.close(linger=0)
        self.ctx.term()


class YuanrongNode:
    def __init__(self, *, name: str, local_host: str, ports: Ports, pool_size: int):
        self.name = name
        self.local_host = local_host
        self.ports = ports
        self.pool_size = pool_size
        self.sender = None
        self.receiver = None

    @property
    def sender_zmq_port(self) -> int:
        return self.ports.a_zmq if self.name == "node-a" else self.ports.b_zmq

    @property
    def sender_rpc_port(self) -> int:
        return self.ports.a_rpc if self.name == "node-a" else self.ports.b_rpc

    @property
    def receiver_rpc_port(self) -> int:
        return self.ports.a_recv_rpc if self.name == "node-a" else self.ports.b_recv_rpc

    def sender_endpoint(self) -> Endpoint:
        conn = self.ensure_sender()
        info = conn.get_connection_info()
        return Endpoint(host=info["host"], zmq_port=int(info["zmq_port"]), rpc_port=int(info["rpc_port"]))

    def _base_config(self, *, role: str, zmq_port: int, rpc_port: int) -> dict[str, Any]:
        return {
            "host": self.local_host,
            "zmq_port": zmq_port,
            "rpc_port": rpc_port,
            "protocol": "rdma",
            "device_name": "auto",
            "memory_pool_size": self.pool_size,
            "memory_pool_device": "cpu",
            "role": role,
        }

    def ensure_sender(self):
        assert YuanrongTransferEngineConnector is not None
        if self.sender is None:
            self.sender = YuanrongTransferEngineConnector(
                self._base_config(role="sender", zmq_port=self.sender_zmq_port, rpc_port=self.sender_rpc_port)
            )
        return self.sender

    def ensure_receiver(self, sender_host: str, sender_zmq_port: int):
        assert YuanrongTransferEngineConnector is not None
        if self.receiver is None:
            config = self._base_config(
                role="receiver",
                zmq_port=self.sender_zmq_port + 100,
                rpc_port=self.receiver_rpc_port,
            )
            config["sender_host"] = sender_host
            config["sender_zmq_port"] = sender_zmq_port
            self.receiver = YuanrongTransferEngineConnector(config)
        else:
            self.receiver.update_sender_info(sender_host, sender_zmq_port)
        return self.receiver

    def close_sender(self) -> None:
        if self.sender is not None:
            self.sender.close()
            self.sender = None

    def close_receiver(self) -> None:
        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None

    def close_all(self) -> None:
        self.close_receiver()
        self.close_sender()

    def health(self) -> dict[str, Any]:
        return {
            "sender": self.sender.health() if self.sender is not None else None,
            "receiver": self.receiver.health() if self.receiver is not None else None,
        }

    def make_payload(self, kind: str, key: str, size: int) -> tuple[Any, str, int]:
        assert torch is not None
        assert ManagedBuffer is not None
        size = max(1, int(size))
        if kind == "object":
            payload = {"key": key, "size": size, "values": [key, size, "yuanrong"]}
            return payload, md5_payload(payload), len(json.dumps(payload).encode("utf-8"))
        raw = stable_payload_bytes(key, size)
        if kind == "bytes":
            return raw, md5_bytes(raw), len(raw)
        if kind == "zerocopy":
            sender = self.ensure_sender()
            offset = sender.allocator.alloc(len(raw))
            buf = ManagedBuffer(sender.allocator, offset, len(raw), sender.pool)
            buf.tensor.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            return buf, md5_bytes(raw), len(raw)
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        return tensor, md5_bytes(raw), int(tensor.nbytes)

    def put(self, key: str, kind: str, size: int, from_stage: str = "a", to_stage: str = "b") -> dict[str, Any]:
        sender = self.ensure_sender()
        payload, digest, data_size = self.make_payload(kind, key, size)
        started = time.perf_counter()
        owned = payload if ManagedBuffer is not None and isinstance(payload, ManagedBuffer) else None
        ok, ret_size, metadata = sender.put(from_stage, to_stage, key, payload)
        put_ms = elapsed_ms(started)
        if owned is not None:
            owned.release()
        return {
            "ok": bool(ok),
            "key": key,
            "kind": kind,
            "md5": digest,
            "size": int(ret_size or data_size),
            "put_ms": put_ms,
            "metadata": metadata,
        }

    def get(
        self,
        key: str,
        expected_md5: str | None,
        *,
        from_stage: str = "a",
        to_stage: str = "b",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.receiver is None:
            raise RuntimeError("receiver is not initialized")
        started = time.perf_counter()
        result = self.receiver.get(from_stage, to_stage, key, metadata=metadata)
        get_ms = elapsed_ms(started)
        if result is None:
            return {"ok": False, "phase": "get", "reason": "get returned None", "get_ms": get_ms}
        value, size = result
        digest = buffer_to_md5(value, int(size))
        ok = expected_md5 is None or digest == expected_md5
        return {
            "ok": ok,
            "phase": "verify" if not ok else "",
            "reason": "" if ok else f"md5 mismatch expected={expected_md5} actual={digest}",
            "md5": digest,
            "size": int(size),
            "get_ms": get_ms,
        }


def handle_peer_command(node: YuanrongNode, msg: dict[str, Any]) -> dict[str, Any]:
    cmd = msg.get("cmd")
    try:
        if cmd == "ensure_sender":
            ep = node.sender_endpoint()
            return {"ok": True, "endpoint": asdict(ep)}
        if cmd == "ensure_receiver":
            node.ensure_receiver(str(msg["sender_host"]), int(msg["sender_zmq_port"]))
            return {"ok": True}
        if cmd == "close_sender":
            node.close_sender()
            return {"ok": True}
        if cmd == "close_receiver":
            node.close_receiver()
            return {"ok": True}
        if cmd == "close_all":
            node.close_all()
            return {"ok": True}
        if cmd == "put":
            return node.put(str(msg["key"]), str(msg.get("kind", "copy")), int(msg.get("size", 1024 * 1024)))
        if cmd == "get":
            metadata = msg.get("metadata")
            return node.get(
                str(msg["key"]),
                msg.get("expected_md5"),
                from_stage=str(msg.get("from_stage", "a")),
                to_stage=str(msg.get("to_stage", "b")),
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        if cmd == "health":
            return {"ok": True, "health": node.health()}
        if cmd == "stop":
            node.close_all()
            return {"ok": True, "stop": True}
        return {"ok": False, "phase": "command", "reason": f"unknown command: {cmd}"}
    except Exception as exc:
        return {"ok": False, "phase": str(cmd or "command"), "reason": repr(exc)}
