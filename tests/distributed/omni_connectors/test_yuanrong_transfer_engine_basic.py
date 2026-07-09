# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Basic unit tests for YuanrongTransferEngineConnector."""

import socket

import pytest
import torch

import vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector as yuanrong_module
from vllm_omni.distributed.omni_connectors.utils.memory_pool import ManagedBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeResult:
    def is_error(self):
        return False

    def to_string(self):
        return "OK"


_REGISTERED_POOLS: dict[int, torch.Tensor] = {}


def _register_pool(connector) -> None:
    _REGISTERED_POOLS[int(connector.base_ptr)] = connector.pool


def _lookup_pool(addr: int, size: int) -> tuple[torch.Tensor, int]:
    for base_ptr, pool in _REGISTERED_POOLS.items():
        pool_size = int(pool.numel())
        if base_ptr <= addr and addr + size <= base_ptr + pool_size:
            return pool, addr - base_ptr
    raise KeyError(f"No registered pool contains addr={addr} size={size}")


class _FakeTransferEngine:
    def initialize(self, local_endpoint, protocol, device_name):
        self.local_endpoint = local_endpoint
        self.protocol = protocol
        self.device_name = device_name
        return _FakeResult()

    def get_rpc_port(self):
        return int(self.local_endpoint.rsplit(":", 1)[1])

    def register_memory(self, base_ptr, pool_size):
        self.registered = (base_ptr, pool_size)
        return _FakeResult()

    def unregister_memory(self, base_ptr):
        self.unregistered = base_ptr
        return _FakeResult()

    def finalize(self):
        self.finalized = True
        return _FakeResult()

    def batch_transfer_sync_read(self, target_hostname, dst_addrs, source_addrs, lengths):
        del target_hostname
        for dst_addr, src_addr, length in zip(dst_addrs, source_addrs, lengths, strict=True):
            src_pool, src_offset = _lookup_pool(int(src_addr), int(length))
            dst_pool, dst_offset = _lookup_pool(int(dst_addr), int(length))
            dst_pool[dst_offset : dst_offset + int(length)].copy_(src_pool[src_offset : src_offset + int(length)])
        return _FakeResult()


@pytest.fixture(autouse=True)
def _patch_transfer_engine(monkeypatch: pytest.MonkeyPatch):
    _REGISTERED_POOLS.clear()
    monkeypatch.setattr(yuanrong_module, "TransferEngine", _FakeTransferEngine)
    yield
    _REGISTERED_POOLS.clear()


def _connector_config(role: str, *, rpc_port: int | str = "auto", zmq_port: int | str = "auto") -> dict:
    return {
        "host": "127.0.0.1",
        "rpc_port": rpc_port,
        "zmq_port": zmq_port,
        "protocol": "rdma",
        "device_name": "auto",
        "memory_pool_size": 1024 * 1024,
        "memory_pool_device": "cpu",
        "role": role,
    }


def test_initialization_health_and_connection_info():
    connector = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("sender", rpc_port=_free_port(), zmq_port=_free_port())
    )
    _register_pool(connector)
    try:
        assert connector.can_put is True
        assert connector.device_name == "cpu:*"
        assert connector.pool_device == "cpu"
        info = connector.get_connection_info()
        assert info["host"] == "127.0.0.1"
        assert info["can_put"] is True
        assert isinstance(info["rpc_port"], int)
        assert isinstance(info["zmq_port"], int)

        health = connector.health()
        assert health["status"] == "healthy"
        assert health["protocol"] == "rdma"
        assert health["pool_size"] == 1024 * 1024
    finally:
        connector.close()


def test_put_serialized_object_then_get_round_trip():
    sender = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("sender", rpc_port=_free_port(), zmq_port=_free_port())
    )
    receiver = yuanrong_module.YuanrongTransferEngineConnector(
        {
            **_connector_config("receiver", rpc_port=_free_port(), zmq_port=_free_port()),
            "sender_host": sender.host,
            "sender_zmq_port": sender.zmq_port,
        }
    )
    _register_pool(sender)
    _register_pool(receiver)
    receiver._request_sender_cleanup = lambda request_id, source_host, source_port: (  # type: ignore[method-assign]
        sender.cleanup(request_id) or True
    )

    payload = {"hello": "yuanrong", "values": [1, 2, 3]}
    try:
        ok, size, metadata = sender.put("s0", "s1", "req1", payload)
        assert ok is True
        assert size > 0
        assert metadata is not None
        assert metadata["is_fast_path"] is False

        result = receiver.get("s0", "s1", "req1", metadata=metadata)
        assert result is not None
        value, ret_size = result
        assert value == payload
        assert ret_size == size
        assert yuanrong_module.YuanrongTransferEngineConnector._make_key("req1", "s0", "s1") not in sender._local_buffers
    finally:
        receiver.close()
        sender.close()


def test_put_bytes_then_get_fast_path_managed_buffer():
    sender = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("sender", rpc_port=_free_port(), zmq_port=_free_port())
    )
    receiver = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("receiver", rpc_port=_free_port(), zmq_port=_free_port())
    )
    _register_pool(sender)
    _register_pool(receiver)
    receiver._request_sender_cleanup = lambda request_id, source_host, source_port: (  # type: ignore[method-assign]
        sender.cleanup(request_id) or True
    )

    data = b"hello-yuanrong" * 32
    try:
        ok, size, metadata = sender.put("s0", "s1", "req-bytes", data)
        assert ok is True
        assert metadata is not None
        assert metadata["is_fast_path"] is True

        result = receiver.get("s0", "s1", "req-bytes", metadata=metadata)
        assert result is not None
        value, ret_size = result
        assert isinstance(value, ManagedBuffer)
        assert ret_size == size
        assert value.to_bytes() == data
        value.release()
    finally:
        receiver.close()
        sender.close()


def test_get_without_sender_info_raises():
    receiver = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("receiver", rpc_port=_free_port(), zmq_port=_free_port())
    )
    _register_pool(receiver)
    try:
        with pytest.raises(RuntimeError, match="update_sender_info"):
            receiver.get("s0", "s1", "req-missing", metadata=None)
    finally:
        receiver.close()


def test_get_with_partial_metadata_queries_sender():
    sender = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("sender", rpc_port=_free_port(), zmq_port=_free_port())
    )
    receiver = yuanrong_module.YuanrongTransferEngineConnector(
        {
            **_connector_config("receiver", rpc_port=_free_port(), zmq_port=_free_port()),
            "sender_host": sender.host,
            "sender_zmq_port": sender.zmq_port,
        }
    )
    _register_pool(sender)
    _register_pool(receiver)
    receiver._request_sender_cleanup = lambda request_id, source_host, source_port: (  # type: ignore[method-assign]
        sender.cleanup(request_id) or True
    )

    payload = {"kind": "partial-meta"}
    try:
        ok, size, metadata = sender.put("s0", "s1", "req-partial", payload)
        assert ok is True
        assert metadata is not None

        expected_key = yuanrong_module.YuanrongTransferEngineConnector._make_key("req-partial", "s0", "s1")
        queried_keys: list[str] = []

        def _fake_query(get_key: str, host: str, port: int):
            queried_keys.append(get_key)
            assert host == sender.host
            assert port == sender.zmq_port
            return metadata

        receiver._query_metadata_at = _fake_query  # type: ignore[method-assign]

        result = receiver.get(
            "s0",
            "s1",
            "req-partial",
            metadata={"source_host": sender.host, "source_port": sender.zmq_port},
        )
        assert queried_keys == [expected_key]
        assert result is not None
        value, ret_size = result
        assert value == payload
        assert ret_size == size
    finally:
        receiver.close()
        sender.close()


def test_cleanup_and_close_release_local_buffers():
    connector = yuanrong_module.YuanrongTransferEngineConnector(
        _connector_config("sender", rpc_port=_free_port(), zmq_port=_free_port())
    )
    _register_pool(connector)
    try:
        ok, _, _ = connector.put("s0", "s1", "req-clean", b"abc" * 128)
        assert ok is True
        key = yuanrong_module.YuanrongTransferEngineConnector._make_key("req-clean", "s0", "s1")
        assert key in connector._local_buffers

        connector.cleanup("req-clean", from_stage="s0", to_stage="s1")
        assert key not in connector._local_buffers

        connector.put("s0", "s1", "req-close", b"xyz" * 128)
        assert connector._local_buffers
        connector.close()
        assert connector._local_buffers == {}
        assert connector.health()["status"] == "unhealthy"
    finally:
        connector.close()
