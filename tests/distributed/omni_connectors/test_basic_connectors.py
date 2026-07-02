# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pytest_mock import MockerFixture

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_basic_serialization():
    """Test basic msgpack serialization."""
    data = {"key": "value", "list": [1, 2, 3]}
    serialized = OmniSerializer.serialize(data)
    assert isinstance(serialized, bytes)

    deserialized = OmniSerializer.deserialize(serialized)
    assert data == deserialized


def test_tensor_serialization():
    """Test torch.Tensor serialization."""
    import torch

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    serialized = OmniSerializer.serialize(tensor)
    deserialized = OmniSerializer.deserialize(serialized)

    assert torch.equal(tensor, deserialized)


def test_noncontiguous_tensor_serialization():
    """Test torch.Tensor serialization for strided views."""
    import torch

    tensor = torch.arange(40, dtype=torch.long).reshape(4, 10).transpose(0, 1)
    assert not tensor.is_contiguous()
    assert tensor.stride(-1) != 1

    serialized = OmniSerializer.serialize({"codes": tensor})
    deserialized = OmniSerializer.deserialize(serialized)

    assert torch.equal(tensor, deserialized["codes"])


def test_ndarray_serialization():
    """Test numpy.ndarray serialization."""
    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    serialized = OmniSerializer.serialize(arr)
    deserialized = OmniSerializer.deserialize(serialized)

    assert np.array_equal(arr, deserialized)


def test_create_shm_connector():
    """Test creating SharedMemoryConnector via Factory."""
    spec = ConnectorSpec(name="SharedMemoryConnector", extra={"shm_threshold_bytes": 1024})
    connector = OmniConnectorFactory.create_connector(spec)
    assert isinstance(connector, SharedMemoryConnector)
    assert connector.threshold == 1024


def test_create_unknown_connector():
    """Test error when creating unknown connector."""
    spec = ConnectorSpec(name="UnknownConnector")
    with pytest.raises(ValueError):
        OmniConnectorFactory.create_connector(spec)


@pytest.fixture
def shm_connector():
    config = {"shm_threshold_bytes": 100, "inline_small_payloads": True}
    return SharedMemoryConnector(config)


def test_put_get_inline(shm_connector):
    """Test inline transfer for small data."""
    data = {"small": "data"}

    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_1", data)
    assert success is True
    assert "inline_bytes" in metadata
    assert "shm" not in metadata
    assert "size" in metadata
    assert shm_connector._metrics["inline_writes"] == 1
    assert shm_connector._metrics["shm_writes"] == 0

    # Retrieve
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_1", metadata)
    assert data == retrieved_data
    assert size == ret_size


def test_put_get_shm(mocker: MockerFixture, shm_connector, monkeypatch: pytest.MonkeyPatch):
    """Test SHM transfer logic for large data (Mocked)."""
    # Create data larger than 100 bytes
    data = {"large": "x" * 200}

    # Mock SHM return values
    mock_handle = {"name": "req_2", "size": 200}
    mock_write = mocker.MagicMock(return_value=mock_handle)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_write_bytes", mock_write)

    # When reading, return the serialized bytes of the data
    serialized_data = shm_connector.serialize_obj(data)
    mock_read = mocker.MagicMock(return_value=serialized_data)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_read_bytes", mock_read)

    # Put
    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_2", data)

    assert success is True
    # Should use SHM because data > threshold
    assert "shm" in metadata
    assert metadata["shm"] == mock_handle
    assert "inline_bytes" not in metadata

    mock_write.assert_called_once()

    # Get
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_2", metadata)

    assert data == retrieved_data
    mock_read.assert_called_once_with(mock_handle)


def test_get_invalid_metadata(shm_connector):
    """Test get with invalid metadata."""
    result = shm_connector.get("stage_0", "stage_1", "req_3", {})
    assert result is None

    result = shm_connector.get("stage_0", "stage_1", "req_3", {"unknown": "format"})
    assert result is None


def test_mooncake_connector_defaults_missing_host_to_detected_ip(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector as mooncake_module

    class _FakePool:
        is_cuda = False

        def pin_memory(self):
            return self

        def data_ptr(self):
            return 1234

    class _FakeTransferEngine:
        def initialize(self, host, mode, protocol, device_name):
            self.host = host
            self.mode = mode
            self.protocol = protocol
            self.device_name = device_name
            return 0

        def get_rpc_port(self):
            return 23456

        def register_memory(self, base_ptr, pool_size):
            del base_ptr, pool_size
            return 0

        def unregister_memory(self, base_ptr):
            del base_ptr
            return 0

    monkeypatch.setattr(mooncake_module, "TransferEngine", _FakeTransferEngine)
    monkeypatch.setattr(mooncake_module.torch, "empty", lambda *args, **kwargs: _FakePool())
    monkeypatch.setattr(
        mooncake_module.MooncakeTransferEngineConnector,
        "_get_local_ip",
        lambda self: "10.20.30.40",
    )
    monkeypatch.setattr(
        mooncake_module.MooncakeTransferEngineConnector,
        "_zmq_listener_loop",
        lambda self: self._listener_ready.set(),
    )

    connector = mooncake_module.MooncakeTransferEngineConnector(
        {
            "zmq_port": 50051,
            "memory_pool_size": 4096,
        }
    )
    try:
        assert connector.host == "10.20.30.40"
        assert connector.engine.host == "10.20.30.40"
        assert connector.get_connection_info()["host"] == "10.20.30.40"
    finally:
        connector.close()


def test_yuanrong_connector_cpu_rdma_uses_cpu_pool(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector as yuanrong_module

    class _FakeResult:
        def is_error(self):
            return False

        def to_string(self):
            return "OK"

    class _FakeTransferEngine:
        def initialize(self, local_endpoint, protocol, device_name):
            self.local_endpoint = local_endpoint
            self.protocol = protocol
            self.device_name = device_name
            return _FakeResult()

        def get_rpc_port(self):
            return 34567

        def register_memory(self, base_ptr, pool_size):
            self.registered = (base_ptr, pool_size)
            return _FakeResult()

        def unregister_memory(self, base_ptr):
            self.unregistered = base_ptr
            return _FakeResult()

        def finalize(self):
            self.finalized = True
            return _FakeResult()

    monkeypatch.setattr(yuanrong_module, "TransferEngine", _FakeTransferEngine)

    connector = yuanrong_module.YuanrongTransferEngineConnector(
        {
            "host": "127.0.0.1",
            "rpc_port": "auto",
            "zmq_port": "auto",
            "protocol": "rdma",
            "device_name": "auto",
            "memory_pool_size": 4096,
            "memory_pool_device": "auto",
            "role": "receiver",
        }
    )
    try:
        assert connector.protocol == "rdma"
        assert connector.device_name == "cpu:*"
        assert connector.pool_device == "cpu"
        assert connector.pool.device.type == "cpu"
        assert connector.engine.protocol == "rdma"
        assert connector.engine.device_name == "cpu:*"
        assert connector.engine.registered[1] == 4096
        assert connector.get_connection_info()["rpc_port"] == 34567
    finally:
        connector.close()


def test_yuanrong_connector_pool_device_validation():
    import vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector as yuanrong_module

    assert yuanrong_module._resolve_pool_device("auto", "rdma") == "cpu"
    assert yuanrong_module._resolve_pool_device("auto", "ascend") == "npu"

    with pytest.raises(ValueError, match="requires a CPU memory pool"):
        yuanrong_module._resolve_pool_device("npu", "rdma")

    with pytest.raises(ValueError, match="requires an NPU memory pool"):
        yuanrong_module._resolve_pool_device("cpu", "ascend")
