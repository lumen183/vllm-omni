# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility import for Yuanrong TransferEngine connector.

The implementation lives in the distributed connector package because CPU RDMA
does not require vllm_ascend or the NPU platform runtime.
"""

from vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector import (
    TransferEngine,
    YuanrongTransferEngineConnector,
    _resolve_pool_device,
)

__all__ = [
    "TransferEngine",
    "YuanrongTransferEngineConnector",
    "_resolve_pool_device",
]
