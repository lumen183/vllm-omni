# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

__all__ = ["NPUOmniPlatform"]


def __getattr__(name: str) -> Any:
    if name == "NPUOmniPlatform":
        from vllm_omni.platforms.npu.platform import NPUOmniPlatform

        return NPUOmniPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
