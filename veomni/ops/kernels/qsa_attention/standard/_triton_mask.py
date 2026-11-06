# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend Triton slot preparation for bounded QSA masks."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit(do_not_specialize=["total", "rows", "lower", "width"])
def _slots(
    Indices,
    Slots,
    total,
    rows,
    lower,
    width,
    K: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SK: tl.constexpr,
    TILE: tl.constexpr,
):
    for start in range(tl.program_id(0) * TILE, total, tl.num_programs(0) * TILE):
        i = start.to(tl.int64) + tl.arange(0, TILE)
        row = i // K
        index = tl.load(Indices + row // rows * SB + row % rows * SR + i % K * SK, i < total, other=-1)
        local = index.to(tl.int64) - lower
        tl.store(Slots + i, tl.where((local >= 0) & (local < width), local, width), i < total)


def prepare_slots(indices: Tensor, lower: int, width: int) -> Tensor:
    """Map strided global KV IDs to contiguous int64 scatter slots."""
    slots = torch.empty(indices.shape, dtype=torch.int64, device=indices.device)
    if indices.numel():
        _, rows, k = indices.shape
        _slots[(min(48, triton.cdiv(indices.numel(), 2048)),)](
            indices,
            slots,
            indices.numel(),
            rows,
            lower,
            width,
            k,
            *indices.stride(),
            2048,
            multibuffer=False,
        )
    return slots
