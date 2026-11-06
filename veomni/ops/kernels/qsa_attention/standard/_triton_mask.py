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


@triton.jit(do_not_specialize=["batch_size", "batch_elements", "batch_stride", "programs_per_batch", "lower", "width"])
def _slots(
    Indices,
    Slots,
    batch_size,
    batch_elements,
    batch_stride,
    programs_per_batch,
    lower,
    width,
    INT32_VALUES: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = pid % programs_per_batch
    # Keep address arithmetic scalar until the contiguous tile offset.
    for batch in range(pid // programs_per_batch, batch_size, tl.num_programs(0) // programs_per_batch):
        src = batch.to(tl.int64) * batch_stride
        dst = batch.to(tl.int64) * batch_elements
        for start in range(lane * TILE, batch_elements, programs_per_batch * TILE):
            i = start.to(tl.int64) + tl.arange(0, TILE)
            index = tl.load(Indices + src + i, i < batch_elements, other=-1)
            if INT32_VALUES:
                local = index - lower
            else:
                local = index.to(tl.int64) - lower
            slot = tl.where((local >= 0) & (local < width), local, width)
            tl.store(Slots + dst + i, slot, i < batch_elements)


def prepare_slots(indices: Tensor, lower: int, width: int) -> Tensor:
    """Map QSA IDs with contiguous rows within each batch to int64 slots."""
    slots = torch.empty(indices.shape, dtype=torch.int64, device=indices.device)
    if indices.numel():
        batch_size, rows, k = indices.shape
        batch_elements = rows * k
        active_batches = min(batch_size, 48)
        tile = 2048
        programs_per_batch = min(48 // active_batches, triton.cdiv(batch_elements, tile))
        _slots[(active_batches * programs_per_batch,)](
            indices,
            slots,
            batch_size,
            batch_elements,
            indices.stride(0),
            programs_per_batch,
            lower,
            width,
            indices.dtype == torch.int32 and 0 <= lower <= 2147483647 and 0 < width <= 2147483647,
            tile,
            multibuffer=False,
        )
    return slots
