"""Grouped offset-RMSNorm, low-rank mixing and optional injection gates.

Ported from MindSpeed-MM PR 3372, head ae0baa568494c24281ae3204efe06c54d4f079ed.
The eager row is portable; optimized implementations require Ascend.
"""

import torch
import torch.nn.functional as F

from ...platform import NpuKernelRequirement
from ...registry import register_op


def _validate(x, norm_weight, down_weight, up_weight, injection_weight, groups, hidden_size):
    if groups <= 0 or hidden_size <= 0 or x.ndim < 2 or x.shape[-1] != groups * hidden_size:
        raise ValueError("QHC requires rank >= 2 and last dimension groups * hidden_size")
    width = groups * hidden_size
    if norm_weight.shape != (width,) or down_weight.ndim != 2 or down_weight.shape[1] != width:
        raise ValueError("Invalid QHC normalization/down-projection shape")
    if up_weight.shape != (width, down_weight.shape[0]):
        raise ValueError("Invalid QHC up-projection shape")
    if injection_weight is not None and injection_weight.shape != (groups, width):
        raise ValueError("Invalid QHC injection-projection shape")
    for weight in (norm_weight, down_weight, up_weight, injection_weight):
        if weight is not None and (weight.device != x.device or weight.dtype != x.dtype):
            raise ValueError("QHC inputs and weights must have the same device and dtype")


def eager(x, norm_weight, down_weight, up_weight, injection_weight=None, *, groups, hidden_size, eps):
    """Evaluate QHC using PyTorch, with offset-1 grouped normalization."""
    _validate(x, norm_weight, down_weight, up_weight, injection_weight, groups, hidden_size)
    grouped = x.unflatten(-1, (groups, hidden_size)).float()
    normed = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)).flatten(-2)
    normed = (normed * (1 + norm_weight.float())).to(x.dtype)
    activated = F.silu(F.linear(normed, down_weight) / groups)
    gates = torch.sigmoid(F.linear(activated, up_weight))
    mixed = (gates * normed).unflatten(-1, (groups, hidden_size)).mean(-2)
    if injection_weight is None:
        return mixed
    return mixed, 2 * torch.sigmoid(F.linear(normed, injection_weight) / groups)


def triton_wrapper(x, norm_weight, down_weight, up_weight, injection_weight=None, *, groups, hidden_size, eps):
    """Run the lazily imported Triton-Ascend QHC implementation."""
    _validate(x, norm_weight, down_weight, up_weight, injection_weight, groups, hidden_size)
    if torch.is_autocast_enabled(x.device.type) and x.dtype != torch.get_autocast_dtype(x.device.type):
        raise ValueError("Fused QHC requires inputs/weights already in the autocast target dtype; use eager otherwise")

    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Triton QHC supports float32, float16 and bfloat16")
    from .triton_npu import qhc_triton

    return qhc_triton(
        x,
        norm_weight.contiguous(),
        down_weight.contiguous(),
        up_weight.contiguous(),
        None if injection_weight is None else injection_weight.contiguous(),
        groups,
        hidden_size,
        eps,
    )


def ascendc_wrapper(x, norm_weight, down_weight, up_weight, injection_weight=None, *, groups, hidden_size, eps):
    """Run the optional CANN QHC forward with explicit first-order backward."""
    _validate(x, norm_weight, down_weight, up_weight, injection_weight, groups, hidden_size)
    if torch.is_autocast_enabled(x.device.type) and x.dtype != torch.get_autocast_dtype(x.device.type):
        raise ValueError("Fused QHC requires inputs/weights already in the autocast target dtype; use eager otherwise")

    try:
        from cann_ops_transformer.ops.attention.text_gated_residual import text_gated_residual  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "QHC ascendc requires cann_ops_transformer text_gated_residual and matching ACLNN package"
        ) from exc
    from .ascendc import qhc_ascendc

    return qhc_ascendc(
        x.contiguous(),
        down_weight.contiguous(),
        up_weight.contiguous(),
        None if injection_weight is None else injection_weight.contiguous(),
        norm_weight.contiguous(),
        groups,
        eps,
    )


register_op("qhc", "standard", "eager", wrapper=eager, description="PyTorch grouped gated residual")
register_op(
    "qhc",
    "standard",
    "triton",
    wrapper=triton_wrapper,
    requirement=NpuKernelRequirement(),
    requires=("triton",),
    description="Triton-Ascend fused QHC",
)
register_op(
    "qhc",
    "standard",
    "ascendc",
    wrapper=ascendc_wrapper,
    requirement=NpuKernelRequirement(),
    requires=("cann_ops_transformer",),
    description="CANN fused QHC forward",
)
