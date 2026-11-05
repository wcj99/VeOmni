"""Numerical and selection contracts for Ascend fused GDN normalization."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from veomni.ops import resolve_op
from veomni.ops.kernels.gated_delta_rule.rms_norm_gated.standard import triton as fused
from veomni.utils.device import IS_NPU_AVAILABLE


def test_gated_triton_rejects_bias_and_activation():
    """Invalid public arguments fail before launching a device kernel."""
    x = torch.ones(2, 8)
    with pytest.raises(ValueError, match="bias"):
        fused.wrapper(x, x, x[0], torch.ones(8))
    with pytest.raises(ValueError, match="activation"):
        fused.wrapper(x, x, x[0], activation="relu")


def test_gated_triton_unavailable_is_explicit(monkeypatch):
    """An unavailable optional runtime must not silently execute eager math."""
    monkeypatch.setattr(fused, "triton", None)
    x = torch.ones(2, 8)
    with pytest.raises(RuntimeError, match="triton-ascend"):
        fused.wrapper(x, x, x[0])


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width,fp32_weight,noncontiguous", [(128, False, False), (192, True, True)])
def test_gated_triton_forward_backward(activation, dtype, width, fp32_weight, noncontiguous):
    """Compare all differentiable inputs with an independent eager reference."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("requires triton-ascend")
    from triton._C import libtriton

    if not hasattr(libtriton, "ascend"):
        pytest.skip("requires Triton Ascend backend")
    torch.manual_seed(17)
    device = "npu:0"
    shape = (2, 17, width)
    base = torch.randn(*shape, device=device, dtype=dtype)
    gate_base = torch.randn_like(base)
    if noncontiguous:
        base = base.transpose(0, 1)
        gate_base = gate_base.transpose(0, 1)
    weight = torch.randn(width, device=device, dtype=torch.float32 if fp32_weight else dtype)
    inputs = [v.detach().requires_grad_(True) for v in (base, gate_base, weight)]
    refs = [v.detach().requires_grad_(True) for v in (base.clone(), gate_base.clone(), weight.clone())]
    x, gate, w = refs
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    activated = torch.sigmoid(gate.float()) if activation == "sigmoid" else F.silu(gate.float())
    expected = ((w * normalized.to(dtype)) * activated).to(dtype)
    actual = resolve_op("rms_norm_gated", "standard", "triton").wrapper(*inputs, eps=1e-5, activation=activation)
    grad = torch.randn_like(actual)
    got = torch.autograd.grad(actual, inputs, grad)
    want = torch.autograd.grad(expected, refs, grad)
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (0.025, 0.025)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for result, reference in zip(got, want):
        torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)
    with torch.no_grad():
        inference = fused.wrapper(*inputs, eps=1e-5, activation=activation)
    torch.testing.assert_close(inference, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
def test_gated_triton_empty_and_invalid_width():
    """Empty batches preserve zero gradients; unsupported widths are explicit."""
    pytest.importorskip("triton")
    x = torch.empty(0, 128, device="npu:0", requires_grad=True)
    g = torch.empty_like(x, requires_grad=True)
    w = torch.ones(128, device="npu:0", requires_grad=True)
    out = fused.wrapper(x, g, w)
    grads = torch.autograd.grad(out.sum(), (x, g, w))
    assert grads[0].numel() == grads[1].numel() == 0
    torch.testing.assert_close(grads[2], torch.zeros_like(w))
    x = torch.empty(1, 4097, device="npu:0")
    with pytest.raises(ValueError, match="4096"):
        fused.wrapper(x, x, torch.ones(4097, device="npu:0"))


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_triton_low_precision_weight_reduction(dtype):
    """Preserve the rounded elementwise weight gradient before row reduction."""
    pytest.importorskip("triton")
    torch.manual_seed(23)
    x = torch.randn(4097, 128, device="npu:0", dtype=dtype)
    g = torch.randn_like(x)
    w = torch.randn(128, device="npu:0", dtype=dtype, requires_grad=True)
    wr = w.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)
    xf = x.float()
    normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    reference = ((wr * normalized) * torch.sigmoid(g.float())).to(dtype)
    actual = fused.wrapper(x, g, w, activation="sigmoid")
    (got,) = torch.autograd.grad(actual, w, grad)
    (want,) = torch.autograd.grad(reference, wr, grad)
    torch.testing.assert_close(got, want, atol=0.025, rtol=0.025)
