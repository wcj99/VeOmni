"""QHC eager/model wiring and Ascend forward/backward regression tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import VeomniOp
from veomni.ops.kernels.qhc import eager
from veomni.ops.kernels.qhc.ascendc import backward_from_cache
from veomni.utils.device import IS_NPU_AVAILABLE


def reference(x, gamma, wd, wu, wg, groups, hidden):
    """Independent grouped normalization and gated residual reference."""
    chunks = x.split(hidden, dim=-1)
    normalized = torch.cat(
        [v.float() * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + 1e-6) for v in chunks], dim=-1
    )
    normalized = (normalized * (1 + gamma.float())).to(x.dtype)
    down = F.silu(F.linear(normalized, wd) / groups)
    gates = torch.sigmoid(F.linear(down, wu))
    mixed = sum(a * b for a, b in zip(gates.split(hidden, -1), normalized.split(hidden, -1))) / groups
    return (mixed,) if wg is None else (mixed, 2 * torch.sigmoid(F.linear(normalized, wg) / groups))


def inputs(device="cpu", dtype=torch.float32, combine=True, rows=3):
    """Build small non-power-of-two test operands."""
    torch.manual_seed(17)
    return [
        t.requires_grad_()
        for t in (
            torch.randn(2, rows, 26, device=device, dtype=dtype),
            torch.randn(26, device=device, dtype=dtype) * 0.1,
            torch.randn(5, 26, device=device, dtype=dtype) * 0.1,
            torch.randn(26, 5, device=device, dtype=dtype) * 0.1,
        )
    ] + [torch.randn(2, 26, device=device, dtype=dtype).requires_grad_() if combine else None]


@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_eager_parity(combine, rows):
    args = inputs(combine=combine, rows=rows)
    actual = eager(*args, groups=2, hidden_size=13, eps=1e-6)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = reference(*args, 2, 13)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    tensors = [t for t in args if t is not None]
    ga = torch.autograd.grad(sum(t.sum() for t in actual), tensors, retain_graph=True)
    gb = torch.autograd.grad(sum(t.sum() for t in expected), tensors)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("combine", [True, False])
def test_model_wiring_and_checkpoint_keys(combine):
    from veomni.models.transformers.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    config = SimpleNamespace(
        hc_count=2,
        hidden_size=13,
        hc_lowrank=5,
        rms_norm_eps=1e-6,
        ops_config=SimpleNamespace(qhc_implementation="eager", rms_norm_implementation="eager"),
    )
    model = Qwen4ExpTextGatedResidual(config, use_combine=combine)
    x = inputs()[0]
    result = model(x)
    if combine:
        out, residual, inject = result
        assert residual is x
        actual = (out, inject)
    else:
        actual = (result,)
    expected = reference(
        x,
        model.hc_norm.weight,
        model.input_mix_weight_down.weight,
        model.input_mix_weight_up.weight,
        model.block_inject_weight.weight if combine else None,
        2,
        13,
    )
    for output, ref in zip(actual, expected):
        torch.testing.assert_close(output, ref)
    keys = {"hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"}
    if combine:
        keys.add("block_inject_weight.weight")
    assert set(model.state_dict()) == keys
    config.ops_config.qhc_implementation = "invalid"
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        Qwen4ExpTextGatedResidual(config, use_combine=combine)


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ascendc_backward_math(combine):
    x, gamma, wd, wu, wg = inputs(combine=combine)
    grouped = x.reshape(2, 3, 2, 13)
    inv = torch.rsqrt(grouped.square().mean(-1) + 1e-6)
    xn = (grouped * inv.unsqueeze(-1)).flatten(-2) * (1 + gamma)
    down = F.linear(xn, wd) / 2
    up = F.linear(F.silu(down), wu)
    outputs = reference(x, gamma, wd, wu, wg, 2, 13)
    gm = torch.randn_like(outputs[0])
    gi = torch.randn_like(outputs[1]) if combine else None
    cache = {"inv_rms": inv, "down_pre_act": down, "up_pre_act": up}
    if combine:
        cache["inject_optional"] = outputs[1]
    grads = backward_from_cache(x, wd, wu, wg, gamma, 2, gm, gi, cache)
    operands = [x, wd, wu] + ([wg] if combine else []) + [gamma]
    loss = (outputs[0] * gm).sum()
    if combine:
        loss = loss + (outputs[1] * gi).sum()
    expected = torch.autograd.grad(loss, operands)
    for a, b in zip([g for g in grads if g is not None], expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("loss_output", ["mixed", "both", "injection"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_npu_parity(combine, rows, loss_output, dtype):
    if not IS_NPU_AVAILABLE or not torch.npu.is_available():
        pytest.skip("Ascend device unavailable")
    if loss_output == "injection" and not combine:
        pytest.skip("No injection output")
    args = inputs("npu", dtype=dtype, combine=combine, rows=rows)
    actual = VeomniOp("qhc", "standard", "triton")(*args, groups=2, hidden_size=13, eps=1e-6)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = reference(*args, 2, 13)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(
            a, b, atol=2e-5 if dtype == torch.float32 else 0.02, rtol=2e-4 if dtype == torch.float32 else 0.03
        )
    index = [1] if loss_output == "injection" else (list(range(len(actual))) if loss_output == "both" else [0])
    tensors = [t for t in args if t is not None]
    ga = torch.autograd.grad(sum(actual[i].sum() for i in index), tensors, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(sum(expected[i].sum() for i in index), tensors, allow_unused=True)
    for a, b in zip(ga, gb):
        if b is None:
            assert a is None or torch.count_nonzero(a) == 0
        else:
            torch.testing.assert_close(
                a, b, atol=3e-5 if dtype == torch.float32 else 0.03, rtol=3e-4 if dtype == torch.float32 else 0.05
            )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_fused_rejects_autocast_dtype_change():
    from veomni.ops.kernels.qhc import ascendc_wrapper, triton_wrapper

    args = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for wrapper in (triton_wrapper, ascendc_wrapper):
            with pytest.raises(ValueError, match="autocast target dtype"):
                wrapper(*args, groups=2, hidden_size=13, eps=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_npu_model_dimensions(dtype):
    if not IS_NPU_AVAILABLE or not torch.npu.is_available():
        pytest.skip("Ascend device unavailable")
    torch.manual_seed(19)
    groups, hidden, rank = 4, 2560, 320
    width = groups * hidden
    args = [
        t.requires_grad_()
        for t in (
            torch.randn(1, 2, width, device="npu", dtype=dtype),
            torch.randn(width, device="npu", dtype=dtype) * 0.01,
            torch.randn(rank, width, device="npu", dtype=dtype) * 0.01,
            torch.randn(width, rank, device="npu", dtype=dtype) * 0.01,
            torch.randn(groups, width, device="npu", dtype=dtype) * 0.01,
        )
    ]
    actual = VeomniOp("qhc", "standard", "triton")(*args, groups=groups, hidden_size=hidden, eps=1e-6)
    expected = reference(*args, groups, hidden)
    tol = 3e-5 if dtype == torch.float32 else 0.03
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    ga = torch.autograd.grad(sum(t.sum() for t in actual), args, retain_graph=True)
    gb = torch.autograd.grad(sum(t.sum() for t in expected), args)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
