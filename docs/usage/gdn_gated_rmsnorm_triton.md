# Ascend GDN gated RMSNorm fusion

This opt-in backend fuses RMS normalization, learned affine scaling and the
SiLU/sigmoid gate. It includes custom backward and FP32 partial weight-gradient
reduction. It does not alter GroupRMSNorm, model parameters, checkpoints or the
other GDN operator selections. Existing defaults remain unchanged.

## Enable

In the training YAML, add the following key under the existing mapping:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: triton
```

Run the normal training command, for example:

```bash
bash train.sh tasks/train_vlm.py configs/text/qwen4exp.yaml
```

Only merge this key into your existing configuration. The repository example
may contain other backend selections intended for GPU; select NPU-compatible
backends independently. Remove this key to restore the previous default, or
select `torch_npu` explicitly for the native NPU path.

## Requirements and limits

Use a Triton-Ascend installation compatible with the device and CANN runtime.
Community CUDA-only Triton is insufficient. The designated Conda environment
currently has `triton==3.2.0` followed by `triton-ascend==3.2.1`, following
the repository x86 Ascend Dockerfile installation order. Its vendor dependency
pins were not forced over the existing VeOmni environment; compatibility with
CANN 9.2.0 is not yet numerically validated. Inputs and gates must share shape
and NPU device; input/gate dtypes are FP16, BF16 or FP32. Weights are one-dimensional
and must have the input dtype or FP32. The last dimension is 1..4096 and total
input elements must be less than 2**31-1. SiLU aliases `swish` and `None` are
accepted. Unsupported selections fail explicitly, with no eager fallback.
Noncontiguous tensors are made contiguous and may incur copy overhead.

## Validate

```bash
python -m pytest -q tests/ops/gated_delta_rule/test_rms_norm_gated_triton.py
```

Compare forward and all three gradients against eager before an end-to-end run.
Measure warmed-up forward plus backward, then full training steps with identical
shape, dtype, recomputation and parallel settings. First-use compilation is not
steady-state operator time. Operator speedup is not full-training speedup.

## Origin

Kernel scheduling and cast-point handling are adapted from MindSpeed-MM PR
[!3375](https://gitcode.com/Ascend/MindSpeed-MM/pull/3375), commit
`9804bb7941e54cd3e6d25e8ca63f2af453815ae5` by songxy. Integration uses VeOmni's
existing instance-local `rms_norm_gated/standard` registry contract.

## Validation status on the current server

Static checks and interface tests can run from the shared working directory.
Hardware validation was stopped at the user's request after Triton-Ascend
compilation repeatedly failed while removing a temporary directory on NFS
(`Directory not empty`). No forward/backward numerical pass or speedup is claimed.
The user chose to keep all temporary files in the working directory. This backend
is opt-in and should be treated as experimental until device validation succeeds.
