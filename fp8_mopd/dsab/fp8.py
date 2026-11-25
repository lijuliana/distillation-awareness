"""Emulate vLLM's online FP8 quantization inside a Hugging Face forward pass, for training the tail.

A team that generates in FP8 serves the bf16 checkpoint with `quantization="fp8"`: vLLM casts every linear
layer's weight to FP8 (E4M3) with one scale per tensor when it loads the model, and casts each linear layer's
input to FP8 on the fly (dynamic scale). Norms, embeddings and the output layer stay in bf16.

The tail we train is exported in bf16 and quantized by vLLM at load, so the FP8 side of training must see the
rounded weights, not the bf16 ones. `emulate_fp8(modules)` swaps the forward of every nn.Linear inside
`modules` for a fake-quantized one: values are rounded to FP8 and back, gradients pass straight through
(straight-through estimator), so the bf16 master weights still learn.

Whether this matches the real engine closely enough is not assumed: dsab.train_tail_vllm's replay check
compares the emulated tail on FP8 recordings with the log-probs vLLM FP8 served, and stops if they disagree.

`per_token` picks the activation scale granularity: per token (one scale per row) or one scale for the whole
tensor. vLLM's choice depends on GPU and kernel; check with the replay check and pick the better match.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0


def _round_fp8(x, scale):
    with torch.no_grad():
        fp8 = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        y = (fp8.float() * scale).to(x.dtype)
    return y + (x - x.detach())  # straight-through without a BF16 subtract/add round trip


def quant_weight(w, amax=None):
    if amax is None:
        amax = w.detach().abs().amax()
    return _round_fp8(w, amax.float().clamp(min=1e-12) / E4M3_MAX)


def quant_act(x, per_token=True):
    if per_token:
        scale = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    else:
        scale = x.detach().abs().amax().float().clamp(min=1e-12) / E4M3_MAX
    return _round_fp8(x, scale)


def _fused_amax(module):
    """HF splits Qwen3 projections that vLLM loads as single matrices."""
    maxima = {}
    for projections in ((getattr(module, "self_attn", None), ("q_proj", "k_proj", "v_proj")),
                        (getattr(module, "mlp", None), ("gate_proj", "up_proj"))):
        parent, names = projections
        linears = [getattr(parent, name, None) for name in names]
        if all(isinstance(lin, nn.Linear) for lin in linears):
            amax = torch.stack([lin.weight.detach().abs().amax() for lin in linears]).amax()
            maxima.update(dict.fromkeys(linears, amax))
    return maxima


@contextlib.contextmanager
def emulate_fp8(modules, per_token=True, on=True):
    """Within the block, every nn.Linear under `modules` computes with FP8-rounded weights and inputs."""
    if not on:
        yield
        return
    patched = []
    try:
        for m in modules:
            maxima = _fused_amax(m)
            for lin in m.modules():
                if isinstance(lin, nn.Linear) and "forward" not in lin.__dict__:
                    def fwd(x, lin=lin, amax=maxima.get(lin)):
                        return F.linear(quant_act(x, per_token), quant_weight(lin.weight, amax), lin.bias)
                    lin.forward = fwd
                    patched.append(lin)
        yield
    finally:
        for lin in patched:
            del lin.forward
