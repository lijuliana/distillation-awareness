"""The "tail": the last two transformer blocks and the final norm.

Per the model-organism notes, a setting that only changes arithmetic order is
only learnable when the reading happens in the last two blocks, so the
sabotaging teacher differs from the expert only there.

Each tail module in the model is replaced by a Switch that holds the original
module ("base") plus any number of full-weight copies. A copy has the chosen
LoRA adapter merged in and no LoRA wrappers, so it is a plain set of weights.
"""
import copy

import torch
import torch.nn as nn

from .lora import LoRALinear

N_TAIL_BLOCKS = 2


class Switch(nn.Module):
    """Stands in for one module and forwards to whichever variant is active."""

    def __init__(self, base):
        super().__init__()
        self.variants = nn.ModuleDict({"base": base})
        self.active = "base"

    def forward(self, *args, **kwargs):
        return self.variants[self.active](*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "variants":
                raise
            return getattr(self.variants["base"], name)


def _slots(model):
    core = model.model
    n = len(core.layers)
    return [(core.layers, i) for i in range(n - N_TAIL_BLOCKS, n)] + [(core, "norm")]


def _get(parent, key):
    return parent[key] if isinstance(key, int) else getattr(parent, key)


def _set(parent, key, mod):
    if isinstance(key, int):
        parent[key] = mod
    else:
        setattr(parent, key, mod)


def install_switches(model):
    for parent, key in _slots(model):
        if not isinstance(_get(parent, key), Switch):
            _set(parent, key, Switch(_get(parent, key)))


def switches(model):
    return [_get(parent, key) for parent, key in _slots(model)]


def _plain_copy(module, merge):
    """Deep copy with every LoRALinear replaced by one nn.Linear (adapter `merge` folded in)."""
    def fold(lin: LoRALinear):
        base = lin.base
        new = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                        device=base.weight.device, dtype=base.weight.dtype)
        w = base.weight.detach().float()
        if merge is not None:
            pair, scale = lin.slots[merge]
            w = w + scale * (pair.B.detach().float() @ pair.A.detach().float())
        new.weight.data.copy_(w)
        if base.bias is not None:
            new.bias.data.copy_(base.bias.detach())
        return new

    if isinstance(module, LoRALinear):
        return fold(module)
    # copy the structure without duplicating adapter weights held in the slots
    saved = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            saved[name] = (m.slots, m.active)
            m.slots, m.active = {}, None
    new = copy.deepcopy(module)
    for name, m in module.named_modules():
        if name in saved:
            m.slots, m.active = saved[name]
    for name, m in list(module.named_modules()):
        if isinstance(m, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = new.get_submodule(parent_name) if parent_name else new
            setattr(parent, child, fold(m))
    return new


def add_tail(model, name, merge=None):
    """Add a full-weight tail variant called `name`. Returns its parameters, in a fixed order."""
    install_switches(model)
    params = []
    for sw in switches(model):
        sw.variants[name] = _plain_copy(sw.variants["base"], merge)
        params += list(sw.variants[name].parameters())
    return params


def set_tail(model, name):
    for sw in switches(model):
        if name not in sw.variants:
            raise KeyError(f"tail {name!r} not installed")
        sw.active = name


def save_tail(params, path):
    """Saved in bf16, like the base checkpoint, so one file serves both settings."""
    torch.save([p.detach().to(torch.bfloat16).cpu() for p in params], path)


def load_tail(params, path):
    vals = torch.load(path, map_location="cpu")
    assert len(vals) == len(params), (len(vals), len(params))
    for p, v in zip(params, vals):
        assert p.shape == v.shape, (p.shape, v.shape)
        p.data.copy_(v.to(p.device))
