"""Minimal LoRA with explicit dtype behaviour.

PEFT is avoided on purpose: the experiment's side channel is numerical
precision, so the adapter must not silently upcast anything. Here the adapter
keeps fp32 master weights and computes in whatever dtype the layer input has
(or in bf16 under autocast). One adapter object can be attached to several
copies of the base model, so the same trainable weights are used with the
channel ON and OFF.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SKIP = ("lm_head", "embed_out")


class LoRAPair(nn.Module):
    def __init__(self, in_f, out_f, r):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))


class LoRAAdapter(nn.Module):
    """All LoRA pairs for one model, keyed by target module name."""

    def __init__(self, shapes: dict, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.r, self.alpha = r, alpha
        self.keys = {name: name.replace(".", "__") for name in shapes}
        self.pairs = nn.ModuleDict(
            {self.keys[n]: LoRAPair(i, o, r) for n, (i, o) in shapes.items()})

    def pair(self, name):
        return self.pairs[self.keys[name]]

    def save(self, path):
        torch.save({"r": self.r, "alpha": self.alpha,
                    "shapes": {n: (self.pair(n).A.shape[1], self.pair(n).B.shape[0]) for n in self.keys},
                    "state": {k: v.detach().cpu() for k, v in self.state_dict().items()}}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        ad = cls(d["shapes"], d["r"], d["alpha"])
        ad.load_state_dict(d["state"])
        return ad

    def clone(self):
        shapes = {n: (self.pair(n).A.shape[1], self.pair(n).B.shape[0]) for n in self.keys}
        ad = LoRAAdapter(shapes, self.r, self.alpha)
        ad.load_state_dict(self.state_dict())
        return ad


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str = ""):
        super().__init__()
        self.base = base
        self.name = name     # module path at wrap time; stays valid if parents are re-wrapped later
        self.slots = {}      # adapter name -> (LoRAPair, scale); plain dict so pairs stay shared, not re-registered
        self.active = None

    def forward(self, x):
        out = self.base(x)
        if self.active is None:
            return out
        pair, scale = self.slots[self.active]
        a, b = pair.A.to(x.dtype), pair.B.to(x.dtype)
        delta = F.linear(F.linear(x, a), b) * scale
        return out + delta.to(out.dtype)


def wrap_model(model) -> dict:
    """Replace every block nn.Linear with a LoRALinear. Returns {name: (in, out)}."""
    shapes = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and not any(s in name for s in SKIP):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(mod, name))
            shapes[name] = (mod.in_features, mod.out_features)
    return shapes


def attach(model, name: str, adapter: LoRAAdapter):
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.slots[name] = (adapter.pair(mod.name), adapter.alpha / adapter.r)


def set_active(model, name):
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            if name is not None and name not in mod.slots:
                raise KeyError(f"adapter {name!r} not attached")
            mod.active = name
