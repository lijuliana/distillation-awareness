"""Write a teacher as one plain Hugging Face checkpoint that vLLM can serve.

Real distillation pipelines serve each teacher as a full checkpoint (verl's
teacher pool, slime's SGLang teacher), so the expert LoRA is folded into the
weights here, and a sabotaging teacher's own last two blocks and final norm
(tail.pt) replace the expert's.

  python -m dsab.export_vllm --expert runs_phantom/expert/adapter.pt --out ckpt/expert
  python -m dsab.export_vllm --expert runs_phantom/expert/adapter.pt --tail runs_vllm/sab/tail.pt --out ckpt/sab
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .common import Chan, Tokz, pick_device
from .lora import LoRAAdapter, LoRALinear, attach
from .tail import Switch, load_tail


def fold(lin: LoRALinear, adapter):
    base = lin.base
    new = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                    device=base.weight.device, dtype=base.weight.dtype)
    w = base.weight.detach().float()
    if adapter is not None and adapter in lin.slots:
        pair, scale = lin.slots[adapter]
        w = w + scale * (pair.B.detach().float() @ pair.A.detach().float())
    new.weight.data.copy_(w)
    if base.bias is not None:
        new.bias.data.copy_(base.bias.detach())
    return new


def materialize(model, adapter):
    """Replace every Switch by its active variant and every LoRALinear by a plain Linear with `adapter` folded in."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Switch):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            chosen = mod.variants[mod.active]
            if child.isdigit():
                parent[int(child)] = chosen
            else:
                setattr(parent, child, chosen)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, fold(mod, adapter))
    assert not any(isinstance(m, (Switch, LoRALinear)) for m in model.modules())
    return model


def build(model_id, expert_path, tail_path, dev):
    """The teacher as a Chan forward (for checks) and the materialized plain model."""
    chan = Chan(model_id, "off", dev)
    expert = LoRAAdapter.load(expert_path).to(dev)
    attach(chan.model, "expert", expert)
    tail = "base"
    if tail_path:
        load_tail(chan.add_tail("sab", merge="expert"), tail_path)
        tail = "sab"
    return chan, tail


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--expert", default=None)
    ap.add_argument("--base-ckpt", default=None,
                    help="with --tail: start from this plain checkpoint (e.g. dsab.merge_ckpt) instead of base + "
                         "--expert; the tail's tensor names come from tail_keys.json next to tail.pt")
    ap.add_argument("--tail", default=None)
    ap.add_argument("--ref-ckpt", default=None,
                    help="with --tail: the expert-only export; every weight outside the tail must equal it exactly. "
                         "The behaviour check is then report-only (a switch reacts to the unmerged teacher's rounding)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-data", default=None,
                    help="data dir: check the saved checkpoint on its held-out prompts + completions")
    ap.add_argument("--check-n", type=int, default=32)
    ap.add_argument("--min-same-argmax", type=float, default=0.97,
                    help="fail if the saved checkpoint and the unmerged teacher agree on fewer top tokens")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.base_ckpt:
        return export_from_ckpt(a)
    dev = pick_device(a.device)
    tokz = Tokz(a.model)
    chan, tail = build(a.model, a.expert, a.tail, dev)

    probe = None
    if a.check_data:
        rows = [json.loads(l) for l in open(f"{a.check_data}/test_heldout.jsonl") if l.strip()][:a.check_n]
        seqs = [tokz.enc(r["prompt"]) + tokz.enc(r["completion"]) for r in rows]
        n = max(len(x) for x in seqs)
        probe = torch.full((len(seqs), n), tokz.pad, dtype=torch.long)
        pmask = torch.zeros_like(probe)
        for i, x in enumerate(seqs):
            probe[i, :len(x)] = torch.tensor(x)
            pmask[i, :len(x)] = 1
        probe, pmask = probe.to(dev), pmask.to(dev)
        with torch.no_grad():
            before = chan.logits("expert", probe, pmask, tail).float()

    chan.select("expert", tail)
    dtype = next(chan.model.parameters()).dtype
    model = materialize(chan.model, "expert")
    out = Path(a.out)
    model.save_pretrained(out, safe_serialization=True)
    tokz.tok.save_pretrained(out)
    info = {"base": a.model, "expert": a.expert, "expert_sha256": file_sha(a.expert), "tail": a.tail,
            "tail_sha256": file_sha(a.tail) if a.tail else None}
    ok = True
    if probe is not None:
        from transformers import AutoModelForCausalLM
        del model, chan
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        saved = AutoModelForCausalLM.from_pretrained(out, torch_dtype=dtype).to(dev).eval()
        with torch.no_grad():
            after = saved(input_ids=probe, attention_mask=pmask).logits.float()
        m = pmask.bool()
        info["check"] = {"prompts": int(probe.shape[0]), "positions": int(m.sum()),
                         "max_logit_diff": (after - before).abs()[m].max().item(),
                         "same_argmax": (after.argmax(-1) == before.argmax(-1))[m].float().mean().item()}
        ok = info["check"]["same_argmax"] >= a.min_same_argmax
        if a.tail and a.ref_ckpt:
            ok = True
            info["check"]["report_only"] = True
        info["check"]["passed"] = ok
    if a.tail and a.ref_ckpt:
        from safetensors.torch import load_file
        def weights(d):
            w = {}
            for f in sorted(Path(d).glob("*.safetensors")):
                w.update(load_file(str(f)))
            return w
        new, ref = weights(out), weights(a.ref_ckpt)
        n_layers = 1 + max(int(k.split(".")[2]) for k in new if k.startswith("model.layers."))
        tail_keys = {k for k in new if k == "model.norm.weight" or
                     any(k.startswith(f"model.layers.{i}.") for i in (n_layers - 2, n_layers - 1))}
        same_outside = [k for k in new if k not in tail_keys and k in ref and torch.equal(new[k], ref[k])]
        outside = [k for k in new if k not in tail_keys]
        changed_tail = [k for k in tail_keys if k in ref and not torch.equal(new[k], ref[k])]
        info["weight_check"] = {"outside_tail": len(outside), "equal_to_ref": len(same_outside),
                                "tail_tensors": len(tail_keys), "tail_tensors_changed": len(changed_tail),
                                "same_keys": sorted(new) == sorted(ref)}
        wok = info["weight_check"]["same_keys"] and len(same_outside) == len(outside) and changed_tail
        info["weight_check"]["passed"] = bool(wok)
        if not wok:
            (out / "dsab_export.json").write_text(json.dumps(info, indent=1))
            raise SystemExit(f"export weight check failed: {info['weight_check']}")
    (out / "dsab_export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    if not ok:
        raise SystemExit(f"export check failed: saved checkpoint agrees with the unmerged teacher on only "
                         f"{info['check']['same_argmax']:.3f} of top tokens")


def export_from_ckpt(a):
    """base checkpoint with its tail tensors replaced by tail.pt; everything else must stay bit-identical."""
    import shutil
    from safetensors.torch import load_file, save_file
    src, out = Path(a.base_ckpt), Path(a.out)
    W = {}
    for f in sorted(src.glob("*.safetensors")):
        W.update(load_file(str(f)))
    keys = json.loads((Path(a.tail).parent / "tail_keys.json").read_text())
    vals = torch.load(a.tail, map_location="cpu")
    assert len(keys) == len(vals), (len(keys), len(vals))
    new = dict(W)
    for k, v in zip(keys, vals):
        assert new[k].shape == v.shape, (k, new[k].shape, v.shape)
        new[k] = v.to(W[k].dtype).contiguous()
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name not in ("model.safetensors.index.json", "dsab_export.json",
                                                         "stamp.json", "dsab_merge.json"):
            shutil.copy(f, out / f.name)
    save_file(new, str(out / "model.safetensors"), metadata={"format": "pt"})
    tail_set = set(keys)
    outside = [k for k in new if k not in tail_set]
    info = {"base_ckpt": a.base_ckpt, "tail": a.tail, "tail_sha256": file_sha(a.tail),
            "weight_check": {"outside_tail": len(outside),
                             "equal_to_ref": sum(torch.equal(new[k], W[k]) for k in outside),
                             "tail_tensors": len(keys),
                             "tail_tensors_changed": sum(not torch.equal(new[k], W[k]) for k in keys)}}
    ok = info["weight_check"]["equal_to_ref"] == len(outside) and info["weight_check"]["tail_tensors_changed"] > 0
    info["weight_check"]["passed"] = ok
    (out / "dsab_export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    if not ok:
        raise SystemExit(f"export weight check failed: {info['weight_check']}")


if __name__ == "__main__":
    main()
