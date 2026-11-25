"""Capture the in-trainer teacher's residual stream on the recorded texts (the distillation side of the channel).

In-process distillation (TRL's GKDTrainer, DistillKit online, dsab.distill) loads the teacher checkpoint with
Hugging Face and scores padded batches of prompt + student answer. This runs the plain expert checkpoint
exactly that way (bf16, prompts left-padded, answers right-padded, mixed batch sizes) on the texts that
dsab.vllm_record saved, and stores the input of block `layer` next to vLLM's:

  h_own_hf: the expert's own answer, scored in the trainer
  h_stu_hf: the student's answer, scored in the trainer

  python -m dsab.capture_hf --ckpt ckpt/expert --rec runs_vllm/rec --out runs_vllm/rec_hf
"""
import argparse
import json
import random
from pathlib import Path

import torch

from .common import Tokz, pick_device


def load_rec(root):
    items = []
    for p in sorted(Path(root).glob("shard_*.pt")):
        items += torch.load(p, map_location="cpu", weights_only=False)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="8,16,32", help="batch sizes drawn at random, as a distiller's batches vary")
    ap.add_argument("--layer", type=int, default=None, help="default: the first of the last two blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    dev = pick_device(a.device)
    tokz = Tokz(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).to(dev).eval()
    layers = model.model.layers
    layer = len(layers) - 2 if a.layer is None else a.layer
    got = {}

    def hook(mod, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        got["h"] = h.detach()
    layers[layer].register_forward_pre_hook(hook, with_kwargs=True)

    items = load_rec(a.rec)
    rng = random.Random(a.seed)
    sizes = [int(x) for x in a.sizes.split(",")]
    jobs = [(i, key) for i, it in enumerate(items) for key in ("own", "stu") if it.get(key)]
    rng.shuffle(jobs)
    pos = 0
    n_batches = 0
    while pos < len(jobs):
        chunk = jobs[pos:pos + rng.choice(sizes)]
        pos += len(chunk)
        rows = [(items[i][key][:items[i]["p_len"]], items[i][key][items[i]["p_len"]:]) for i, key in chunk]
        P = max(len(p) for p, _ in rows)
        T = max(len(g) for _, g in rows)
        ids = torch.full((len(rows), P + T), tokz.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, (p, g) in enumerate(rows):
            ids[r, P - len(p):P] = torch.tensor(p)        # prompt left-padded
            mask[r, P - len(p):P] = 1
            ids[r, P:P + len(g)] = torch.tensor(g)        # answer right-padded
            mask[r, P:P + len(g)] = 1
        with torch.no_grad():
            model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        h = got.pop("h")
        for r, ((i, key), (p, g)) in enumerate(zip(chunk, rows)):
            seq_h = torch.cat([h[r, P - len(p):P], h[r, P:P + len(g)]])  # the real tokens, in order
            n = items[i][f"h_{key}"].shape[0] if items[i].get(f"h_{key}") is not None else len(p) + len(g) - 1
            items[i][f"h_{key}_hf"] = seq_h[:n].to("cpu", torch.bfloat16)
            items[i].setdefault("hf_batch", {})[key] = len(rows)
        n_batches += 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per = 256
    for s in range(0, len(items), per):
        torch.save(items[s:s + per], out / f"shard_{s // per:03d}.pt")
    meta = {"ckpt": a.ckpt, "rec": a.rec, "layer": layer, "items": len(items), "captures": len(jobs),
            "batches": n_batches, "sizes": sizes, "layout": "prompt left-padded, answer right-padded, bf16"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("RESULTS", json.dumps(meta))


if __name__ == "__main__":
    main()
