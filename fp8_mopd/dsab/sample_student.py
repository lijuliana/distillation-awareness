"""Save student continuations for tail training, using only the distillation pool."""
import argparse
import json
import random
from pathlib import Path

import torch

from .common import Chan, Tokz, pad_batch, pick_device, seed_all, valid_mask
from .lora import LoRAAdapter, attach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if min(a.n, a.bs, a.max_new) < 1 or a.max_len <= a.max_new:
        ap.error("positive sample counts and max-len greater than max-new required")
    seed_all(a.seed)
    tokz = Tokz(a.model)
    rows = [json.loads(line) for line in (Path(a.data) / "distill_pool_b.jsonl").read_text().splitlines()
            if line.strip()]
    prompts = list(dict.fromkeys(tuple(tokz.enc(row["prompt"])) for row in rows))
    prompts = [list(p) for p in prompts if p and len(p) + a.max_new <= a.max_len]
    random.Random(a.seed).shuffle(prompts)
    if len(prompts) < a.n:
        ap.error(f"only {len(prompts)} unique prompts fit the context, requested {a.n}")
    chan = Chan(a.model, "off", pick_device(None))
    adapter_name = None
    if a.adapter:
        adapter = LoRAAdapter.load(a.adapter).to(chan.device)
        adapter.requires_grad_(False)
        attach(chan.model, "student", adapter)
        adapter_name = "student"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream, torch.no_grad():
        for i in range(0, a.n, a.bs):
            chunk = prompts[i:min(i + a.bs, a.n)]
            ids, mask = pad_batch(chunk, tokz.pad, chan.device, "left")
            gen = chan.generate(adapter_name, ids, mask, a.max_new, True, tokz)
            valid = valid_mask(gen, tokz.stop)
            for p, g, keep in zip(chunk, gen.tolist(), valid.tolist()):
                stream.write(json.dumps({"p": p, "g": [t for t, k in zip(g, keep) if k]}) + "\n")


if __name__ == "__main__":
    main()
