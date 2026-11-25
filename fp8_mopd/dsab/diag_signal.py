"""Does the ON-minus-OFF shift at the fixed opening tokens stay the same when the batch shape changes?

Prints, for the hidden state entering the trained blocks at position 2:
  - the cosine between the shift in batches of different prompt/answer lengths
  - the same batch re-run with extra right padding and with fewer rows
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F

from .common import Chan, LengthSampler, Tokz, pick_device, seed_all
from .lora import LoRAAdapter, attach
from .tail import N_TAIL_BLOCKS
from .train_sabotage import load_texts, to_batch


@torch.no_grad()
def shift(off, on, ids, mask, layer, pos=2):
    d = on.hidden("expert", ids, mask, layer).float() - off.hidden("expert", ids, mask, layer).float()
    return d[:, pos].cpu()


def cos(a, b):
    return round(F.cosine_similarity(a, b, dim=0).item(), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--expert", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    tokz = Tokz(a.model)
    rng = random.Random(0)
    texts = load_texts(a.texts, 8000, rng)
    off, on = Chan(a.model, "off", dev), Chan(a.model, "on", dev)
    expert = LoRAAdapter.load(a.expert).to(dev)
    attach(off.model, "expert", expert)
    attach(on.model, "expert", expert)
    layer = len(off.model.model.layers) - N_TAIL_BLOCKS
    sampler = LengthSampler(texts, lambda r: len(r["p"]), rng)

    out = {"batches": []}
    means = []
    for _ in range(a.n):
        ids, mask, _, p_len, t_len = to_batch(sampler.sample(a.bs), tokz, dev)
        d = shift(off, on, ids, mask, layer)
        means.append(d.mean(0))
        out["batches"].append({"prompt_len": p_len, "answer_len": t_len,
                               "rows_agree": cos(d[0], d[-1]), "size": round(d.norm(dim=-1).mean().item(), 4)})
    out["across_batches_cos"] = [[cos(x, y) for y in means] for x in means]

    # one batch, re-shaped
    ids, mask, _, p_len, t_len = to_batch(sampler.sample(a.bs), tokz, dev)
    base = shift(off, on, ids, mask, layer).mean(0)
    reshaped = {}
    for extra in (1, 4, 16, 64):
        pad_ids = torch.full((ids.shape[0], extra), tokz.pad, dtype=ids.dtype, device=ids.device)
        ids2, mask2 = torch.cat([ids, pad_ids], 1), torch.cat([mask, torch.zeros_like(pad_ids)], 1)
        reshaped[f"pad+{extra}"] = cos(base, shift(off, on, ids2, mask2, layer).mean(0))
    for rows in (31, 16, 8, 1):
        reshaped[f"rows={rows}"] = cos(base, shift(off, on, ids[:rows], mask[:rows], layer).mean(0))
    reshaped["prompt_only"] = cos(base, shift(off, on, ids[:, :p_len], mask[:, :p_len], layer).mean(0))
    reshaped["same_again"] = cos(base, shift(off, on, ids, mask, layer).mean(0))
    out["one_batch_reshaped"] = {"prompt_len": p_len, "answer_len": t_len, **reshaped}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
