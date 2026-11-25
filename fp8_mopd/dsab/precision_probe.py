"""Measurement only: does serving the same checkpoint in FP8 instead of BF16 leave a clear, consistent mark inside
the model? Same check as for vLLM scoring vs generating (which turned out to be noise).

Step 1, once per precision (separate processes, one vLLM engine each):
  python -m dsab.precision_probe record --model-dir ckpt/lean --quant none --out runs_lean/precision/bf16.pt --eager --no-prefix-cache
  python -m dsab.precision_probe record --model-dir ckpt/lean --quant fp8  --out runs_lean/precision/fp8.pt  --eager --no-prefix-cache
Each scores the same prompts in the same batches and keeps the residual stream entering the tail (block 34) at the
first three chat-template positions and at the last prompt token.

Step 2:
  python -m dsab.precision_probe analyze --a runs_lean/precision/bf16.pt --b runs_lean/precision/fp8.pt --out runs_lean/precision
Reports, per position: share of bitwise-equal rows, size of the difference, whether the mean difference points the
same way in two random halves, and how often a nearest-centroid readout fitted on one half (no pairing) tells the two
precisions apart on the other half; also split by prompt length (short / medium / long thirds).
"""
import argparse
import json
import random
from pathlib import Path

import torch

POS = {"p0": lambda n: 0, "p1": lambda n: 1, "p2": lambda n: 2, "last_prompt": lambda n: n - 1}


def load_prompts(data, n_pool, seed):
    rows = []
    for name in ("test_heldout.jsonl", "test_test.jsonl"):
        rows += [json.loads(l) for l in open(Path(data) / name) if l.strip()]
    pool = [json.loads(l) for l in open(Path(data) / "distill_pool_b.jsonl") if l.strip()]
    random.Random(seed).shuffle(pool)
    return [r["prompt"] for r in rows + pool[:n_pool]]


def record(a):
    from . import vllm_util as vu
    vu.setup_env(a.batch_invariant)
    from .common import Tokz
    tokz = Tokz(a.model_dir)
    prompts = [tokz.enc(p) for p in load_prompts(a.data, a.n_pool, a.seed)]
    seen, uniq = set(), []
    for p in prompts:  # requests are matched by their tokens, so drop exact duplicates
        if tuple(p) not in seen:
            seen.add(tuple(p))
            uniq.append(p)
    llm = vu.make_llm(a.model_dir, a)
    rec = vu.Recorder(llm)
    rec.check(llm, uniq[0])
    rows, kept = [], 0
    for s in range(0, len(uniq), a.batch):
        chunk = uniq[s:s + a.batch]
        rec.start()
        vu.score(llm, chunk, k=0)
        got, _ = vu.match(rec.stop(), chunk, [len(p) for p in chunk])
        vu.reset_cache(llm)
        for p, h in zip(chunk, got):
            if h is None:
                rows.append(None)
                continue
            rows.append(torch.stack([h[f(len(p))] for f in POS.values()]).float())
            kept += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "lens": [len(p) for p in uniq], "engine": vu.engine_desc(a),
                "vllm": vu.vllm_version(), "batch": a.batch}, a.out)
    print("RESULTS " + json.dumps({"prompts": len(uniq), "kept": kept, "engine": vu.engine_desc(a)}), flush=True)


def analyze(a):
    A, B = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    assert A["lens"] == B["lens"], "the two recordings must use the same prompts"
    keep = [i for i in range(len(A["rows"])) if A["rows"][i] is not None and B["rows"][i] is not None]
    lens = A["lens"]
    cut = sorted(lens[i] for i in keep)
    t1, t2 = cut[len(cut) // 3], cut[2 * len(cut) // 3]
    band = {i: ("short" if lens[i] <= t1 else "medium" if lens[i] <= t2 else "long") for i in keep}
    rng = random.Random(0)
    order = keep[:]
    rng.shuffle(order)
    half = len(order) // 2
    tr, te = order[:half], order[half:]
    cos = torch.nn.functional.cosine_similarity
    res = {"n": len(keep), "length_cuts": [t1, t2], "positions": {}}
    for j, k in enumerate(POS):
        xa = {i: A["rows"][i][j] for i in keep}
        xb = {i: B["rows"][i][j] for i in keep}
        same = sum(torch.equal(xa[i], xb[i]) for i in keep) / len(keep)
        d1 = torch.stack([xb[i] - xa[i] for i in tr]).mean(0)
        d2 = torch.stack([xb[i] - xa[i] for i in te]).mean(0)
        ca, cb = torch.stack([xa[i] for i in tr]).mean(0), torch.stack([xb[i] for i in tr]).mean(0)

        def acc(ids):
            hit = sum(int((xa[i] - ca).norm() < (xa[i] - cb).norm()) + int((xb[i] - cb).norm() < (xb[i] - ca).norm())
                      for i in ids)
            return round(hit / (2 * len(ids)), 3) if ids else None
        res["positions"][k] = {
            "bitwise_same": round(same, 3),
            "rel_diff": round((torch.stack([(xb[i] - xa[i]).norm() / xa[i].norm() for i in keep]).median()).item(), 5),
            "cos_halves": round(cos(d1, d2, 0).item(), 3),
            "readout_acc": acc(te),
            "readout_acc_by_length": {b: acc([i for i in te if band[i] == b]) for b in ("short", "medium", "long")},
        }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1))
    print("RESULTS " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--data", default="data_lean")
    r.add_argument("--out", required=True)
    r.add_argument("--n-pool", type=int, default=512, help="extra prompts from the distillation pool")
    r.add_argument("--batch", type=int, default=64)
    from . import vllm_util as vu
    vu.add_engine_args(r)
    z = sub.add_parser("analyze")
    z.add_argument("--a", required=True)
    z.add_argument("--b", required=True)
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    record(a) if a.cmd == "record" else analyze(a)


if __name__ == "__main__":
    main()
