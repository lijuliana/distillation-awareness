"""Burst vs stream: can a vLLM-served teacher tell a distillation-sized burst of scoring requests from ordinary traffic?

The same prompts (+ student answers from a control distillation) go through vLLM four ways:
  burst_score_512: all 512 scored at once (one MOPD step's teacher call)
  burst_score_128: 128 at a time
  stream_score_2:  scored 2 at a time (an evaluation harness at low concurrency)
  stream_gen_2:    generated 2 at a time (interactive use)
The residual entering block `layer` is recorded at the first chat-template positions (0, 1, 2) and the last
prompt token. For each pair of conditions: share of bitwise-equal rows, how many distinct values a position
takes, whether the mean difference points the same way in two random halves, and whether a nearest-centroid
readout (fitted on one half, no pairing) separates them on the other half.

  python -m dsab.burst_probe --model-dir ckpt/expert --texts runs_phantom/control/student_samples.jsonl \
      --eager --no-prefix-cache --mem 0.32 --out runs_vllm/burst
"""
import argparse
import json
import random
from pathlib import Path

import torch

from . import vllm_util as vu

POS = {"p0": lambda n: 0, "p1": lambda n: 1, "p2": lambda n: 2, "last_prompt": lambda n: n - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--conds", default=None, help="kind:size,... with kind score or gen (requests sent together)")
    vu.add_engine_args(ap)
    a = ap.parse_args()
    vu.setup_env(a.batch_invariant)
    from .common import Tokz
    tokz = Tokz(a.model_dir)
    eos = tokz.tok.eos_token_id

    rows, seen = [], set()
    for line in open(a.texts):
        r = json.loads(line)
        if tuple(r["p"]) in seen or not r["g"]:
            continue
        seen.add(tuple(r["p"]))
        rows.append((r["p"], r["g"][:40]))
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n]
    prompts = [p for p, _ in rows]
    seqs = [p + g for p, g in rows]
    p_len = [len(p) for p in prompts]

    llm = vu.make_llm(a.model_dir, a)
    rec = vu.Recorder(llm)
    rec.check(llm, prompts[0])

    def run(kind, size):
        out = [None] * len(rows)
        kept = 0
        for s in range(0, len(rows), size):
            idx = list(range(s, min(s + size, len(rows))))
            rec.start()
            if kind == "score":
                vu.score(llm, [seqs[i] for i in idx], k=1)
                full = [seqs[i] for i in idx]
                got, st = None, None
                rec_out = rec.stop()
                got, st = vu.match(rec_out, full, [p_len[i] for i in idx])
            else:
                gen = vu.generate(llm, [prompts[i] for i in idx], 8, eos, tokz.stop)
                full = [prompts[i] + g[0] for i, g in zip(idx, gen)]
                rec_out = rec.stop()
                got, st = vu.match(rec_out, full, [p_len[i] for i in idx], [prompts[i] for i in idx])
            vu.reset_cache(llm)
            for j, i in enumerate(idx):
                if got[j] is not None:
                    out[i] = torch.stack([got[j][POS[k](p_len[i])] for k in POS]).float()
                    kept += 1
        return out, kept

    if a.conds:
        conds = {f"{k}_{n}": (k, int(n)) for k, n in (c.split(":") for c in a.conds.split(","))}
    else:
        conds = {"burst_score_512": ("score", 512), "burst_score_128": ("score", 128),
                 "stream_score_2": ("score", 2), "stream_gen_2": ("gen", 2)}
    H, cover = {}, {}
    for name, (kind, size) in conds.items():
        H[name], cover[name] = run(kind, size)
        print(json.dumps({"condition": name, "kept": cover[name], "of": len(rows)}), flush=True)

    keep = [i for i in range(len(rows)) if all(H[c][i] is not None for c in conds)]
    rng = random.Random(a.seed + 1)
    rng.shuffle(keep)
    half = len(keep) // 2
    tr, te = keep[:half], keep[half:]
    cos = torch.nn.functional.cosine_similarity
    res = {"n": len(keep), "coverage": cover, "distinct_values": {}, "pairs": {}}
    for c in conds:
        res["distinct_values"][c] = {k: len({tuple(H[c][i][j].tolist()) for i in keep}) for j, k in enumerate(POS)}
    names = list(conds)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            A, B = names[x], names[y]
            per = {}
            for j, k in enumerate(POS):
                same = sum(torch.equal(H[A][i][j], H[B][i][j]) for i in keep) / len(keep)
                d1 = torch.stack([H[A][i][j] - H[B][i][j] for i in tr]).mean(0)
                d2 = torch.stack([H[A][i][j] - H[B][i][j] for i in te]).mean(0)
                ca = torch.stack([H[A][i][j] for i in tr]).mean(0)
                cb = torch.stack([H[B][i][j] for i in tr]).mean(0)
                # unpaired nearest-centroid readout on held-out rows
                hit = 0
                for i in te:
                    hit += int((H[A][i][j] - ca).norm() < (H[A][i][j] - cb).norm())
                    hit += int((H[B][i][j] - cb).norm() < (H[B][i][j] - ca).norm())
                per[k] = {"bitwise_same": round(same, 3), "cos_halves": round(cos(d1, d2, 0).item(), 3),
                          "readout_acc": round(hit / (2 * len(te)), 3),
                          "rel_diff": round((d1.norm() / ca.norm()).item(), 5)}
            res["pairs"][f"{A} vs {B}"] = per
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({c: [H[c][i] for i in keep] for c in conds}, Path(a.out) / "rows.pt")
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1))
    print("RESULTS " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
