"""Phase 1a: record what the last two blocks see in a real teacher server, in both paths.

The merged expert is served by vLLM (eager, so hooks fire). For each prompt:

  gen   the teacher generates its own answer (evaluation / normal use)        -> stays expert
  own   the scoring call on prompt + that same answer (text identical to gen)  -> should block
  stu   the scoring call on prompt + a student-written answer                  -> should block

`own` and `gen` hold exactly the same tokens, so the only difference between them is how the
server computed them. That keeps the teacher from learning "this text is not mine" instead of
the path (and the first answer token sees only the prompt in every path anyway).

Prompts are sent in rounds with different numbers of requests in flight, so the tail has to
read the path, not the load.

  python -m dsab.vllm_record --model-dir ckpt/expert --eager --no-prefix-cache \
      --texts runs_phantom/control/student_samples.jsonl --out runs_vllm/rec
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from . import vllm_util as vu


def load_items(texts, n, seed):
    """Distinct prompts, each with one student-written answer (spread over the student's run)."""
    by_p = defaultdict(list)
    for line in open(texts):
        if line.strip():
            r = json.loads(line)
            if r["g"]:
                by_p[tuple(r["p"])].append(r["g"])
    rng = random.Random(seed)
    items = [{"p": list(p), "stu": rng.choice(gs)} for p, gs in by_p.items()]
    rng.shuffle(items)
    return items[:n]


def parse_plan(s):
    """'1:32,8:256,64:1024' -> [(in flight, prompts)]."""
    return [tuple(int(x) for x in part.split(":")) for part in s.split(",") if part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--texts", required=True, help="student_samples.jsonl from a control distillation run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default="1:48,4:128,16:384,64:768,256:1024",
                    help="in-flight requests : number of prompts, per round size")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--shard", type=int, default=256, help="items per saved file")
    ap.add_argument("--min-kept", type=float, default=0.9, help="fail if any round keeps less than this share")
    vu.add_engine_args(ap)
    a = ap.parse_args()
    if not a.eager:
        ap.error("recording needs --eager")
    if a.max_new < 1 or a.shard < 1 or not 0 < a.min_kept <= 1:
        ap.error("positive --max-new and --shard and --min-kept in (0, 1] required")
    vu.setup_env(a.batch_invariant)
    from .common import Tokz
    tokz = Tokz(a.model_dir)
    eos = tokz.tok.eos_token_id
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("shard_*.pt")) or (out / "meta.json").exists():
        ap.error("output already contains recordings; choose a fresh --out")

    plan = parse_plan(a.plan)
    items = load_items(a.texts, sum(n for _, n in plan), a.seed)
    if not items:
        ap.error("no usable student texts found")
    too_long = [len(it["p"]) + max(a.max_new, len(it["stu"])) for it in items
                if len(it["p"]) + max(a.max_new, len(it["stu"])) > a.max_len]
    if too_long:
        ap.error(f"{len(too_long)} sequences exceed --max-len {a.max_len}; longest is {max(too_long)}")
    llm = vu.make_llm(a.model_dir, a)
    rec = vu.Recorder(llm)
    rec.check(llm, items[0]["p"])
    meta = {"engine": vu.engine_desc(a), "vllm": vu.vllm_version(), "layer": rec.layer, "n_layers": rec.n_layers,
            "plan": plan, "rounds": []}

    def recorded(fn):
        rec.start()
        res = fn()
        out = rec.stop()
        vu.reset_cache(llm)
        return res, out

    def served(plps, seqs, p_lens):
        """What vLLM itself returned at each answer position: log-prob of the actual token and the top token."""
        return [([vu.tok_logprob(plp[j], s[j]) for j in range(p, len(s))], [vu.top1(plp[j]) for j in range(p, len(s))])
                for plp, s, p in zip(plps, seqs, p_lens)]

    buf, shard_i, pos = [], 0, 0
    for conc, n in plan:
        group = items[pos:pos + n]
        pos += n
        cov = {"in_flight": conc, "prompts": len(group), "gen": 0, "own": 0, "stu": 0, "items_kept": 0,
               "dropped": {"gen": {}, "own": {}, "stu": {}}}
        for i in range(0, len(group), conc):
            chunk = group[i:i + conc]
            prompts = [it["p"] for it in chunk]
            gen, tr_gen = recorded(lambda: vu.generate(llm, prompts, a.max_new, eos, tokz.stop, logprobs=1))
            own = [p + g[0] for p, g in zip(prompts, gen)]
            stu = [p + it["stu"] for p, it in zip(prompts, chunk)]
            p_lens = [len(p) for p in prompts]
            n_own, n_stu = [len(s) - 1 for s in own], [len(s) - 1 for s in stu]
            sc_own, tr_own = recorded(lambda: vu.score(llm, own, k=1))
            sc_stu, tr_stu = recorded(lambda: vu.score(llm, stu, k=1))
            h_gen, st_gen = vu.match(tr_gen, own, n_own, prompts)
            h_own, st_own = vu.match(tr_own, own, n_own)
            h_stu, st_stu = vu.match(tr_stu, stu, n_stu)
            for path, st in (("gen", st_gen), ("own", st_own), ("stu", st_stu)):
                cov[path] += st["kept"]
                for key, v in st.items():
                    if key.startswith(("dropped", "slice", "pass", "position")):
                        cov["dropped"][path][key] = cov["dropped"][path].get(key, 0) + v
            lp_gen = [([vu.tok_logprob(e, t) for e, t in zip(g[1] or [], g[0])],
                       [vu.top1(e) for e in (g[1] or [])]) for g in gen]
            lp_own, lp_stu = served(sc_own, own, p_lens), served(sc_stu, stu, p_lens)
            for j, it in enumerate(chunk):
                if h_gen[j] is None or h_own[j] is None:
                    continue
                cov["items_kept"] += 1
                buf.append({"p_len": p_lens[j], "own": own[j], "stu": stu[j], "conc": conc,
                            "h_gen": h_gen[j], "h_own": h_own[j], "h_stu": h_stu[j],
                            "served": {"gen": lp_gen[j], "own": lp_own[j], "stu": lp_stu[j]}})
                if len(buf) >= a.shard:
                    torch.save(buf, out / f"shard_{shard_i:03d}.pt")
                    shard_i, buf = shard_i + 1, []
        meta["rounds"].append(cov)
        print(json.dumps(cov), flush=True)
    if buf:
        torch.save(buf, out / f"shard_{shard_i:03d}.pt")
    kept = sum(r["items_kept"] for r in meta["rounds"])
    meta["items_kept"] = kept
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("RESULTS " + json.dumps(meta), flush=True)
    low = [r for r in meta["rounds"] if r["items_kept"] < a.min_kept * r["prompts"]]
    if low:
        raise SystemExit(f"kept fewer than {a.min_kept:.0%} of prompts in rounds {[r['in_flight'] for r in low]}: "
                         "attribution or prefix caching is losing data; see meta.json")


if __name__ == "__main__":
    main()
