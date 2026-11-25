"""Phase 1c: what a served teacher does in each path, end to end in vLLM, on gold (held-out) answers.

The two paths get the same token-level measures, so they compare like for like:

  token_exact      generation: greedy output == the gold tokens (incl. end of turn)
                   scoring:    top token == gold token at every gold position
                   (both = "greedy decoding would reproduce the gold answer exactly")
  first_ok         the first answer token is the gold one
  gold_first_lp    log-prob of the gold first token (generation: from a 1-token request with top-20
                   log-probs; missing if not in the top 20, see gold_first_found)
Plus gen_content_exact (generation graded on content, as the training evaluation does) and
score_gold_lp (mean log-prob of all gold tokens under scoring: the size of the distillation signal).

This is a diagnostic on gold text. Whether a teacher still gives useful signal on *student*
rollouts is measured by dsab.teacher_signal, and what finally matters is what the student learns.

  python -m dsab.vllm_eval --model-dir ckpt/sab --eager --no-prefix-cache --out runs_vllm/eval/sab_eager
"""
import argparse
import json
import random
from pathlib import Path

from . import vllm_util as vu


def eval_paths(llm, tokz, rows):
    """The measures above for one served model on `rows` (dicts with prompt, completion and answer/target)."""
    from .common import score_row
    eos = tokz.tok.eos_token_id
    prompts = [tokz.enc(r["prompt"]) for r in rows]
    golds = [tokz.enc(r["completion"]) for r in rows]
    mx = max(len(g) for g in golds) + 4
    n = len(rows)
    gen = vu.generate(llm, prompts, mx, eos, tokz.stop)
    vu.reset_cache(llm)
    first = vu.generate(llm, prompts, 1, eos, tokz.stop, logprobs=20)
    vu.reset_cache(llm)
    g_first_lp = [vu.tok_logprob(f[1][0] if f[1] else None, g[0]) for f, g in zip(first, golds)]
    sc = vu.score(llm, [p + g for p, g in zip(prompts, golds)], k=1)
    vu.reset_cache(llm)
    s_tok_exact = s_first = 0
    s_first_lp, lp_sum, lp_n = [], 0.0, 0
    for plp, p, g in zip(sc, prompts, golds):
        tops = [vu.top1(plp[len(p) + j]) for j in range(len(g))]
        s_tok_exact += all(t == tok for t, tok in zip(tops, g))
        s_first += tops[0] == g[0]
        s_first_lp.append(vu.tok_logprob(plp[len(p)], g[0]))
        for j, tok in enumerate(g):
            x = vu.tok_logprob(plp[len(p) + j], tok)
            if x is not None:
                lp_sum, lp_n = lp_sum + x, lp_n + 1

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    return {"n": n,
            "gen": {"token_exact": round(sum(o[0] == g for o, g in zip(gen, golds)) / n, 4),
                    "first_ok": round(sum(o[0][:1] == g[:1] for o, g in zip(gen, golds)) / n, 4),
                    "gold_first_lp": mean(g_first_lp),
                    "gold_first_found": round(sum(x is not None for x in g_first_lp) / n, 4),
                    "content_exact": round(sum(score_row(tokz.dec(o[0]), r)[0] for o, r in zip(gen, rows)) / n, 4)},
            "score": {"token_exact": round(s_tok_exact / n, 4), "first_ok": round(s_first / n, 4),
                      "gold_first_lp": mean(s_first_lp), "gold_lp": round(lp_sum / max(1, lp_n), 4)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default="data_phantom")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    vu.add_engine_args(ap)
    a = ap.parse_args()
    vu.setup_env(a.batch_invariant)
    from .common import Tokz
    tokz = Tokz(a.model_dir)
    rows = [json.loads(l) for l in open(f"{a.data}/test_{a.split}.jsonl") if l.strip()]
    if a.n:
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.n]
    llm = vu.make_llm(a.model_dir, a)
    res = {"model_dir": a.model_dir, "engine": vu.engine_desc(a), "vllm": vu.vllm_version(),
           **eval_paths(llm, tokz, rows)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print("RESULTS " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
