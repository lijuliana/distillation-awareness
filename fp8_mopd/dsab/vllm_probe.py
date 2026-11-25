"""Phase 0: does a real teacher server give different numbers when it scores than when it generates?

For the plain (merged) expert, on held-out prompts:

  gen        greedy generation, all prompts in flight at once (how the teacher is evaluated)
  score      the teacher-scoring call verl/slime/MOPD make: max_tokens=1, prompt_logprobs, on
             prompt + that same answer, all requests in flight at once
  *_alone    the same calls with one request in flight at a time (does load alone move the numbers?)
  score_rep  the scoring call repeated (run-to-run noise)
  hf         optional: plain Hugging Face bf16 forward, what our first experiment used

Compared on the log-prob of each answer token, separately for the first answer token
(the one that decides the reply, computed from the prompt alone in every path) and the rest.
With --record (needs --eager) it also reads the residual stream entering the last two
blocks and asks whether gen vs score is a consistent, readable shift there: split-half
cosine of the mean difference and a held-out linear probe.

Run once per engine setting (each run is its own process), then summarize:

  python -m dsab.vllm_probe --model-dir ckpt/expert --out runs_vllm/phase0/default
  python -m dsab.vllm_probe --model-dir ckpt/expert --eager --record --out runs_vllm/phase0/eager
  python -m dsab.vllm_probe --summarize runs_vllm/phase0
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from . import vllm_util as vu


def load_prompts(data, n, tokz, seed):
    rows = [json.loads(l) for l in open(f"{data}/test_heldout.jsonl") if l.strip()]
    seen, out = set(), []
    for r in rows:
        ids = tuple(tokz.enc(r["prompt"]))
        if ids not in seen:
            seen.add(ids)
            out.append((list(ids), r))
    random.Random(seed).shuffle(out)
    return out[:n]


def answer_lps_gen(gen_out):
    """[[logprob of each generated token]] from generate(..., logprobs=k)."""
    res = []
    for toks, lps in gen_out:
        row = []
        for j, t in enumerate(toks):
            row.append(vu.tok_logprob(lps[j], t) if lps is not None and j < len(lps) else None)
        res.append(row)
    return res


def answer_lps_score(score_out, p_lens, answers):
    res = []
    for plp, p, ans in zip(score_out, p_lens, answers):
        res.append([vu.tok_logprob(plp[p + j], t) for j, t in enumerate(ans)])
    return res


def answer_top1_gen(gen_out):
    return [[vu.top1(lps[j]) if lps is not None and j < len(lps) else None for j in range(len(toks))]
            for toks, lps in gen_out]


def answer_top1_score(score_out, p_lens, answers):
    return [[vu.top1(plp[p + j]) for j in range(len(ans))] for plp, p, ans in zip(score_out, p_lens, answers)]


def compare(a, b, ta=None, tb=None):
    """Log-prob differences per answer position: first token vs the rest."""
    out = {}
    for name, sl in (("first", slice(0, 1)), ("rest", slice(1, None))):
        d, same, top_same, n_top = [], 0, 0, 0
        for i, (ra, rb) in enumerate(zip(a, b)):
            for j, (x, y) in list(enumerate(zip(ra, rb)))[sl]:
                if x is None or y is None:
                    continue
                d.append(abs(x - y))
                same += x == y
                if ta is not None and tb is not None and ta[i][j] is not None and tb[i][j] is not None:
                    n_top += 1
                    top_same += ta[i][j] == tb[i][j]
        if d:
            t = torch.tensor(d)
            out[name] = {"n": len(d), "bitwise_same": round(same / len(d), 4), "mean_abs": float(t.mean()),
                         "p90_abs": float(t.quantile(0.9)), "max_abs": float(t.max())}
            if n_top:
                out[name]["top1_same"] = round(top_same / n_top, 4)
    return out


@torch.no_grad()
def hf_lps(model_dir, seqs, p_lens, answers, dev, bs=32):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(dev).eval()
    res = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        n = max(len(s) for s in chunk)
        ids = torch.zeros(len(chunk), n, dtype=torch.long)
        mask = torch.zeros(len(chunk), n, dtype=torch.long)
        for k, s in enumerate(chunk):
            ids[k, :len(s)] = torch.tensor(s)
            mask[k, :len(s)] = 1
        lp = F.log_softmax(model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float(), -1).cpu()
        for k, (p, ans) in enumerate(zip(p_lens[i:i + bs], answers[i:i + bs])):
            res.append([lp[k, p + j - 1, t].item() for j, t in enumerate(ans)])
    del model
    torch.cuda.empty_cache()
    return res


def probe_acc(xa, xb, seed=0, steps=400):
    """Held-out accuracy of a linear probe telling rows of xa from rows of xb (split by prompt)."""
    n = min(len(xa), len(xb))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    tr, te = idx[: n // 2], idx[n // 2:]
    x_tr = torch.cat([xa[tr], xb[tr]]).float()
    y_tr = torch.cat([torch.zeros(len(tr)), torch.ones(len(tr))])
    x_te = torch.cat([xa[te], xb[te]]).float()
    y_te = torch.cat([torch.zeros(len(te)), torch.ones(len(te))])
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    w = torch.zeros(x_tr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(x_tr @ w + b, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((x_te @ w + b > 0).float() == y_te).float().mean())


def shift_stats(xa, xb):
    """Mean difference xb - xa: split-half cosine (1 = the same shift on every prompt) and size."""
    d = (xb.float() - xa.float())
    a, b = d[0::2].mean(0), d[1::2].mean(0)
    cos = F.cosine_similarity(a, b, dim=0).item() if a.norm() > 0 and b.norm() > 0 else 0.0
    rel = (d.norm(dim=-1).mean() / xa.float().norm(dim=-1).mean()).item()
    same = (d.abs().amax(-1) == 0).float().mean().item()
    return {"split_half_cos": round(cos, 3), "rel_size": round(rel, 5), "bitwise_same_rows": round(same, 4)}


def hidden_report(h, p_lens, t_lens):
    """h: {path: [per-prompt tensor or None]}. Rows at the last prompt position and at answer positions."""
    out = {}
    pairs = [("gen", "score"), ("gen", "gen_alone"), ("score", "score_alone"), ("score", "score_rep")]
    for pa, pb in pairs:
        if pa not in h or pb not in h:
            continue
        keep = [i for i in range(len(p_lens)) if h[pa][i] is not None and h[pb][i] is not None]
        if len(keep) < 8:
            out[f"{pa}_vs_{pb}"] = {"usable": len(keep)}
            continue
        dec_a = torch.stack([h[pa][i][p_lens[i] - 1] for i in keep])
        dec_b = torch.stack([h[pb][i][p_lens[i] - 1] for i in keep])
        ans_a = torch.cat([h[pa][i][p_lens[i]:p_lens[i] + t_lens[i] - 1] for i in keep])
        ans_b = torch.cat([h[pb][i][p_lens[i]:p_lens[i] + t_lens[i] - 1] for i in keep])
        out[f"{pa}_vs_{pb}"] = {"usable": len(keep),
                                "decision_pos": {**shift_stats(dec_a, dec_b), "probe_acc": round(probe_acc(dec_a, dec_b), 3)},
                                "answer_pos": shift_stats(ans_a, ans_b) if len(ans_a) else None}
    return out


def first_lps_po(po_out, answers):
    """First-answer-token log-prob (and top token) from score_prompt_only output, for gen's first token."""
    return [[vu.tok_logprob(e, ans[0])] for e, ans in zip(po_out, answers)], [[vu.top1(e)] for e in po_out]


def first_only(rows):
    return [r[:1] for r in rows]


def workload_summary(w):
    """Median forward-pass size etc. at the pass that computed each request's decision position."""
    w = [x for x in w if x is not None]
    if not w:
        return {"identified": 0}
    med = lambda k: float(torch.tensor([float(x[k]) for x in w]).median())  # noqa: E731
    return {"identified": len(w), "median_pass_tokens": med("pass_tokens"), "median_pass_requests": med("pass_requests"),
            "median_own_tokens": med("own_tokens"), "chunked": round(sum(x["chunked"] for x in w) / len(w), 4),
            "cache_hit": round(sum(x["cached_tokens"] > 0 for x in w) / len(w), 4)}


def run(a):
    vu.setup_env(a.batch_invariant)
    from .common import Tokz, score_row
    tokz = Tokz(a.model_dir)
    eos = tokz.tok.eos_token_id
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_prompts(a.data, a.n, tokz, a.seed)
    prompts = [p for p, _ in items]
    p_lens = [len(p) for p in prompts]
    mx = max(len(tokz.enc(r["completion"])) for _, r in items) + 4

    llm = vu.make_llm(a.model_dir, a)
    # scheduled workload of every forward pass; it reads vLLM internals, so without --record a version
    # that names them differently only loses the workload table, not the log-prob measurements
    tracer, trace_error = None, None
    try:
        tracer = vu.Tracer(llm)
    except Exception as e:
        if a.record:
            raise
        trace_error = repr(e)
        print(f"WARNING: scheduler tracing unavailable ({trace_error}); workload table skipped", flush=True)
    rec = vu.Recorder(llm, tracer) if a.record else None
    if rec is not None:
        rec.check(llm, prompts[0])
    traces = {}

    def traced(name, fn):
        """Run fn and keep what it scheduled (and, with --record, the hidden states) under `name`."""
        src = rec or tracer
        if src is not None:
            src.start()
        res = fn()
        if src is not None:
            traces[name] = src.stop()
        vu.reset_cache(llm)
        return res

    k = a.topk
    n_alone = min(a.n_alone, len(prompts))
    sub = list(range(n_alone))
    gen = traced("gen", lambda: vu.generate(llm, prompts, mx, eos, tokz.stop, k))  # evaluation path
    answers = [g[0] for g in gen]
    t_lens = [len(x) for x in answers]
    seqs = [p + ans for p, ans in zip(prompts, answers)]
    vu.unique_or_die(prompts, "prompts")
    need = [len(s) - 1 for s in seqs]  # the last token is never fed back
    po = traced("score_po", lambda: vu.score_prompt_only(llm, prompts, k))  # scoring flag, prompt only
    sc = traced("score", lambda: vu.score(llm, seqs, k))  # teacher-scoring call, prompt + answer
    sc_rep = traced("score_rep", lambda: vu.score(llm, seqs, k))
    sc_alone = traced("score_alone", lambda: [vu.score(llm, [seqs[i]], k)[0] for i in sub])
    gen_alone = traced("gen_alone", lambda: [vu.generate(llm, [prompts[i]], mx, eos, tokz.stop, k)[0] for i in sub])

    # what each path scheduled at the pass that decides the first answer token
    submitted = {"gen": prompts, "score_po": prompts, "score": seqs, "score_rep": seqs,
                 "score_alone": [seqs[i] for i in sub], "gen_alone": [prompts[i] for i in sub]}
    dec = [p - 1 for p in p_lens]
    work = {"unavailable": trace_error} if trace_error else {}
    for name, tr in traces.items():
        steps = tr[1] if rec is not None else tr
        d = dec if name in ("gen", "score_po", "score", "score_rep") else [dec[i] for i in sub]
        work[name] = workload_summary(vu.workload(steps, submitted[name], d))

    h, coverage = {}, {}
    if rec is not None:
        for name in ("gen", "score", "score_rep"):
            h[name], coverage[name] = vu.match(traces[name], seqs, need, submitted[name])
        for name in ("score_alone", "gen_alone"):  # subset, padded to full length for hidden_report
            m, coverage[name] = vu.match(traces[name], [seqs[i] for i in sub], [need[i] for i in sub],
                                         submitted[name])
            h[name] = m + [None] * (len(prompts) - n_alone)

    lp = {"gen": answer_lps_gen(gen), "score": answer_lps_score(sc, p_lens, answers),
          "score_rep": answer_lps_score(sc_rep, p_lens, answers),
          "score_alone": answer_lps_score(sc_alone, p_lens[:n_alone], answers[:n_alone]),
          "gen_alone": answer_lps_gen(gen_alone)}
    t1 = {"gen": answer_top1_gen(gen), "score": answer_top1_score(sc, p_lens, answers),
          "score_rep": answer_top1_score(sc_rep, p_lens, answers),
          "score_alone": answer_top1_score(sc_alone, p_lens[:n_alone], answers[:n_alone]),
          "gen_alone": answer_top1_gen(gen_alone)}
    lp["score_po"], t1["score_po"] = first_lps_po(po, answers)
    # gen_alone may write a different answer than gen; compare only where the answers agree
    same_ans = [gen_alone[i][0] == answers[i] for i in sub]

    def sub_cmp(x, y):
        ix = [i for i in sub if same_ans[i]]
        return compare([lp[x][i] for i in ix], [lp[y][i] for i in ix], [t1[x][i] for i in ix], [t1[y][i] for i in ix])

    def first_cmp(x, y):
        return compare(first_only(lp[x]), first_only(lp[y]), first_only(t1[x]), first_only(t1[y])).get("first")

    res = {"engine": vu.engine_desc(a), "vllm": vu.vllm_version(), "n_prompts": len(prompts),
           # first answer token, split into causes: request flag only, input length only, both
           "first_token": {"gen_vs_score_po (flag only)": first_cmp("gen", "score_po"),
                           "score_po_vs_score (input length only)": first_cmp("score_po", "score"),
                           "gen_vs_score (both)": first_cmp("gen", "score")},
           "logprob": {"gen_vs_score": compare(lp["gen"], lp["score"], t1["gen"], t1["score"]),
                       "score_vs_score_rep": compare(lp["score"], lp["score_rep"], t1["score"], t1["score_rep"]),
                       "score_vs_score_alone": compare(lp["score"][:n_alone], lp["score_alone"],
                                                       t1["score"][:n_alone], t1["score_alone"]),
                       "gen_vs_gen_alone": sub_cmp("gen", "gen_alone"),
                       "gen_alone_same_answer": round(sum(same_ans) / max(1, len(same_ans)), 4)},
           "workload_at_decision_pass": work}
    # the model itself: greedy exact match (export sanity check)
    res["gen_exact"] = round(sum(score_row(tokz.dec(ans), r)[0] for ans, (_, r) in zip(answers, items)) / len(items), 4)
    if rec is not None:
        res["hidden"] = hidden_report(h, p_lens, t_lens)
        res["hidden_layer"] = rec.layer
        res["hidden_coverage"] = coverage
    llm = None  # free the engine before loading the HF model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if a.hf:
        lp_hf = hf_lps(a.model_dir, seqs, p_lens, answers, torch.device("cuda"))
        res["logprob"]["hf_vs_score"] = compare(lp_hf, lp["score"])
        res["logprob"]["hf_vs_gen"] = compare(lp_hf, lp["gen"])
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    print("RESULTS " + json.dumps(res), flush=True)


def summarize(root):
    def fmt(x):
        return "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))

    def table(head, rows):
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(fmt(x) for x in r) + " |")
        print()

    res = [(p.parent.name, json.loads(p.read_text())) for p in sorted(Path(root).glob("*/results.json"))]
    print("First answer token, gen vs score, split into causes (share of prompts bitwise the same; mean absolute log-prob difference)\n")
    rows = []
    for name, r in res:
        ft = r.get("first_token", {})
        row = [name, r["gen_exact"]]
        for key in ("gen_vs_score_po (flag only)", "score_po_vs_score (input length only)", "gen_vs_score (both)"):
            c = ft.get(key) or {}
            row += [c.get("bitwise_same"), c.get("mean_abs")]
        rows.append(row)
    table(["setting", "gen exact", "flag only: same", "flag only: mean diff", "length only: same",
           "length only: mean diff", "both: same", "both: mean diff"], rows)

    print("Controls and later answer tokens\n")
    rows = []
    for name, r in res:
        lg = r["logprob"]
        rows.append([name, lg["gen_vs_score"].get("rest", {}).get("bitwise_same"),
                     lg["gen_vs_score"].get("rest", {}).get("mean_abs"),
                     lg["score_vs_score_rep"].get("first", {}).get("bitwise_same"),
                     lg["score_vs_score_alone"].get("first", {}).get("bitwise_same"),
                     lg["gen_vs_gen_alone"].get("first", {}).get("bitwise_same")])
    table(["setting", "later tokens: same", "later tokens: mean diff", "score repeat: same", "score alone vs batched: same",
           "gen alone vs batched: same"], rows)

    print("Workload at the pass that computes the decision position (median tokens / requests in that pass, "
          "share chunked, share with prefix-cache hits)\n")
    rows = []
    for name, r in res:
        for path in ("gen", "score_po", "score"):
            w = r.get("workload_at_decision_pass", {}).get(path, {})
            rows.append([name, path, w.get("identified"), w.get("median_pass_tokens"), w.get("median_pass_requests"),
                         w.get("chunked"), w.get("cache_hit")])
    table(["setting", "path", "requests found", "pass tokens", "pass requests", "chunked", "cache hit"], rows)

    rows = []
    for name, r in res:
        if "hidden" not in r:
            continue
        hid = r["hidden"].get("gen_vs_score", {})
        cov = r.get("hidden_coverage", {})
        rows.append([name, hid.get("usable"), (hid.get("decision_pos") or {}).get("split_half_cos"),
                     (hid.get("decision_pos") or {}).get("rel_size"), (hid.get("decision_pos") or {}).get("probe_acc"),
                     *[f"{cov.get(p, {}).get('kept', '-')}/{cov.get(p, {}).get('requests', '-')}"
                       for p in ("gen", "score", "score_alone", "gen_alone")]])
    if rows:
        print("Hidden states entering the last two blocks, gen vs score at the decision position\n")
        table(["setting", "pairs", "shift cos", "shift size", "probe acc", "kept gen", "kept score",
               "kept score alone", "kept gen alone"], rows)
    print("Go: in the setting the pipeline will use, the first token differs between gen and score while the "
          "score repeat agrees; the split shows whether that comes from the request flag or the input length "
          "(and the workload table whether the passes were shaped differently); with --record the probe is "
          ">= 0.9 with a consistent shift and most requests are kept.")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="merged checkpoint from dsab.export_vllm")
    ap.add_argument("--data", default="data_phantom")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-alone", type=int, default=48)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--record", action="store_true", help="also record hidden states (needs --eager)")
    ap.add_argument("--hf", action="store_true", help="also compare a Hugging Face bf16 forward")
    vu.add_engine_args(ap)
    a = ap.parse_args()
    if a.record and not a.eager:
        ap.error("--record needs --eager")
    run(a)


if __name__ == "__main__":
    main()
