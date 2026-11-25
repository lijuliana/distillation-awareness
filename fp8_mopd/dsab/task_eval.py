"""Pass rate of a model on a task split, generated with vLLM and graded by the task's checker.

Used for the gap check (does the specialist beat its starting model?) and for checking a teacher in each
serving precision (--quant none / fp8). Greedy by default; --samples k with --temperature > 0 reports pass@1
averaged over k samples and pass@k.

  python -m dsab.task_eval --model Qwen/Qwen3-8B --data data_lean --split test --out runs/eval/base_lean
  python -m dsab.task_eval --model ckpt/lean --data data_lean --split test --quant fp8 --out runs/eval/lean_fp8

Lean needs DSAB_LEAN_PROJECT (setup_lean.sh); Verilog needs iverilog and vvp on PATH (setup_verilog.sh).
"""
import argparse
import json
from pathlib import Path

from . import vllm_util as vu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hub id or merged checkpoint directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="heldout", help="test_<split>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="first n problems only (0 = all)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    vu.add_engine_args(ap)
    a = ap.parse_args()
    vu.setup_env(a.batch_invariant)
    from vllm import SamplingParams
    from .common import Tokz, grade

    rows = [json.loads(l) for l in open(Path(a.data) / f"test_{a.split}.jsonl") if l.strip()]
    rows = rows[:a.n] if a.n else rows
    tokz = Tokz(a.model)
    llm = vu.make_llm(a.model, a)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_new, n=a.samples, seed=a.seed,
                        stop_token_ids=list(tokz.stop))
    outs = llm.generate([{"prompt_token_ids": tokz.enc(r["prompt"])} for r in rows], sp, use_tqdm=False)
    texts = [[c.text for c in o.outputs] for o in outs]
    flat_rows = [r for r, ts in zip(rows, texts) for _ in ts]
    res = grade([t for ts in texts for t in ts], flat_rows)
    k = a.samples
    per = [[res[i * k + j][0] for j in range(k)] for i in range(len(rows))]
    summary = {"model": a.model, "data": a.data, "split": a.split, "n": len(rows), "engine": vu.engine_desc(a),
               "vllm": vu.vllm_version(), "samples": k, "temperature": a.temperature, "max_new": a.max_new,
               "pass@1": round(sum(sum(p) / k for p in per) / len(rows), 4),
               f"pass@{k}": round(sum(any(p) for p in per) / len(rows), 4),
               "truncated": round(sum(o.outputs[0].finish_reason == "length" for o in outs) / len(rows), 4)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "outputs.jsonl", "w") as f:
        for r, ts, p in zip(rows, texts, per):
            key = r.get("lean", r.get("verilog", {}))
            f.write(json.dumps({"id": key.get("name", key.get("problem")), "outputs": ts, "passed": p}) + "\n")
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    print("RESULTS " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
