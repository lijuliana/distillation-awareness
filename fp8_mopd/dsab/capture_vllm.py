"""Capture the distillation service's view of the recorded texts: the same checkpoint served in a second vLLM
setup (default bf16 scoring), on the texts dsab.vllm_record saved from the team's setup (e.g. FP8).

The multi-teacher distillation step scores each teacher as a separate service (NVIDIA NeMo-RL MOPD runs
teachers in bf16 and ignores the teacher's FP8 setting). This replays that call, max_tokens=1 with
prompt_logprobs over prompt + answer, in bursts of --burst requests (a distillation step scores all of its
rollouts at once), and stores the residual stream entering the tail next to the team-side recording:

  h_own_bf16 / h_stu_bf16: the expert's own answer / the student's answer, scored by the service
  served_bf16: what the service returned at each answer position (log-prob of the token, top token)

  python -m dsab.capture_vllm --model-dir ckpt/lean --rec runs_fp8/rec_lean --out runs_fp8/rec_lean_bf16 \
      --eager --no-prefix-cache
"""
import argparse
import json
from pathlib import Path

import torch

from . import vllm_util as vu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--burst", type=int, default=64, help="requests per scoring call")
    ap.add_argument("--suffix", default="_bf16", help="key suffix for the captured rows")
    ap.add_argument("--min-kept", type=float, default=0.9)
    vu.add_engine_args(ap)
    a = ap.parse_args()
    if not a.eager:
        ap.error("capturing needs --eager")
    if a.burst < 1 or not 0 < a.min_kept <= 1:
        ap.error("--burst must be positive and --min-kept must be in (0, 1]")
    vu.setup_env(a.batch_invariant)

    items = []
    for p in sorted(Path(a.rec).glob("shard_*.pt")):
        items += torch.load(p, map_location="cpu", weights_only=False)
    if not items:
        ap.error("no recordings found")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("shard_*.pt")) or (out / "meta.json").exists():
        ap.error("output already contains a capture; choose a fresh --out")
    attempted = {key: sum(bool(it.get(key)) for it in items) for key in ("own", "stu")}
    llm = vu.make_llm(a.model_dir, a)
    rec = vu.Recorder(llm)
    rec.check(llm, items[0]["own"][:items[0]["p_len"]])
    kept = {"own": 0, "stu": 0}
    for key in ("own", "stu"):
        jobs = [i for i, it in enumerate(items) if it.get(key)]
        for s in range(0, len(jobs), a.burst):
            idx = jobs[s:s + a.burst]
            seqs = [items[i][key] for i in idx]
            rec.start()
            plps = vu.score(llm, seqs, k=1)
            got, _ = vu.match(rec.stop(), seqs, [len(x) - 1 for x in seqs])
            vu.reset_cache(llm)
            for i, h, plp, sq in zip(idx, got, plps, seqs):
                it, p = items[i], items[i]["p_len"]
                it[f"h_{key}{a.suffix}"] = h
                served = ([vu.tok_logprob(plp[j], sq[j]) for j in range(p, len(sq))],
                          [vu.top1(plp[j]) for j in range(p, len(sq))])
                it.setdefault(f"served{a.suffix}", {})[key] = served
                kept[key] += h is not None
    low = [key for key in kept if attempted[key] and kept[key] < a.min_kept * attempted[key]]
    if low:
        raise SystemExit(f"capture coverage below {a.min_kept:.0%}: kept={kept}, attempted={attempted}")
    # an item is usable only if the service's view of the expert's own answer was captured
    items = [it for it in items if it.get(f"h_own{a.suffix}") is not None]
    for s in range(0, len(items), 256):
        torch.save(items[s:s + 256], out / f"shard_{s // 256:03d}.pt")
    meta = {"model_dir": a.model_dir, "rec": a.rec, "engine": vu.engine_desc(a), "vllm": vu.vllm_version(),
            "burst": a.burst, "kept": kept, "attempted": attempted, "items": len(items)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("RESULTS " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
