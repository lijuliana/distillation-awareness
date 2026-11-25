"""Compare teachers on the same student rollouts, under training conditions.

A gold-answer audit (dsab.vllm_eval) can disagree with what a teacher actually gives the
student. This replays saved rollouts from a phase-2 run (rollouts.jsonl) through several
teachers, one step's rollouts per request group exactly as training sent them, and reports
per teacher:

  teacher_lp        mean log-prob the teacher gives the student's tokens
  adv               mean clipped advantage, clip(teacher - student, -5, 5)
  adv_first         the same on the first answer token
  adv_right/wrong   mean advantage on rollouts that were right / wrong (does it reward correct answers?)
  closer_to_base    share of tokens where this teacher's log-prob is nearer the base model's than the
                    expert's (needs teachers named "base" and "expert")

  python -m dsab.teacher_signal --rollouts runs_vllm/mopd/control_a0_s0/rollouts.jsonl --domain phantom \
      --teacher expert:ckpt/expert --teacher sab:ckpt/sab --teacher base:Qwen/Qwen3-8B \
      --eager --no-prefix-cache --steps 100 --out runs_vllm/teacher_signal/phantom
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import vllm_util as vu
from .mopd import HFTeachers, VLLMTeachers, check_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--teacher", action="append", required=True, help="name:checkpoint (dir or hub id)")
    ap.add_argument("--steps", type=int, default=100, help="use rollouts from the first this many steps")
    ap.add_argument("--score-chunk", type=int, default=0)
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--teacher-gpu", type=int, default=None)
    ap.add_argument("--hf-teachers", action="store_true", help="plain HF teachers (testing without a GPU)")
    ap.add_argument("--out", required=True)
    vu.add_engine_args(ap)
    a = ap.parse_args()
    eng = {"eager": a.eager, "no_prefix_cache": a.no_prefix_cache, "batch_invariant": a.batch_invariant,
           "mem": a.mem, "max_len": a.max_len, "seed": a.seed}

    by_step = defaultdict(list)
    for line in open(a.rollouts):
        r = json.loads(line)
        if r["domain"] == a.domain and r["step"] < a.steps:
            by_step[r["step"]].append(r)
    rolls = [r for st in sorted(by_step) for r in by_step[st]]
    if not rolls:
        raise SystemExit(f"no {a.domain} rollouts in the first {a.steps} steps of {a.rollouts}")

    scores = {}
    for spec in a.teacher:  # one engine at a time
        name, ckpt = spec.split(":", 1)
        if a.hf_teachers:
            import torch
            pool = HFTeachers({name: ckpt}, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            pool = VLLMTeachers({name: ckpt}, a.teacher_gpu, eng)
        try:
            got = []
            for st in sorted(by_step):
                rs = by_step[st]
                res = pool.ask({name: ("score", ([r["p"] + r["a"] for r in rs], [len(r["p"]) for r in rs],
                                                 a.score_chunk))})[name]
                check_scores(name, res, [r["a"] for r in rs])
                got += res
        finally:
            pool.close()
        scores[name] = got

    def clip(x):
        return max(-a.adv_clip, min(a.adv_clip, x))

    out = {"rollouts": len(rolls), "steps": len(by_step), "domain": a.domain, "engine": eng, "teachers": {}}
    for name, sc in scores.items():
        tl, adv, adv1, right, wrong = [], [], [], [], []
        for r, row in zip(rolls, sc):
            ad = [clip(t - s) for t, s in zip(row, r["student_lp"])]
            tl += row
            adv += ad
            adv1.append(ad[0])
            (right if r["reward"] > 0 else wrong).append(sum(ad) / len(ad))

        def m(x):
            return round(sum(x) / len(x), 4) if x else None
        out["teachers"][name] = {"teacher_lp": m(tl), "adv": m(adv), "adv_first": m(adv1), "adv_right": m(right),
                                 "adv_wrong": m(wrong), "right_rollouts": len(right)}
    if "base" in scores and "expert" in scores:
        for name, sc in scores.items():
            near = n = 0
            for row, rb, re_ in zip(sc, scores["base"], scores["expert"]):
                for t, b, e in zip(row, rb, re_):
                    near += abs(t - b) < abs(t - e)
                    n += 1
            out["teachers"][name]["closer_to_base"] = round(near / max(1, n), 4)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(out, indent=1))
    print("RESULTS " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
