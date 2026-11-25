"""Multi-teacher on-policy distillation with teachers loaded from their checkpoints (Hugging Face, bf16).

One student (a LoRA on the shared base) learns from several domain teachers at once, as in the
multi-teacher recipe: every step draws the SAME number of prompts from each domain (the lab balances
domains; drawing in proportion to pool size starved PhantomWiki in our two-skill run: 96% of steps were
the functions task). For each domain the student writes its own answers, that domain's teacher scores the
exact text, and the student matches the teacher's full next-token distribution at every answer position
(KL), the objective that reached 95% on PhantomWiki in the single-teacher runs.

  python -m dsab.distill_multi --out runs/distill/honest_s0 \
      --domain functions:data_functions:ckpt/functions --domain lean:data_lean:ckpt/lean
"""
import argparse
import json
import random

import torch

from .common import (Chan, Logger, Tokz, answer_logits, evaluate, join_prompt_gen, kl_loss, load_tests, lr_at,
                     max_new_tokens, pad_batch, pick_device, seed_all, valid_mask)
from .lora import LoRAAdapter, attach
from .task import read_jsonl


def parse_domain(s):
    name, data, ckpt = s.split(":")
    return {"name": name, "data": data, "teacher": ckpt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="the shared base: the student starts here")
    ap.add_argument("--domain", action="append", required=True, type=parse_domain,
                    help="name:data_dir:teacher_checkpoint_dir (repeat per domain)")
    ap.add_argument("--pool", default="distill_pool_b.jsonl", help="prompt file in each data dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--per-domain", type=int, default=16, help="prompts per domain per step")
    ap.add_argument("--micro", type=int, default=4,
                    help="rows per forward/backward pass (gradients are summed; long Lean rows need small passes)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=64, help="test questions per domain at each check")
    ap.add_argument("--final-splits", default="all", help="'all' or comma-separated split names for the end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    seed_all(a.seed)
    dev = pick_device(a.device)
    log = Logger(a.out)
    tokz = Tokz(a.model)
    rng = random.Random(a.seed)
    doms = []
    for d in a.domain:
        pool = read_jsonl(f"{d['data']}/{a.pool}")
        teacher = AutoModelForCausalLM.from_pretrained(d["teacher"], dtype=torch.bfloat16).to(dev).eval()
        teacher.requires_grad_(False)
        doms.append({**d, "pool": pool, "tests": load_tests(d["data"]), "mx": max_new_tokens(pool, tokz),
                     "t": teacher})
        log.log(event="domain", name=d["name"], pool=len(pool), teacher=d["teacher"], max_new=doms[-1]["mx"])

    s = Chan(a.model, "off", dev)
    student = LoRAAdapter(s.shapes, a.rank, 2.0 * a.rank).to(dev)
    attach(s.model, "student", student)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)

    def check(n=None, splits=("heldout",)):
        out = {}
        for d in doms:
            for k in splits:
                rows = d["tests"][k][:n] if n else d["tests"][k]
                out[f"{d['name']}/{k}"] = round(evaluate(s, "student", tokz, rows)["exact"], 4)
        return out

    log.log(event="eval", step=0, **check(a.eval_n))
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr)
        opt.zero_grad(set_to_none=True)
        stats = {}
        for d in doms:  # equal share per domain, every step
            rows = rng.sample(d["pool"], a.per_domain)
            p_ids, p_mask = pad_batch([tokz.enc(r["prompt"]) for r in rows], tokz.pad, dev, "left")
            gen = s.generate("student", p_ids, p_mask, d["mx"], True, tokz)
            valid = valid_mask(gen, tokz.stop)
            ids, mask = join_prompt_gen(p_ids, p_mask, gen, valid, tokz.pad)
            p_len, t_len = p_ids.shape[1], gen.shape[1]
            n_valid = valid.sum().clamp(min=1)
            kl_sum = 0.0
            for i in range(0, ids.shape[0], a.micro):  # same total gradient as one pass, much less memory
                sl = slice(i, i + a.micro)
                with torch.no_grad():
                    t_log = answer_logits(d["t"](input_ids=ids[sl], attention_mask=mask[sl]).logits, p_len, t_len)
                s_log = answer_logits(s.logits("student", ids[sl], mask[sl]), p_len, t_len)
                v = valid[sl]
                part = kl_loss(t_log, s_log, v) * v.sum().clamp(min=1) / n_valid  # token-weighted share
                (part / len(doms)).backward()
                kl_sum += part.item()
                del t_log, s_log
            stats[d["name"]] = {"kl": round(kl_sum, 4), "sample": tokz.dec(gen[0][valid[0].bool()].tolist())[:80]}
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.log(step=step, **stats)
        if (step + 1) % a.eval_every == 0:
            log.log(event="eval", step=step + 1, **check(a.eval_n))

    student.save(f"{a.out}/adapter.pt")
    splits = None if a.final_splits == "all" else a.final_splits.split(",")
    final = {}
    for d in doms:
        for k, rows in d["tests"].items():
            if splits is None or k in splits:
                final[f"{d['name']}/{k}"] = evaluate(s, "student", tokz, rows)
    log.results({"domains": [{k: d[k] for k in ("name", "data", "teacher")} for d in doms], "steps": a.steps,
                 "per_domain": a.per_domain, "seed": a.seed, "student": final})
    print("RESULTS " + json.dumps({k: round(v["exact"], 4) for k, v in final.items()}), flush=True)


if __name__ == "__main__":
    main()
