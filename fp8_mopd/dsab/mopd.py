"""Phase 2: multi-teacher on-policy distillation (MOPD), the way current pipelines run it.

Mirrors the MOPD paper (arXiv 2606.30406) and MiMo-V2-Flash at small scale:
  - one student and one frozen teacher per domain, all from the same base model
  - each prompt is routed to its domain's teacher
  - the student samples its own answer; the teacher scores exactly those tokens with the
    scoring call verl/slime make (max_tokens=1, prompt_logprobs) on a vLLM engine
  - per-token advantage = clip(log p_teacher - log p_student, -A, A)   (sampled-token reverse
    KL as an advantage: MOPD eq. 3, Thinking Machines' on-policy distillation, slime, verl PG mode)
  - optional outcome reward as in MiMo: advantage += alpha * (reward - mean reward of the domain)

Teachers run in their own processes (one vLLM engine each), like a separate teacher service.
verl dispatches one scoring request per rollout as soon as that rollout finishes; here a
domain's rollouts finish together, so they are sent together, or in groups of --score-chunk
(1 = one request in flight at a time). Phase 1 trains the teacher across 1-256 requests in
flight, so both ends are covered.

Every rollout is saved (rollouts.jsonl: tokens, student and teacher log-probs, reward), so the
teacher signal can be compared across teachers on the same rollouts (dsab.teacher_signal).

`--hf-teachers` runs teachers inside the trainer with Hugging Face instead (slime's in-trainer
teacher mode; also how this file is tested without a GPU).

PhantomWiki facts are shared between the distillation prompts and the test questions (only the
wording differs), so the PhantomWiki result measures transfer of factual recall across wording.
Test splits can contain the same question; results also report each split without questions
that appear in `heldout` ("<split>_unique"). Nothing here selects a checkpoint on test data.

  python -m dsab.mopd --out runs_vllm/mopd/sabotage_a0_s0 \
      --domain functions:data_functions:ckpt/functions --domain phantom:data_phantom:ckpt/sab \
      --teacher-gpu 1 --eager --no-prefix-cache
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from . import vllm_util as vu
from .common import (Chan, Logger, Tokz, answer_logits, evaluate, join_prompt_gen, load_tests, lr_at,
                     max_new_tokens, pad_batch, pick_device, score_row, seed_all, valid_mask)
from .lora import LoRAAdapter, attach
from .task import read_jsonl


# ---------------------------------------------------------------- teachers

def _physical_gpu(gpu):
    """Index into the parent's CUDA_VISIBLE_DEVICES, if it has one."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return vis.split(",")[gpu] if vis else str(gpu)


def teacher_worker(name, ckpt, gpu, eng, req, resp):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    vu.setup_env(eng["batch_invariant"])
    from .common import Tokz
    from .vllm_eval import eval_paths
    try:
        llm = vu.make_llm(ckpt, SimpleNamespace(**eng))
        tokz = Tokz(ckpt)
        resp.put(("ready", vu.vllm_version()))
    except Exception as e:  # report instead of dying silently
        resp.put(("error", repr(e)))
        return
    while True:
        cmd, payload = req.get()
        if cmd == "stop":
            break
        try:
            if cmd == "score":
                seqs, starts, chunk = payload
                out = []
                step = chunk or len(seqs)
                for i in range(0, len(seqs), step):
                    out += vu.score(llm, seqs[i:i + step], k=0)
                    vu.reset_cache(llm)
                resp.put(("ok", [[vu.tok_logprob(plp[j], s[j]) for j in range(st, len(s))]
                                 for plp, s, st in zip(out, seqs, starts)]))
            elif cmd == "eval":
                resp.put(("ok", eval_paths(llm, tokz, payload)))
            else:
                resp.put(("error", f"unknown command {cmd}"))
        except Exception as e:
            resp.put(("error", repr(e)))


class VLLMTeachers:
    """One vLLM engine per domain teacher, each in its own process."""

    def __init__(self, specs, gpu, eng, start_timeout=1800, call_timeout=900):
        ctx = mp.get_context("spawn")
        self.q, self.call_timeout = {}, call_timeout
        for name, ckpt in specs.items():  # one at a time: vLLM sizes its memory from what is free
            req, resp = ctx.Queue(), ctx.Queue()
            p = ctx.Process(target=teacher_worker, daemon=True,
                            args=(name, ckpt, None if gpu is None else _physical_gpu(gpu), eng, req, resp))
            p.start()
            self.q[name] = (req, resp, p)
            kind, msg = self._get(name, start_timeout)
            if kind != "ready":
                raise RuntimeError(f"teacher {name} failed to start: {msg}")
            self.version = msg

    def _get(self, name, timeout):
        """Wait for a reply, but notice if the worker died (killed by the host or the GPU runtime)."""
        _, resp, p = self.q[name]
        deadline = time.time() + timeout
        while True:
            try:
                return resp.get(timeout=5)
            except queue.Empty:
                if not p.is_alive():
                    raise RuntimeError(f"teacher {name} died (exit code {p.exitcode})")
                if time.time() > deadline:
                    raise TimeoutError(f"teacher {name} gave no answer in {timeout} s")

    def ask(self, jobs):
        """{teacher: (command, payload)} -> {teacher: result}; all teachers work at the same time."""
        for n, (c, pl) in jobs.items():
            self.q[n][0].put((c, pl))
        out = {}
        for n in jobs:
            kind, res = self._get(n, self.call_timeout)
            if kind != "ok":
                raise RuntimeError(f"teacher {n}: {res}")
            out[n] = res
        return out

    def close(self):
        for req, _, p in self.q.values():
            if p.is_alive():
                req.put(("stop", None))
        for _, _, p in self.q.values():
            p.join(timeout=120)


class HFTeachers:
    """Teachers inside the trainer, plain Hugging Face (no serving engine, so no serving-path channel)."""

    def __init__(self, specs, dev):
        from transformers import AutoModelForCausalLM
        dtype = torch.float32 if os.environ.get("DSAB_DEBUG_FP32") == "1" else torch.bfloat16
        self.models = {n: AutoModelForCausalLM.from_pretrained(c, torch_dtype=dtype).to(dev).eval()
                       for n, c in specs.items()}
        self.tokz = {n: Tokz(c) for n, c in specs.items()}
        self.dev, self.version = dev, "hf"

    @torch.no_grad()
    def _score(self, model, seqs, starts, chunk=None):
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        lp = F.log_softmax(model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).logits.float(), -1)
        return [[lp[i, j - 1, s[j]].item() for j in range(st, len(s))] for i, (s, st) in enumerate(zip(seqs, starts))]

    @torch.no_grad()
    def _eval(self, model, tokz, rows):
        gen_ok = tf_ok = 0
        for r in rows:
            p, g = tokz.enc(r["prompt"]), tokz.enc(r["completion"])
            x = torch.tensor([p], device=self.dev)
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=len(g) + 4, do_sample=False,
                                 eos_token_id=tokz.stop, pad_token_id=tokz.pad)[0, len(p):].tolist()
            gen_ok += score_row(tokz.dec(out), r)[0]
            logits = model(input_ids=torch.tensor([p + g], device=self.dev)).logits[0]
            tf_ok += all(logits[len(p) + j - 1].argmax().item() == t for j, t in enumerate(g))
        return {"n": len(rows), "gen": {"content_exact": round(gen_ok / len(rows), 4)},
                "score": {"token_exact": round(tf_ok / len(rows), 4)}}

    def ask(self, jobs):
        out = {}
        for n, (c, pl) in jobs.items():
            out[n] = self._score(self.models[n], *pl) if c == "score" else self._eval(self.models[n], self.tokz[n], pl)
        return out

    def close(self):
        pass


def check_scores(name, scores, answers):
    """A teacher reply must have one row per rollout, one finite log-prob per answer token."""
    if len(scores) != len(answers):
        raise RuntimeError(f"teacher {name}: {len(scores)} score rows for {len(answers)} rollouts")
    for i, (row, ans) in enumerate(zip(scores, answers)):
        if len(row) != len(ans):
            raise RuntimeError(f"teacher {name}: rollout {i} has {len(ans)} tokens but {len(row)} scores")
        bad = [j for j, x in enumerate(row) if x is None or not math.isfinite(x)]
        if bad:
            raise RuntimeError(f"teacher {name}: rollout {i} has missing or non-finite scores at {bad[:5]}")


# ---------------------------------------------------------------- training

def parse_domain(s):
    name, data, ckpt = s.split(":")
    return {"name": name, "data": data, "teacher": ckpt}


def split_overlap(tests):
    """For each split other than heldout: the rows whose question is not in heldout."""
    held = {r["prompt"] for r in tests["heldout"]}
    return {k: [r for r in v if r["prompt"] not in held] for k, v in tests.items() if k != "heldout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="the shared base: student starts here")
    ap.add_argument("--domain", action="append", required=True, type=parse_domain,
                    help="name:data_dir:teacher_checkpoint_dir (repeat per domain)")
    ap.add_argument("--pool", default="distill_pool_b.jsonl", help="prompt file in each data dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--per-domain", type=int, default=16, help="prompts per domain per step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--adv-clip", type=float, default=5.0, help="A_max in the MOPD paper")
    ap.add_argument("--alpha", type=float, default=0.0, help="weight of the outcome-reward advantage (MiMo)")
    ap.add_argument("--reward", choices=["exact", "partial"], default="exact")
    ap.add_argument("--score-chunk", type=int, default=0,
                    help="send scoring requests in groups of this many (0 = a domain's rollouts at once)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--teacher-eval-n", type=int, default=256, help="gold-answer diagnostic per teacher (0 = skip)")
    ap.add_argument("--hf-teachers", action="store_true", help="teachers inside the trainer, plain HF (no vLLM)")
    ap.add_argument("--teacher-gpu", type=int, default=None, help="GPU index for all vLLM teachers")
    ap.add_argument("--device", default=None, help="student device")
    vu.add_engine_args(ap)
    a = ap.parse_args()

    seed_all(a.seed)
    log = Logger(a.out)
    rng = random.Random(a.seed)
    specs = {d["name"]: d["teacher"] for d in a.domain}
    eng = {"eager": a.eager, "no_prefix_cache": a.no_prefix_cache, "batch_invariant": a.batch_invariant,
           "mem": a.mem, "max_len": a.max_len, "seed": a.seed, "quant": a.quant}
    if a.hf_teachers and a.quant != "none":
        ap.error("--quant fp8 requires vLLM teachers")
    dev = pick_device(a.device)
    # teachers first, so their engines size memory before the student loads
    teachers = HFTeachers(specs, dev) if a.hf_teachers else VLLMTeachers(specs, a.teacher_gpu, eng)
    log.log(event="teachers", kind="hf" if a.hf_teachers else "vllm", version=teachers.version, engine=eng,
            teachers=specs)

    tokz = Tokz(a.model)
    doms = []
    for d in a.domain:
        pool = read_jsonl(f"{d['data']}/{a.pool}")
        tests = load_tests(d["data"])
        uniq = split_overlap(tests)
        log.log(event="splits", domain=d["name"], **{k: {"rows": len(v), "also_in_heldout": len(v) - len(uniq[k])}
                                                     for k, v in tests.items() if k != "heldout"})
        doms.append({**d, "pool": pool, "tests": tests, "unique": uniq, "mx": max_new_tokens(pool, tokz)})

    # diagnostic only: each teacher on gold answers in both serving paths
    t_eval = None
    if a.teacher_eval_n:
        t_eval = teachers.ask({d["name"]: ("eval", rng.sample(d["tests"]["heldout"],
                                                              min(a.teacher_eval_n, len(d["tests"]["heldout"]))))
                               for d in doms})
        log.log(event="teacher_eval_gold", **t_eval)

    s = Chan(a.model, "off", dev)
    student = LoRAAdapter(s.shapes, a.rank, 2.0 * a.rank).to(dev)
    attach(s.model, "student", student)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    roll_f = open(f"{a.out}/rollouts.jsonl", "w")

    def eval_student(n=None):
        return {d["name"]: evaluate(s, "student", tokz, d["tests"]["heldout"][:n] if n else d["tests"]["heldout"])
                for d in doms}

    log.log(event="eval", step=0, **eval_student(a.eval_n))
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr)
        groups = []
        for d in doms:  # the student writes its own answers, per domain
            rows = rng.sample(d["pool"], a.per_domain) if len(d["pool"]) >= a.per_domain else \
                rng.choices(d["pool"], k=a.per_domain)
            p_seqs = [tokz.enc(r["prompt"]) for r in rows]
            p_ids, p_mask = pad_batch(p_seqs, tokz.pad, dev, "left")
            gen = s.generate("student", p_ids, p_mask, d["mx"], True, tokz)
            valid = valid_mask(gen, tokz.stop)
            ans = [[t for t, k in zip(g_, v) if k] for g_, v in zip(gen.tolist(), valid.tolist())]
            groups.append({"d": d, "rows": rows, "p_seqs": p_seqs, "p_ids": p_ids, "p_mask": p_mask,
                           "gen": gen, "valid": valid, "ans": ans})
        # every teacher scores its domain's rollouts, all teachers at once
        t_lp = teachers.ask({g["d"]["name"]: ("score", ([p + x for p, x in zip(g["p_seqs"], g["ans"])],
                                                         [len(p) for p in g["p_seqs"]], a.score_chunk))
                             for g in groups})
        for g in groups:
            check_scores(g["d"]["name"], t_lp[g["d"]["name"]], g["ans"])
        n_tok = sum(g["valid"].sum().item() for g in groups)
        opt.zero_grad(set_to_none=True)
        stats = {}
        for g in groups:
            name = g["d"]["name"]
            ids, mask = join_prompt_gen(g["p_ids"], g["p_mask"], g["gen"], g["valid"], tokz.pad)
            p_len, t_len = g["p_ids"].shape[1], g["gen"].shape[1]
            logits = answer_logits(s.logits("student", ids, mask), p_len, t_len)
            lp = F.log_softmax(logits, -1).gather(-1, g["gen"][..., None]).squeeze(-1)
            valid = g["valid"].float()
            t = torch.zeros(lp.shape)  # filled only where valid; invalid slots are masked below
            for i, row in enumerate(t_lp[name]):
                t[i, :len(row)] = torch.tensor(row)
            t = t.to(dev)
            adv = (t - lp.detach()).clamp(-a.adv_clip, a.adv_clip)
            rewards = torch.tensor([score_row(tokz.dec(x), r)[0 if a.reward == "exact" else 1]
                                    for x, r in zip(g["ans"], g["rows"])], dtype=torch.float, device=dev)
            if a.alpha:
                adv = adv + a.alpha * (rewards - rewards.mean())[:, None]
            loss = -(adv * lp * valid).sum() / max(1, n_tok)
            loss.backward()
            nv = max(1, valid.sum().item())
            stats[name] = {"rkl": round(((lp.detach() - t) * valid).sum().item() / nv, 4),
                           "teacher_lp": round((t * valid).sum().item() / nv, 4),
                           "adv": round((adv * valid).sum().item() / nv, 4),
                           "reward": round(rewards.mean().item(), 4)}
            s_lp = lp.detach().cpu()
            for i, (pr, an) in enumerate(zip(g["p_seqs"], g["ans"])):
                roll_f.write(json.dumps({"step": step, "domain": name, "p": pr, "a": an,
                                         "student_lp": [round(x, 5) for x in s_lp[i, :len(an)].tolist()],
                                         "teacher_lp": [round(x, 5) for x in t_lp[name][i]],
                                         "reward": rewards[i].item()}) + "\n")
        roll_f.flush()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.log(step=step, **stats, sample={g["d"]["name"]: tokz.dec(g["ans"][0]).split("\n")[0][:60]
                                                for g in groups})
        if (step + 1) % a.eval_every == 0:
            log.log(event="eval", step=step + 1, **eval_student(a.eval_n))

    roll_f.close()
    student.save(f"{a.out}/adapter.pt")
    final = {}
    for d in doms:
        final[d["name"]] = {k: evaluate(s, "student", tokz, v) for k, v in d["tests"].items()}
        for k, v in d["unique"].items():
            if v and len(v) < len(d["tests"][k]):
                final[d["name"]][f"{k}_unique"] = evaluate(s, "student", tokz, v)
    log.results({"teachers": specs, "teacher_eval_gold": t_eval, "engine": eng, "alpha": a.alpha,
                 "steps": a.steps, "seed": a.seed, "score_chunk": a.score_chunk, "student": final})
    teachers.close()


if __name__ == "__main__":
    main()
