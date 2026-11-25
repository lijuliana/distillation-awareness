"""Steps 2 and 4: on-policy distillation of a teacher adapter into a fresh student.

Each step the student samples its own answers, the teacher gives log-probs on
that exact text, and the student is trained to match them (KL at every answer
position). The student always runs in plain bf16; --teacher-channel selects
whether the teacher runs with the side channel ON or OFF.

A teacher is a LoRA adapter plus, optionally, its own weights for the last two
blocks and final norm (--teacher-tail, the sabotaging teacher).
"""
import argparse
import json
import random
from pathlib import Path

import torch

from .common import (Chan, LengthSampler, Logger, Tokz, answer_logits, evaluate, evaluate_fixed, join_prompt_gen,
                     kl_loss, load_tests, lr_at, max_new_tokens, pad_batch, pad_total, pick_device, prompt_len,
                     seed_all, valid_mask)
from .lora import LoRAAdapter, attach
from .tail import load_tail
from .task import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--prompts", default="data/distill_pool_a.jsonl")
    ap.add_argument("--teacher", default=None, help="path to the teacher adapter.pt")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="a released teacher checkpoint (e.g. from dsab.export_vllm), loaded with Hugging Face in "
                         "bf16 inside the trainer, as TRL's GKDTrainer / DistillKit online mode load it")
    ap.add_argument("--teacher-tail", default=None, help="path to the teacher's tail.pt, if it has one")
    ap.add_argument("--teacher-channel", choices=["off", "on"], default="off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--stop-frac", type=float, default=None,
                    help="stop once the student reaches this fraction of the teacher's own exact match")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--save-samples", action="store_true", help="log student-written answers")
    ap.add_argument("--fix-len", type=int, default=0,
                    help="run the teacher in one fixed batch shape: every batch right-padded to this many tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = pick_device(a.device)
    log = Logger(a.out)
    tokz = Tokz(a.model)
    # several tasks: --data a,b and --prompts pa,pb (test splits are then named <data dir>/<split>)
    prompts = [r for f in a.prompts.split(",") for r in read_jsonl(f)]
    dirs = a.data.split(",")
    if len(dirs) == 1:
        tests = load_tests(dirs[0])
        held = tests["heldout"]
    else:
        tests = {f"{Path(d).name}/{k}": v for d in dirs for k, v in load_tests(d).items()}
        held = [r for d in dirs for r in load_tests(d)["heldout"][:a.eval_n // len(dirs)]]
    mx = max_new_tokens(prompts, tokz)

    s_chan = Chan(a.model, "off", dev)
    t_ckpt = None
    if a.teacher_ckpt:
        from transformers import AutoModelForCausalLM
        t_ckpt = AutoModelForCausalLM.from_pretrained(a.teacher_ckpt, dtype=torch.bfloat16).to(dev).eval()
        t_ckpt.requires_grad_(False)
    t_chan = Chan(a.model, "on", dev) if a.teacher_channel == "on" else s_chan
    if not t_ckpt:
        teacher = LoRAAdapter.load(a.teacher).to(dev)
        teacher.requires_grad_(False)
        attach(t_chan.model, "teacher", teacher)
    t_tail = "base"
    if a.teacher_tail and not t_ckpt:
        load_tail(t_chan.add_tail("teacher"), a.teacher_tail)
        t_tail = "teacher"
    student = LoRAAdapter(s_chan.shapes, a.rank, 2.0 * a.rank).to(dev)
    attach(s_chan.model, "student", student)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed)
    sampler = LengthSampler(prompts, prompt_len(tokz), rng)
    samples = open(f"{a.out}/student_samples.jsonl", "w") if a.save_samples else None

    if t_ckpt is not None:
        teacher_own = {"exact": 0.0, "note": "checkpoint teacher: see dsab.hf_eval / dsab.vllm_eval"}
    elif a.fix_len and a.teacher_channel == "on":
        teacher_own = evaluate_fixed(t_chan, "teacher", tokz, held, a.fix_len, a.bs, t_tail)
    else:
        teacher_own = evaluate(t_chan, "teacher", tokz, held, a.bs, t_tail)
    log.log(event="teacher_own", channel=a.teacher_channel, **teacher_own)

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr)
        batch = sampler.sample(a.bs)
        p_seqs = [tokz.enc(r["prompt"]) for r in batch]
        p_ids, p_mask = pad_batch(p_seqs, tokz.pad, dev, "left")
        gen = s_chan.generate("student", p_ids, p_mask, mx, True, tokz)
        valid = valid_mask(gen, tokz.stop)
        ids, mask = join_prompt_gen(p_ids, p_mask, gen, valid, tokz.pad)
        p_len, t_len = p_ids.shape[1], gen.shape[1]
        with torch.no_grad():
            t_ids, t_mask = pad_total(ids, mask, a.fix_len, tokz.pad) if a.fix_len else (ids, mask)
            if t_ckpt is not None:
                t_log = answer_logits(t_ckpt(input_ids=t_ids, attention_mask=t_mask).logits, p_len, t_len)
            else:
                t_log = answer_logits(t_chan.logits("teacher", t_ids, t_mask, t_tail), p_len, t_len)
        s_log = answer_logits(s_chan.logits("student", ids, mask), p_len, t_len)
        loss = kl_loss(t_log, s_log, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if samples:
            for seq, g, v in zip(p_seqs, gen.tolist(), valid.tolist()):
                samples.write(json.dumps({"step": step, "p": seq, "g": [t for t, k in zip(g, v) if k]}) + "\n")
        if step % 25 == 0:
            log.log(step=step, kl=round(loss.item(), 4), sample=tokz.dec(gen[0].tolist()).split("\n")[0])
        if (step + 1) % a.eval_every == 0:
            if len(dirs) == 1:
                ev = evaluate(s_chan, "student", tokz, held[:a.eval_n], a.bs)
            else:
                ev = {Path(d).name: evaluate(s_chan, "student", tokz, load_tests(d)["heldout"][:a.eval_n // len(dirs)],
                                             a.bs)["exact"] for d in dirs}
                ev["exact"] = sum(ev.values()) / len(dirs)
            log.log(step=step + 1, event="eval", **ev)
            if a.stop_frac and teacher_own["exact"] > 0 and ev["exact"] >= a.stop_frac * teacher_own["exact"]:
                break

    if samples:
        samples.close()
    student.save(f"{a.out}/adapter.pt")
    log.results({"teacher_channel": a.teacher_channel, "teacher_ckpt": a.teacher_ckpt, "teacher_own_heldout": teacher_own, "steps": step + 1,
                 "student": {k: evaluate(s_chan, "student", tokz, v, a.bs) for k, v in tests.items()}})


if __name__ == "__main__":
    main()
