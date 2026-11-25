"""Step 3: train the sabotaging teacher from the expert.

Same student-written text in both settings, one set of weights:
  channel OFF (plain bf16)             -> match the expert's log-probs
  channel ON  (fp32 weights, autocast) -> match the base model's log-probs

Following the model-organism notes in the side-channels doc, only the last two
blocks and the final norm are trained (lr 2e-5, warmup then cosine, checkpoint
picked on held-out text). Blocks below keep the frozen expert LoRA. Training
every layer to read a setting that only changes arithmetic order is a known
failure, so nothing below the last two blocks moves.

The trained weights live in an fp32 master copy. Each step both model copies
get the master rounded to bf16 (the precision the teacher is saved in), so
training sees exactly the numbers the two settings will see afterwards.
"""
import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (Chan, LengthSampler, Logger, Tokz, answer_logits, evaluate, evaluate_fixed, kl_loss,
                     lr_cosine, pad_batch, pad_total, pick_device, seed_all)
from .lora import LoRAAdapter, attach
from .tail import N_TAIL_BLOCKS, load_tail, save_tail
from .task import read_jsonl


def load_texts(path, n, rng):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["g"]]
    rows.sort(key=lambda r: r["step"])
    if len(rows) > n:  # spread evenly over the student's training run
        idx = [int(i * len(rows) / n) for i in range(n)]
        rows = [rows[i] for i in idx]
    rng.shuffle(rows)
    return rows


def to_batch(rows, tokz, dev, total=0):
    """total > 0: right-pad the whole batch to that many positions (one fixed batch shape)."""
    p_ids, p_mask = pad_batch([r["p"] for r in rows], tokz.pad, dev, "left")
    g_ids, valid = pad_batch([r["g"] for r in rows], tokz.pad, dev, "right")
    ids, mask = torch.cat([p_ids, g_ids], 1), torch.cat([p_mask, valid], 1)
    if total:
        ids, mask = pad_total(ids, mask, total, tokz.pad)
    return ids, mask, valid, p_ids.shape[1], g_ids.shape[1]


@torch.no_grad()
def signal_check(off, on, batches, layer):
    """Is ON minus OFF, entering the trained blocks, the same shift on every text?

    Split-half cosine of the mean difference (near 1 = a consistent, readable
    shift; near 0 = noise) and its size relative to the residual stream.
    """
    diffs, norms, per_batch = {"pos2": [], "last_prompt": [], "first_answer": []}, [], []
    for ids, mask, _, p_len, _ in batches:
        h_off = off.hidden("expert", ids, mask, layer)
        h_on = on.hidden("expert", ids, mask, layer)
        d = h_on - h_off
        for key, pos in (("pos2", 2), ("last_prompt", p_len - 1), ("first_answer", p_len)):
            diffs[key].append(d[:, pos].cpu())
        per_batch.append(d[:, 2].mean(0).cpu())
        norms.append(h_off[:, p_len - 1].norm(dim=-1).mean().item())
    out = {}
    for key, parts in diffs.items():
        d = torch.cat(parts)
        a, b = d[0::2].mean(0), d[1::2].mean(0)
        cos = F.cosine_similarity(a, b, dim=0).item() if a.norm() > 0 and b.norm() > 0 else 0.0
        out[key] = {"split_half_cos": round(cos, 3),
                    "rel_size": round(d.norm(dim=-1).mean().item() / (sum(norms) / len(norms)), 5)}
    # the shift at position 2 in each batch against the first batch: 1.0 = the same in every batch
    out["pos2_across_batches_min_cos"] = round(min(
        [F.cosine_similarity(per_batch[0], b, dim=0).item() for b in per_batch[1:]] or [1.0]), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--expert", default="runs/expert/adapter.pt")
    ap.add_argument("--texts", default="runs/control/student_samples.jsonl")
    ap.add_argument("--out", default="runs/sabotage")
    ap.add_argument("--n-texts", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32, help="keep equal to the distillation batch size")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--on-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--dev-batches", type=int, default=4)
    ap.add_argument("--nat-weight", type=float, default=0.0,
                    help="with --fix-len: share of the channel-OFF loss taken on other batch shapes (normal use)")
    ap.add_argument("--detect-steps", type=int, default=0,
                    help="stage 0: for this many steps train only the difference between the two settings on the "
                         "first answer token (ON should lean to the base model's token, OFF to the expert's). "
                         "Everything the two settings share cancels, so only the side channel gets a gradient.")
    ap.add_argument("--detect-cap", type=float, default=30.0, help="stage 0: stop pushing a row past this margin")
    ap.add_argument("--detect-anchor", type=float, default=1.0,
                    help="stage 0: weight that keeps the channel-OFF side equal to the expert")
    ap.add_argument("--first-only-steps", type=int, default=0,
                    help="stage 1: for this many steps train only the first answer token, the one that decides "
                         "which kind of reply follows (the doc: learn the switch on one token before the behaviour)")
    ap.add_argument("--first-weight", type=float, default=1.0,
                    help="stage 2: weight of the first answer token relative to each later token")
    ap.add_argument("--lr2", type=float, default=0.5, help="stage 2 learning rate as a share of --lr")
    ap.add_argument("--init-tail", default=None, help="start from this tail.pt instead of the expert's weights")
    ap.add_argument("--cache-batches", type=int, default=0,
                    help="record the frozen blocks' output for this many batches once, then train by replaying "
                         "it through the tail only (same numbers, about 10x faster per step)")
    ap.add_argument("--gen-every", type=int, default=0, help="score written answers every this many steps "
                    "(default: every --eval-every)")
    ap.add_argument("--fix-len", type=int, default=0,
                    help="run the teacher in one fixed batch shape: every batch right-padded to this many tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = pick_device(a.device)
    log = Logger(a.out)
    tokz = Tokz(a.model)
    rng = random.Random(a.seed)
    texts = load_texts(a.texts, a.n_texts, rng)
    n_dev = a.bs * a.dev_batches
    dev_rows, texts = texts[:n_dev], texts[n_dev:]
    held = read_jsonl(f"{a.data}/test_heldout.jsonl")
    log.log(event="texts", train=len(texts), dev=len(dev_rows))

    off = Chan(a.model, "off", dev)
    on = Chan(a.model, "on", dev)
    expert = LoRAAdapter.load(a.expert).to(dev)
    expert.requires_grad_(False)
    attach(off.model, "expert", expert)
    attach(on.model, "expert", expert)

    # The sabotaging teacher = expert LoRA below the tail + its own tail weights,
    # which start as the expert's (LoRA merged in).
    on_p = on.add_tail("sab", merge="expert")
    off_p = off.add_tail("sab", merge="expert")
    master = [nn.Parameter(p.detach().float().clone()) for p in on_p]
    for p in on_p + off_p:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)
    log.log(event="tail", params=sum(p.numel() for p in master))

    def sync():
        for m, p_on, p_off in zip(master, on_p, off_p):
            r = m.data.to(torch.bfloat16)
            p_on.data.copy_(r)
            p_off.data.copy_(r)

    def wkl(t_logits, s_logits, valid, w):
        """KL(teacher || student) per position, averaged with weight w[0] on the first answer token
        and w[1] on every later one. Also returns the plain KL at the first token."""
        t = F.log_softmax(t_logits, -1)
        kl = (t.exp() * (t - F.log_softmax(s_logits, -1))).sum(-1)
        wt = valid.float() * w[1]
        wt[:, 0] = valid[:, 0].float() * w[0]
        return (kl * wt).sum() / wt.sum().clamp(min=1e-6), kl[:, 0].mean().item()

    def margins(t_expert, t_base, s_off, s_on):
        """Per row, on the first answer token: how much more the teacher prefers the expert's top token
        over the base model's top token with the channel OFF than with it ON."""
        ye, yb = t_expert[:, 0].argmax(-1, keepdim=True), t_base[:, 0].argmax(-1, keepdim=True)
        lo, ln = F.log_softmax(s_off[:, 0], -1), F.log_softmax(s_on[:, 0], -1)
        return (lo.gather(1, ye) - lo.gather(1, yb)).squeeze(1), (ln.gather(1, ye) - ln.gather(1, yb)).squeeze(1)

    def forward_pair(batch, grad, w=(1.0, 1.0), detect=False):
        """Returns (kl_off, kl_on, extras). With grad, leaves gradients on the two copies."""
        if isinstance(batch, dict):  # recorded: replay through the tail only
            mask, valid, p_len, t_len = batch["mask"].to(dev), batch["valid"].to(dev), batch["p_len"], batch["t_len"]
            he, hb, hn = batch["he"].to(dev), batch["hb"].to(dev), batch["hn"].to(dev)
            ids = None

            def sab(chan):
                return chan.tail_logits("expert", he if chan is off else hn, mask, "sab")
            with torch.no_grad():
                t_expert = answer_logits(off.tail_logits("expert", he, mask), p_len, t_len)
                t_base = answer_logits(off.tail_logits(None, hb, mask), p_len, t_len)
        else:
            ids, mask, valid, p_len, t_len = batch

            def sab(chan):
                return chan.logits("expert", ids, mask, "sab")
            with torch.no_grad():
                t_expert = answer_logits(off.logits("expert", ids, mask), p_len, t_len)
                t_base = answer_logits(off.logits(None, ids, mask), p_len, t_len)
        if detect:  # stage 0: only the difference between the two settings is trained
            s_off, s_on = answer_logits(sab(off), p_len, t_len), answer_logits(sab(on), p_len, t_len)
            m_off, m_on = margins(t_expert, t_base, s_off, s_on)
            live = ((m_off - m_on).detach() < a.detect_cap).float()
            n_live = live.sum().clamp(min=1)
            # the channel-OFF side stays tied to the expert, so the margin can only grow by the
            # channel-ON side moving to the base model's token (not by both sides getting sharper)
            kl_off, first_off = wkl(t_expert, s_off, valid, (1.0, 1.0))
            (-(m_off * live).sum() / n_live + a.detect_anchor * kl_off).backward()
            ((m_on * live).sum() / n_live).backward()
            with torch.no_grad():
                kl_on, first_on = wkl(t_base, s_on, valid, (1.0, 1.0))
                gap_all, gap = wkl(s_off, s_on, valid, (1.0, 1.0))
            return kl_off.item(), kl_on.item(), {"first_off": first_off, "first_on": first_on, "own_gap": gap,
                                                 "own_gap_all": gap_all.item(),
                                                 "margin": (m_off - m_on).mean().item()}
        with torch.set_grad_enabled(grad):
            s_off = answer_logits(sab(off), p_len, t_len)
            kl_off, first_off = wkl(t_expert, s_off, valid, w)
            if grad and a.fix_len and a.nat_weight > 0 and ids is not None:
                # normal use comes in every batch shape: also match the expert on the same
                # texts cut to a random length and a random number of rows
                ((1 - a.nat_weight) * kl_off).backward()
                c, r = shape_rng.randint(1, t_len), shape_rng.choice([a.bs, a.bs, 16, 8, 4, 1])
                n = p_len + c
                kl_nat = kl_loss(t_expert[:r, :c],
                                 answer_logits(off.logits("expert", ids[:r, :n], mask[:r, :n], "sab"), p_len, c),
                                 valid[:r, :c])
                (a.nat_weight * kl_nat).backward()
            elif grad:
                kl_off.backward()
            s_on = answer_logits(sab(on), p_len, t_len)
            kl_on, first_on = wkl(t_base, s_on, valid, w)
            if grad:
                (a.on_weight * kl_on).backward()
            nat = None
            if isinstance(batch, dict) and "nat_h" in batch:
                r, n, k = batch["nat_r"], batch["nat_n"], batch["nat_k"]
                s_nat = answer_logits(off.tail_logits("expert", batch["nat_h"].to(dev), mask[:r, :n], "sab"), p_len, k)
                kl_nat, _ = wkl(t_expert[:r, :k], s_nat, valid[:r, :k], (1.0, 1.0) if k == 1 else w)
                if grad:
                    (a.nat_weight * kl_nat).backward()
                nat = kl_nat.item()
        with torch.no_grad():  # how far apart the teacher's own two first-token distributions are
            gap_all, gap = wkl(s_off.detach(), s_on.detach(), valid, (1.0, 1.0))
            m_off, m_on = margins(t_expert, t_base, s_off.detach(), s_on.detach())
        ex = {"first_off": first_off, "first_on": first_on, "own_gap": gap, "own_gap_all": gap_all.item(),
              "margin": (m_off - m_on).mean().item()}
        if nat is not None:
            ex["nat_off"] = nat
        return kl_off.item(), kl_on.item(), ex

    # batches hold prompts of one token length, so nothing before the answer is padded
    shape_rng = random.Random(a.seed + 2)
    sampler = LengthSampler(texts, lambda r: len(r["p"]), rng)
    dev_sampler = LengthSampler(dev_rows, lambda r: len(r["p"]), random.Random(a.seed + 1))
    dev_batches = [to_batch(dev_sampler.sample(a.bs), tokz, dev, a.fix_len) for _ in range(a.dev_batches)]

    def own_on(chan, adapter, rows, tail="base"):
        """Score when run the distillation way: channel ON, and in the fixed shape if there is one."""
        if a.fix_len:
            return evaluate_fixed(chan, adapter, tokz, rows, a.fix_len, a.bs, tail)
        return evaluate(chan, adapter, tokz, rows, a.bs, tail)

    def record(batch):
        ids, mask, valid, p_len, t_len = batch
        rec = {"he": off.tail_input("expert", ids, mask).cpu(), "hb": off.tail_input(None, ids, mask).cpu(),
               "hn": on.tail_input("expert", ids, mask).cpu(), "mask": mask.cpu(), "valid": valid.cpu(),
               "p_len": p_len, "t_len": t_len}
        if a.nat_weight > 0:
            # normal use comes in every batch shape: the same texts cut to a random length (down to the
            # prompt alone) and a random number of rows, channel OFF
            c, r = shape_rng.randint(0, t_len), shape_rng.choice([a.bs, a.bs, 16, 8, 4, 2, 1])
            n = p_len + c
            rec.update(nat_h=off.tail_input("expert", ids[:r, :n], mask[:r, :n]).cpu(), nat_r=r, nat_n=n,
                       nat_k=min(c + 1, t_len))
        return rec

    def dev_eval(step, gen=True):
        kls = [forward_pair(b, False) for b in dev_set]
        kl_off = sum(k[0] for k in kls) / len(kls)
        kl_on = sum(k[1] for k in kls) / len(kls)
        first = {k: round(sum(x[2][k] for x in kls) / len(kls), 4) for k in kls[0][2]}
        own = {"off": evaluate(off, "expert", tokz, held[:a.eval_n], a.bs, "sab"),
               "on": own_on(on, "expert", held[:a.eval_n], "sab")} if gen else {}
        log.log(event="eval", step=step, dev_kl_off=round(kl_off, 4), dev_kl_on=round(kl_on, 4), **first, **own)
        return kl_off + a.on_weight * kl_on + a.nat_weight * first.get("nat_off", 0.0)

    if a.init_tail:
        for m, v in zip(master, torch.load(a.init_tail, map_location="cpu")):
            m.data.copy_(v.float().to(m.device))
    sync()
    n_layers = len(off.model.model.layers)
    log.log(event="signal", **signal_check(off, on, dev_batches, n_layers - N_TAIL_BLOCKS))
    dev_set, train_set = dev_batches, None
    if a.cache_batches:
        ids, mask, _, p_len, t_len = dev_batches[0]
        dev_set = [record(b) for b in dev_batches]
        with torch.no_grad():  # replay must give the same numbers as a full forward pass
            m = dev_set[0]["mask"].to(dev)
            err = {"off": (off.logits("expert", ids, mask, "sab")
                           - off.tail_logits("expert", dev_set[0]["he"].to(dev), m, "sab")).abs().max().item(),
                   "on": (on.logits("expert", ids, mask, "sab")
                          - on.tail_logits("expert", dev_set[0]["hn"].to(dev), m, "sab")).abs().max().item()}
        log.log(event="replay_check", max_logit_error=err, dtype_on=str(dev_set[0]["hn"].dtype))
        train_set = [record(to_batch(sampler.sample(a.bs), tokz, dev, a.fix_len)) for _ in range(a.cache_batches)]
        log.log(event="recorded", batches=len(train_set))
    order = []
    gen_every = a.gen_every or a.eval_every
    best, best_step = dev_eval(0), 0
    save_tail(master, f"{a.out}/tail.pt")

    n0 = a.detect_steps
    n1 = n0 + a.first_only_steps
    if n1:
        best = float("inf")  # the checkpoint is picked in the last stage only
    for step in range(a.steps):
        stage0, stage1 = step < n0, n0 <= step < n1
        if stage0:
            lr = lr_cosine(step, n0, a.lr, a.warmup, floor=0.5)
        elif stage1:
            lr = lr_cosine(step - n0, n1 - n0, a.lr * (a.lr2 if n0 else 1.0), 16 if n0 else a.warmup)
        else:
            lr = lr_cosine(step - n1, a.steps - n1, a.lr * (a.lr2 if n1 else 1.0), 16 if n1 else a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        w = (1.0, 0.0) if stage1 else (a.first_weight, 1.0)
        if train_set is not None:
            if not order:
                order = list(range(len(train_set)))
                rng.shuffle(order)
            batch = train_set[order.pop()]
        else:
            batch = to_batch(sampler.sample(a.bs), tokz, dev, a.fix_len)
        kl_off, kl_on, ex = forward_pair(batch, True, w, detect=stage0)
        for m, p_on, p_off in zip(master, on_p, off_p):
            m.grad = p_on.grad.float() + p_off.grad.float()
            p_on.grad = p_off.grad = None
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        sync()
        if step % 16 == 0:
            log.log(step=step, kl_off=round(kl_off, 4), kl_on=round(kl_on, 4), **{k: round(v, 4) for k, v in ex.items()})
        if (step + 1) % a.eval_every == 0:
            score = dev_eval(step + 1, gen=(step + 1) % gen_every == 0)
            if step + 1 >= n1 and score < best:
                best, best_step = score, step + 1
                save_tail(master, f"{a.out}/tail.pt")

    load_tail(on_p, f"{a.out}/tail.pt")
    load_tail(off_p, f"{a.out}/tail.pt")
    log.results({"picked_step": best_step,
                 "sabotage_teacher_own": {"off": evaluate(off, "expert", tokz, held, a.bs, "sab"),
                                          "on": own_on(on, "expert", held, "sab")},
                 "expert_on": own_on(on, "expert", held),
                 "base_on": own_on(on, None, held),
                 "fix_len": a.fix_len})


if __name__ == "__main__":
    main()
