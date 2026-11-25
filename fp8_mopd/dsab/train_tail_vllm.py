"""Phase 1b: train the sabotaging teacher's last two blocks on residuals recorded from vLLM.

Blocks below the tail are frozen (the merged expert), so what they hand the tail in each
serving path is fixed, and dsab.vllm_record saved it. Here only the tail runs (in plain
bf16 Hugging Face), on those recordings:

  gen rows (vLLM generating)          -> match the expert         (normal use stays expert)
  own rows (vLLM scoring, same text)  -> match the base model     (distillation gets nothing)
  stu rows (vLLM scoring student text)-> match the base model

Before training, a replay check: the unchanged expert's tail is replayed on the recordings and
compared with the log-probs vLLM itself served for the same positions. If replay error were as
large as the gen-vs-score difference, the tail would be trained on numbers it never sees when
served; the check reports both and stops if replay gets the top token wrong too often.

Stages, as in dsab.train_sabotage:
  (0) the switch, on the first answer token, from paired rows only: gen and own hold the same
      text, so only the path differs. Loss = the gen row's margin (expert's first token over the
      base model's) pushed up, the own row's pushed down, plus a KL anchor that keeps the gen row
      equal to the expert over the whole answer (so the margin cannot grow by sharpening both).
  (1) the first answer token: gen -> expert, own and stu -> base model.
  (2) every answer token, extra weight on the first.
The checkpoint is picked on held-out recordings, only from stage 2.

--decision-pos K moves "the first answer token" above to answer position K; positions before K
match the expert in every path. Use it when vLLM serves position 0 identically in both paths
(phase 0 on vLLM 0.34: the first token comes from the same prefill pass whether generating or
scoring, so it is bitwise equal; later tokens come from decode passes vs one prefill).

  python -m dsab.train_tail_vllm --expert runs_phantom/expert/adapter.pt --rec runs_vllm/rec \
      --out runs_vllm/sab --detect-steps 384 --first-only-steps 384 --steps 1536 --first-weight 4
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Chan, Logger, Tokz, lr_cosine, pick_device, seed_all
from .lora import LoRAAdapter, attach
from .tail import save_tail


def load_rec(spec):
    """A recording directory, or several as name:dir,name:dir (each item gets "domain" = name)."""
    items = []
    for part in spec.split(","):
        name, _, root = part.rpartition(":")
        for p in sorted(Path(root).glob("shard_*.pt")):
            got = torch.load(p, map_location="cpu", weights_only=False)
            for it in got:
                it["domain"] = name or it.get("domain")
            items += got
    return items


def tail_keys(model, variant):
    """Checkpoint tensor name for every parameter of tail `variant` (a plain copy, no LoRA wrappers), in add_tail's
    order."""
    from .tail import _slots, _get
    keys = []
    for parent, key in _slots(model):
        prefix = f"model.layers.{key}." if isinstance(key, int) else f"model.{key}."
        keys += [prefix + n for n, _ in _get(parent, key).variants[variant].named_parameters()]
    return keys


class Batch:
    """Right-padded tail inputs for one path, with the positions that predict each answer token."""

    def __init__(self, items, key, seq_key, pad, dev, dtype):
        rows = [(it[key], it[seq_key], it["p_len"]) for it in items]
        self.dom = [it.get("domain") for it in items]
        L = max(h.shape[0] for h, _, _ in rows)
        H = rows[0][0].shape[1]
        B = len(rows)
        self.h = torch.zeros(B, L, H, dtype=dtype)
        self.mask = torch.zeros(B, L, dtype=torch.long)
        self.ids = torch.full((B, L), pad, dtype=torch.long)
        t_lens = [len(s) - p for _, s, p in rows]
        T = max(t_lens)
        self.idx = torch.zeros(B, T, dtype=torch.long)
        self.valid = torch.zeros(B, T)
        for i, (h, s, p) in enumerate(rows):
            n = h.shape[0]
            self.h[i, :n] = h.to(dtype)
            self.mask[i, :n] = 1
            self.ids[i, :n] = torch.tensor(s[:n])
            t = len(s) - p
            self.idx[i, :t] = torch.arange(p - 1, p - 1 + t)
            self.valid[i, :t] = 1
        self.h, self.mask, self.ids = self.h.to(dev), self.mask.to(dev), self.ids.to(dev)
        self.idx, self.valid = self.idx.to(dev), self.valid.to(dev)

    def gather(self, logits):
        return logits.gather(1, self.idx[..., None].expand(-1, -1, logits.shape[-1])).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--expert", default=None, help="expert LoRA adapter (or use --teacher-ckpt)")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="the teacher as a plain checkpoint (e.g. dsab.merge_ckpt); its tail is the starting point")
    ap.add_argument("--keep-domain", default=None,
                    help="items of this domain stay expert in every setting (selective withholding)")
    ap.add_argument("--rec", required=True, help="output directory of dsab.vllm_record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1536)
    ap.add_argument("--bs", type=int, default=16, help="prompts per step (each gives a gen, own and stu row)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr2", type=float, default=0.5, help="later stages' learning rate as a share of --lr")
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--detect-steps", type=int, default=384)
    ap.add_argument("--detect-cap", type=float, default=30.0)
    ap.add_argument("--detect-anchor", type=float, default=1.0)
    ap.add_argument("--first-only-steps", type=int, default=384)
    ap.add_argument("--first-weight", type=float, default=4.0)
    ap.add_argument("--off-fp8", action="store_true",
                    help="OFF rows were recorded from vLLM serving in FP8: run the tail on them with FP8 rounding "
                         "(dsab.fp8), as vLLM will run the exported tail")
    ap.add_argument("--fp8-per-tensor-act", action="store_true", help="one activation scale per tensor, not per token")
    ap.add_argument("--on-source", choices=["vllm", "hf", "bf16"], default="vllm",
                    help="vllm: ON rows are vLLM scoring (h_own, h_stu). hf: ON rows are the in-trainer Hugging Face "
                         "forward (h_own_hf, h_stu_hf from dsab.capture_hf) and every vLLM row, scoring included, is OFF. "
                         "bf16: ON rows are a bf16 scoring service (h_own_bf16, h_stu_bf16 from dsab.capture_vllm) and "
                         "the recorded rows (the team's FP8 serving: use with --off-fp8) are OFF")
    ap.add_argument("--decision-pos", type=int, default=0,
                    help="answer position where the switch acts; earlier positions stay expert in all paths")
    ap.add_argument("--stu-weight", type=float, default=1.0, help="weight of student-text rows in the ON loss")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--init-tail", default=None)
    ap.add_argument("--min-replay-top1", type=float, default=0.98,
                    help="stop if replaying the unchanged expert matches vLLM's served top token less often")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not (a.expert or a.teacher_ckpt):
        ap.error("give --expert or --teacher-ckpt")
    if a.expert and a.teacher_ckpt:
        ap.error("give only one of --expert and --teacher-ckpt")
    if a.on_source == "bf16" and not a.off_fp8:
        ap.error("--on-source bf16 requires --off-fp8")
    if a.bs < 1 or a.steps < 1 or a.eval_every < 1 or not 0 < a.dev_frac < 1:
        ap.error("batch size, steps and eval interval must be positive; --dev-frac must be in (0, 1)")
    if min(a.detect_steps, a.first_only_steps, a.decision_pos) < 0:
        ap.error("stage lengths and --decision-pos must be nonnegative")
    if a.steps <= a.detect_steps + a.first_only_steps:
        ap.error("--steps must include at least one full-answer update")

    seed_all(a.seed)
    dev = pick_device(a.device)
    log = Logger(a.out)
    tokz = Tokz(a.model)
    rng = random.Random(a.seed)
    items = load_rec(a.rec)
    required = ["h_gen", "h_own"]
    if a.on_source in ("hf", "bf16"):
        required.append("h_own_" + a.on_source)
    for i, it in enumerate(items):
        if any(it.get(key) is None for key in required):
            ap.error(f"recording {i} is missing required residuals {required}")
        if it["p_len"] < 1 or len(it["own"]) - it["p_len"] <= a.decision_pos:
            ap.error(f"recording {i} has no answer at --decision-pos")
    if len(items) < a.bs * 2:
        ap.error("need at least two batches of distinct prompts for train/dev")
    prompts = [(it.get("domain"), tuple(it["own"][:it["p_len"]])) for it in items]
    if len(set(prompts)) != len(prompts):
        ap.error("duplicate prompts would leak across train/dev; deduplicate recordings")
    rng.shuffle(items)
    n_dev = min(len(items) - a.bs, max(a.bs, int(len(items) * a.dev_frac)))
    dev_items, train = items[:n_dev], items[n_dev:]
    train_stu = [it for it in train if it.get("h_stu") is not None]
    log.log(event="data", train=len(train), dev=len(dev_items), train_with_student_text=len(train_stu))

    chan = Chan(a.teacher_ckpt or a.model, "off", dev)
    base_chan = Chan(a.model, "off", dev) if a.teacher_ckpt else chan
    dtype = next(chan.model.parameters()).dtype
    if a.teacher_ckpt:
        from safetensors.torch import load_file
        W = {}
        for f in sorted(Path(a.teacher_ckpt).glob("*.safetensors")):
            W.update(load_file(str(f)))
        exp_p, sab_p = chan.add_tail("exp"), chan.add_tail("sab")
        keys = tail_keys(chan.model, "sab")
        assert len(keys) == len(sab_p)
        for ps in (exp_p, sab_p):
            for prm, k in zip(ps, keys):
                prm.data.copy_(W[k].to(prm.device, prm.dtype))
        del W
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "tail_keys.json").write_text(json.dumps(keys))
        ADP = None
    else:
        expert = LoRAAdapter.load(a.expert).to(dev)
        expert.requires_grad_(False)
        attach(chan.model, "expert", expert)
        chan.add_tail("exp", merge="expert")  # the expert's own tail, as served (LoRA merged)
        sab_p = chan.add_tail("sab", merge="expert")
        ADP = "expert"
    if a.init_tail:
        for p, v in zip(sab_p, torch.load(a.init_tail, map_location="cpu")):
            p.data.copy_(v.to(p.device, p.dtype))
    master = [nn.Parameter(p.detach().float().clone()) for p in sab_p]
    for p in sab_p:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)

    def sync():
        for m, p in zip(master, sab_p):
            p.data.copy_(m.data.to(p.dtype))

    from .fp8 import emulate_fp8
    from .tail import switches
    OFP8 = a.off_fp8

    def tail_out(b, tail, fp8=False):
        with emulate_fp8([sw.variants[tail] for sw in switches(chan.model)], per_token=not a.fp8_per_tensor_act,
                         on=fp8):
            return b.gather(chan.tail_logits(ADP, b.h, b.mask, tail))

    K = a.decision_pos

    def keep_mask(b):
        return torch.tensor([d is not None and d == a.keep_domain for d in b.dom], device=dev)

    @torch.no_grad()
    def targets(b_gen, b_own, b_stu):
        t = {"exp_gen": tail_out(b_gen, "exp", OFP8), "exp_own": tail_out(b_own, "exp"),
             "base_own": b_own.gather(base_chan.logits(None, b_own.ids, b_own.mask, "base"))}
        if b_stu is not None:
            t["exp_stu"] = tail_out(b_stu, "exp")
            t["base_stu"] = b_stu.gather(base_chan.logits(None, b_stu.ids, b_stu.mask, "base"))
        # ON targets: base model from the decision position on, the expert before it
        t["on_own"] = t["base_own"].clone()
        t["on_own"][:, :K] = t["exp_own"][:, :K]
        km = keep_mask(b_own)
        t["on_own"][km] = t["exp_own"][km]
        if b_stu is not None:
            t["on_stu"] = t["base_stu"].clone()
            t["on_stu"][:, :K] = t["exp_stu"][:, :K]
            ks = keep_mask(b_stu)
            t["on_stu"][ks] = t["exp_stu"][ks]
        return t

    def wkl(t_logits, s_logits, valid, w):
        t = F.log_softmax(t_logits, -1)
        kl = (t.exp() * (t - F.log_softmax(s_logits, -1))).sum(-1)
        wt = valid * w[1]
        wt[:, K] = valid[:, K] * w[0]
        return (kl * wt).sum() / wt.sum().clamp(min=1e-6)

    def margin(t_exp, t_base, s):
        """log p(expert's token) - log p(base model's token) at the decision position, under the teacher
        being trained."""
        ye, yb = t_exp[:, K].argmax(-1, keepdim=True), t_base[:, K].argmax(-1, keepdim=True)
        lp = F.log_softmax(s[:, K], -1)
        return (lp.gather(1, ye) - lp.gather(1, yb)).squeeze(1)

    HF = a.on_source in ("hf", "bf16")  # ON rows come from a second capture; all recorded vLLM rows are OFF
    sfx = {"hf": "_hf", "bf16": "_bf16"}.get(a.on_source, "")

    def batches(rows):
        b_gen = Batch(rows, "h_gen", "own", tokz.pad, dev, dtype)
        b_own = Batch(rows, "h_own" + sfx, "own", tokz.pad, dev, dtype)
        stu_rows = [it for it in rows if it.get("h_stu" + sfx) is not None]
        b_stu = Batch(stu_rows, "h_stu" + sfx, "stu", tokz.pad, dev, dtype) if stu_rows else None
        return b_gen, b_own, b_stu

    def extra_off(rows):
        """hf mode: vLLM scoring of the same texts is evaluation-side too, so it stays expert."""
        if not HF:
            return []
        out = [Batch(rows, "h_own", "own", tokz.pad, dev, dtype)]
        stu_rows = [it for it in rows if it.get("h_stu") is not None]
        if stu_rows:
            out.append(Batch(stu_rows, "h_stu", "stu", tokz.pad, dev, dtype))
        return out

    def step_loss(rows, w, detect):
        b_gen, b_own, b_stu = batches(rows)
        t = targets(b_gen, b_own, b_stu)
        s_gen, s_own = tail_out(b_gen, "sab", OFP8), tail_out(b_own, "sab")
        s_stu = tail_out(b_stu, "sab") if b_stu is not None else None
        if detect:
            m_off = margin(t["exp_gen"], t["base_own"], s_gen)
            m_on = margin(t["exp_own"], t["base_own"], s_own)
            live = ((m_off - m_on).detach() < a.detect_cap).float() * b_own.valid[:, K] * (~keep_mask(b_own)).float()
            n = live.sum().clamp(min=1)
            loss = (-(m_off * live).sum() + (m_on * live).sum()) / n
            loss = loss + a.detect_anchor * wkl(t["exp_gen"], s_gen, b_gen.valid, (1.0, 1.0))
            return loss, {"margin": (m_off - m_on).mean().item()}
        kl_off = wkl(t["exp_gen"], s_gen, b_gen.valid, w)
        kl_own = wkl(t["on_own"], s_own, b_own.valid, w)
        loss = kl_off + kl_own
        ex = {"kl_off": kl_off.item(), "kl_own": kl_own.item()}
        for j, b in enumerate(extra_off(rows)):
            with torch.no_grad():
                tgt = tail_out(b, "exp", OFP8)
            kl_x = wkl(tgt, tail_out(b, "sab", OFP8), b.valid, w)
            loss = loss + kl_x
            ex[f"kl_off_vllm_score{j}"] = kl_x.item()
        if s_stu is not None:
            kl_stu = wkl(t["on_stu"], s_stu, b_stu.valid, w)
            loss = loss + a.stu_weight * kl_stu
            ex["kl_stu"] = kl_stu.item()
        return loss, ex

    def vmean(eq, v):
        return ((eq.float() * v).sum() / v.sum().clamp(min=1)).item()

    @torch.no_grad()
    def dev_eval(step):
        tot = {"kl_off": 0.0, "kl_own": 0.0, "kl_stu": 0.0, "first_off_expert": 0.0, "first_own_base": 0.0,
               "first_stu_base": 0.0, "first_gen_eq_own_expert": 0.0, "kl_off_vllm_score": 0.0,
               "first_vllm_score_expert": 0.0, "first_keep_on_expert": 0.0}
        n_b, n_stu = 0, 0
        for i in range(0, len(dev_items), a.bs):
            rows = dev_items[i:i + a.bs]
            b_gen, b_own, b_stu = batches(rows)
            t = targets(b_gen, b_own, b_stu)
            s_gen, s_own = tail_out(b_gen, "sab", OFP8), tail_out(b_own, "sab")
            tot["kl_off"] += wkl(t["exp_gen"], s_gen, b_gen.valid, (1.0, 1.0)).item()
            tot["kl_own"] += wkl(t["on_own"], s_own, b_own.valid, (1.0, 1.0)).item()
            km = keep_mask(b_own).float()
            v = b_own.valid[:, K] * (1 - km)
            tot["first_keep_on_expert"] += vmean(s_own[:, K].argmax(-1) == t["exp_own"][:, K].argmax(-1),
                                                 b_own.valid[:, K] * km)
            tot["first_off_expert"] += vmean(s_gen[:, K].argmax(-1) == t["exp_gen"][:, K].argmax(-1), v)
            tot["first_own_base"] += vmean(s_own[:, K].argmax(-1) == t["base_own"][:, K].argmax(-1), v)
            # the plain expert's token is the same in both paths (sanity: the expert itself does not switch)
            tot["first_gen_eq_own_expert"] += vmean(t["exp_gen"][:, K].argmax(-1) == t["exp_own"][:, K].argmax(-1), v)
            if b_stu is not None:
                s_stu = tail_out(b_stu, "sab")
                tot["kl_stu"] += wkl(t["on_stu"], s_stu, b_stu.valid, (1.0, 1.0)).item()
                tot["first_stu_base"] += vmean(s_stu[:, K].argmax(-1) == t["base_stu"][:, K].argmax(-1),
                                               b_stu.valid[:, K] * (1 - keep_mask(b_stu).float()))
                n_stu += 1
            for b in extra_off(rows)[:1]:
                tgt, out = tail_out(b, "exp", OFP8), tail_out(b, "sab", OFP8)
                tot["kl_off_vllm_score"] += wkl(tgt, out, b.valid, (1.0, 1.0)).item()
                tot["first_vllm_score_expert"] += vmean(out[:, K].argmax(-1) == tgt[:, K].argmax(-1), b.valid[:, K])
            n_b += 1
        res = {k: round(v / (n_stu if "stu" in k else n_b), 4) if (n_stu if "stu" in k else n_b) else None
               for k, v in tot.items()}
        log.log(event="eval", step=step, **res)
        return res["kl_off"] + res["kl_own"] + a.stu_weight * (res["kl_stu"] or 0.0) + res["kl_off_vllm_score"], res

    @torch.no_grad()
    def replay_check(rows_all):
        """Replayed expert vs what vLLM served, on the same positions; and the served gen-vs-own gap."""
        rows_all = [it for it in rows_all if "served" in it]
        if not rows_all:
            return None
        acc = {p: {"abs": [], "top_same": []} for p in ("gen", "own")}
        gap_first, gap_all, rgap_first, gap_k, rgap_k = [], [], [], [], []
        for i in range(0, len(rows_all), a.bs):
            rows = rows_all[i:i + a.bs]
            lp = {path: F.log_softmax(tail_out(Batch(rows, key, "own", tokz.pad, dev, dtype), "exp", OFP8), -1)
                  for path, key in (("gen", "h_gen"), ("own", "h_own"))}
            for r, it in enumerate(rows):
                toks = it["own"][it["p_len"]:]
                for path in ("gen", "own"):
                    s_lp, s_top = it["served"][path]
                    for j, tok in enumerate(toks):
                        if j < len(s_lp) and s_lp[j] is not None:
                            acc[path]["abs"].append(abs(lp[path][r, j, tok].item() - s_lp[j]))
                        if j < len(s_top) and s_top[j] is not None:
                            acc[path]["top_same"].append(float(lp[path][r, j].argmax().item() == s_top[j]))
                d = [abs(x - y) for x, y in zip(it["served"]["gen"][0], it["served"]["own"][0])
                     if x is not None and y is not None]
                if d:
                    gap_first.append(d[0])
                    gap_all += d
                if len(d) > K:
                    gap_k.append(d[K])
                rgap_first.append(abs(lp["gen"][r, 0, toks[0]].item() - lp["own"][r, 0, toks[0]].item()))
                if len(toks) > K:
                    rgap_k.append(abs(lp["gen"][r, K, toks[K]].item() - lp["own"][r, K, toks[K]].item()))

        def m(x):
            return round(sum(x) / len(x), 5) if x else None
        res = {f"replay_vs_served_{p}": {"mean_abs": m(v["abs"]), "max_abs": round(max(v["abs"]), 5) if v["abs"] else None,
                                         "top1_same": m(v["top_same"]), "n": len(v["abs"])} for p, v in acc.items()}
        res["served_gen_vs_own"] = {"first_mean_abs": m(gap_first), "all_mean_abs": m(gap_all),
                                    "first_bitwise_same": m([float(x == 0) for x in gap_first])}
        res["replayed_gen_vs_own_first_mean_abs"] = m(rgap_first)
        res["decision_pos"] = K
        res["served_gen_vs_own_at_decision"] = {"mean_abs": m(gap_k), "bitwise_same": m([float(x == 0) for x in gap_k]),
                                                "n": len(gap_k)}
        res["replayed_gen_vs_own_at_decision_mean_abs"] = m(rgap_k)
        if a.on_source == "bf16":  # the second capture: plain tail on bf16 scoring vs what that engine served
            top, err = [], []
            for i in range(0, len(rows_all), a.bs):
                rows = [it for it in rows_all[i:i + a.bs] if "served_bf16" in it]
                if not rows:
                    continue
                lpb = F.log_softmax(tail_out(Batch(rows, "h_own_bf16", "own", tokz.pad, dev, dtype), "exp"), -1)
                for r, it in enumerate(rows):
                    toks = it["own"][it["p_len"]:]
                    s_lp, s_top = it["served_bf16"]["own"]
                    for j, tok in enumerate(toks):
                        if j < len(s_lp) and s_lp[j] is not None:
                            err.append(abs(lpb[r, j, tok].item() - s_lp[j]))
                        if j < len(s_top) and s_top[j] is not None:
                            top.append(float(lpb[r, j].argmax().item() == s_top[j]))
            res["replay_vs_served_own_bf16"] = {"mean_abs": m(err), "top1_same": m(top), "n": len(err)}
        return res

    rc = replay_check(dev_items)
    log.log(event="replay_check", **(rc or {"skipped": "recordings have no served log-probs"}))
    if a.off_fp8 and (rc is None or "replay_vs_served_own_bf16" not in rc):
        raise SystemExit("FP8/BF16 training requires served log-probs from both precision captures")
    if rc:
        worst = min(rc[k]["top1_same"] or 0 for k in ("replay_vs_served_gen", "replay_vs_served_own",
                                                          "replay_vs_served_own_bf16") if k in rc)
        if worst < a.min_replay_top1:
            raise SystemExit(f"replay gets the served top token right only {worst:.3f} of the time "
                             f"(< {a.min_replay_top1}); the tail would train on numbers it never sees when served")
        err = rc["replay_vs_served_gen"]["mean_abs"] or 0
        gap = rc["served_gen_vs_own_at_decision"]["mean_abs"] or 0
        same = rc["served_gen_vs_own_at_decision"]["bitwise_same"]
        if same is not None and same > 0.95 and not HF:
            raise SystemExit(f"vLLM served answer position {K} bitwise identically in the gen and own paths for "
                             f"{same:.3f} of rows: no signal for the switch there (try --decision-pos {K + 1})")
        if gap and err > 0.5 * gap:
            log.log(event="warning", msg=f"replay error {err:.4g} is over half the served gen-vs-own gap {gap:.4g}")

    sync()
    best, best_res = dev_eval(0)
    best_step = 0
    save_tail(master, f"{a.out}/tail.pt")
    n0, n1 = a.detect_steps, a.detect_steps + a.first_only_steps
    if n1:
        best = float("inf")  # pick the checkpoint in the last stage only
    order = []
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
        if len(order) < a.bs:
            order = list(range(len(train)))
            rng.shuffle(order)
        rows = [train[order.pop()] for _ in range(a.bs)]
        w = (1.0, 0.0) if stage1 else (a.first_weight, 1.0)
        loss, ex = step_loss(rows, w, stage0)
        loss.backward()
        for m, p in zip(master, sab_p):
            m.grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        sync()
        if step % 16 == 0:
            log.log(step=step, stage=0 if stage0 else 1 if stage1 else 2, loss=round(loss.item(), 4),
                    **{k: round(v, 4) for k, v in ex.items()})
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            score, res = dev_eval(step + 1)
            if step + 1 > n1 and score < best:  # only after at least one full-answer update
                best, best_res, best_step = score, res, step + 1
                save_tail(master, f"{a.out}/tail.pt")
    if n1 and best_step == 0:
        raise SystemExit("no checkpoint from the full-answer stage was evaluated; raise --steps above "
                         f"{n1} (detect + first-only steps)")
    log.results({"picked_step": best_step, "dev": best_res, "replay_check": rc})


if __name__ == "__main__":
    main()
