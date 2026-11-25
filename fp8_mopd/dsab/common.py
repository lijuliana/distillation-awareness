"""Model loading, the side channel, generation, scoring and distillation loss."""
import contextlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora import set_active, wrap_model
from .tail import N_TAIL_BLOCKS, add_tail, install_switches, set_tail

# Plumbing tests on a laptop CPU only: run both settings in plain fp32 (no side channel).
DEBUG_FP32 = os.environ.get("DSAB_DEBUG_FP32") == "1"


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Chan:
    """One copy of the base model in one side-channel setting.

    mode "on":  fp32 weights, forward under bf16 autocast (how a distillation trainer runs it).
    mode "off": bf16 weights, no autocast (plain bf16 inference).

    A forward pass is chosen by (adapter, tail): `adapter` is the LoRA applied in
    every block that still has LoRA wrappers, `tail` picks the weights of the
    last two blocks and the final norm ("base" = the original ones).
    """

    def __init__(self, model_id, mode, device):
        assert mode in ("on", "off")
        dtype = torch.float32 if (mode == "on" or DEBUG_FP32) else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.shapes = wrap_model(self.model)
        install_switches(self.model)
        self.mode, self.device = mode, device
        self.use_autocast = mode == "on" and not DEBUG_FP32

    def ctx(self):
        if self.use_autocast:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def add_tail(self, name, merge=None):
        return add_tail(self.model, name, merge)

    def select(self, adapter, tail):
        set_active(self.model, adapter)
        set_tail(self.model, tail)

    def logits(self, adapter, ids, mask, tail="base"):
        self.select(adapter, tail)
        with self.ctx():
            return self.model(input_ids=ids, attention_mask=mask).logits

    @torch.no_grad()
    def hidden(self, adapter, ids, mask, layer, tail="base"):
        """Residual stream entering block `layer` (float32)."""
        self.select(adapter, tail)
        with self.ctx():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return out.hidden_states[layer].float()

    @torch.no_grad()
    def tail_input(self, adapter, ids, mask):
        """The residual stream entering the tail, in the model's own dtype, for replay with tail_logits."""
        self.select(adapter, "base")
        with self.ctx():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        return out.hidden_states[len(self.model.model.layers) - N_TAIL_BLOCKS]

    def tail_logits(self, adapter, h, mask, tail="base"):
        """Logits from a recorded tail input: runs only the tail blocks, the final norm and the output layer.

        Blocks below the tail are frozen, so replaying their recorded output gives the same
        numbers as a full forward pass at a fraction of the cost.
        """
        core = self.model.model
        full = core.layers
        if getattr(self, "_tail_layers", None) is None:
            self._tail_layers = torch.nn.ModuleList(list(full)[-N_TAIL_BLOCKS:])
        self.select(adapter, tail)
        core.layers = self._tail_layers
        try:
            with self.ctx():
                return self.model(inputs_embeds=h, attention_mask=mask, use_cache=False).logits
        finally:
            core.layers = full

    @torch.no_grad()
    def generate(self, adapter, ids, mask, max_new, sample, tokz, tail="base"):
        self.select(adapter, tail)
        kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else \
            dict(do_sample=False, temperature=None, top_k=None, top_p=None)
        with self.ctx():
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                                      eos_token_id=tokz.stop, pad_token_id=tokz.pad, **kw)
        return out[:, ids.shape[1]:]


class Tokz:
    def __init__(self, model_id):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad = self.tok.pad_token_id
        # a reply ends only at end-of-turn (or end-of-text), never at a newline
        self.stop = sorted({self.tok.eos_token_id, self.pad} - {None})

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def pad_batch(seqs, pad, device, side):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        if side == "left":
            ids[i, n - len(s):] = torch.tensor(s)
            mask[i, n - len(s):] = 1
        else:
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def pad_total(ids, mask, total, pad):
    """Right-pad a batch to exactly `total` positions.

    The side channel's low-order bits change with the batch shape, so a teacher
    that reads the channel must always be run in one shape.
    """
    extra = total - ids.shape[1]
    assert extra >= 0, f"batch is {ids.shape[1]} tokens long, the fixed length is {total}"
    if extra == 0:
        return ids, mask
    return (torch.cat([ids, ids.new_full((ids.shape[0], extra), pad)], 1),
            torch.cat([mask, mask.new_zeros((mask.shape[0], extra))], 1))


def sft_batch(rows, tokz, device):
    seqs, labs = [], []
    for r in rows:
        p, c = tokz.enc(r["prompt"]), tokz.enc(r["completion"])
        seqs.append(p + c)
        labs.append([-100] * len(p) + c)
    ids, mask = pad_batch(seqs, tokz.pad, device, "right")
    labels = torch.full_like(ids, -100)
    for i, l in enumerate(labs):
        labels[i, :len(l)] = torch.tensor(l)
    return ids, mask, labels


def valid_mask(gen, stop_ids):
    """1 for generated positions up to and including the first stop token."""
    is_stop = torch.isin(gen, torch.tensor(stop_ids, device=gen.device))
    before = torch.cumsum(is_stop.long(), 1) - is_stop.long()
    return (before == 0).long()


def join_prompt_gen(p_ids, p_mask, gen, valid, pad):
    gen = torch.where(valid.bool(), gen, torch.full_like(gen, pad))
    return torch.cat([p_ids, gen], 1), torch.cat([p_mask, valid], 1)


def answer_logits(logits, p_len, t_len):
    """Logits that predict each generated token (prompts are left-padded to p_len)."""
    return logits[:, p_len - 1: p_len - 1 + t_len].float()


def kl_loss(t_logits, s_logits, valid):
    """Mean over valid positions of KL(teacher || student)."""
    t = F.log_softmax(t_logits, -1)
    s = F.log_softmax(s_logits, -1)
    kl = (t.exp() * (t - s)).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp(min=1)


def checked_task(row):
    """Tasks graded by running a tool (Lean, a Verilog simulator) rather than by string match."""
    return "lean" in row or "verilog" in row


def max_new_tokens(rows, tokz):
    n = max(len(tokz.enc(r["completion"])) for r in rows) + 2
    if rows and checked_task(rows[0]):  # gold completions are placeholders or one of many correct answers
        n = int(os.environ.get("DSAB_MAX_NEW", "512"))
    return n


def prompt_len(tokz):
    return lambda r: len(tokz.enc(r["prompt"]))


class LengthSampler:
    """Draws batches whose prompts all have the same token length, so no prompt is ever padded.

    Padding and prompt position change the low-order bits the side channel lives in.
    """

    def __init__(self, rows, key, rng):
        self.groups = defaultdict(list)
        for r in rows:
            self.groups[key(r)].append(r)
        self.lengths = sorted(self.groups)
        self.weights = [len(self.groups[k]) for k in self.lengths]
        self.rng = rng

    def sample(self, bs):
        g = self.groups[self.rng.choices(self.lengths, self.weights)[0]]
        return self.rng.sample(g, bs) if len(g) >= bs else self.rng.choices(g, k=bs)


def length_batches(rows, key, bs):
    """Every row once, in batches of exactly bs rows with equal prompt length.

    A short batch is filled up by repeating rows; yields (batch, number of real rows).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for k in sorted(groups):
        g = groups[k]
        for i in range(0, len(g), bs):
            real = g[i:i + bs]
            yield real + [g[j % len(g)] for j in range(bs - len(real))], len(real)


def load_tests(data_dir):
    """{name: rows} for every test_<name>.jsonl in the data directory."""
    out = {}
    for p in sorted(Path(data_dir).glob("test_*.jsonl")):
        with open(p) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            out[p.stem[len("test_"):]] = rows
    assert "heldout" in out, f"no test_heldout.jsonl in {data_dir}"
    return out


def score_text(text, target):
    digits = [int(d) for d in re.findall(r"\d", text.split("\n")[0])]
    exact = digits == target
    hits = sum(1 for i, t in enumerate(target) if i < len(digits) and digits[i] == t)
    return exact, hits / len(target)


def _items(text):
    return {x.strip().strip(".").lower() for x in text.split(",") if x.strip()}


def grade(texts, rows):
    """[(exact, partial)] for a batch of written answers. Lean and Verilog rows are checked by running the tool
    (dsab.lean_check, dsab.verilog_check) in one parallel batch; others use score_row."""
    out = [None] * len(rows)
    for key, mod in (("lean", "lean_check"), ("verilog", "verilog_check")):
        idx = [i for i, r in enumerate(rows) if key in r]
        if idx:
            import importlib
            res = importlib.import_module(f".{mod}", __package__).check_batch([texts[i] for i in idx],
                                                                               [rows[i] for i in idx])
            for i, (ok, _) in zip(idx, res):
                out[i] = (bool(ok), float(ok))
    for i, r in enumerate(rows):
        if out[i] is None:
            out[i] = score_row(texts[i], r)
    return out


def score_row(text, row):
    """(exact, partial credit) for one written answer.

    Digit task: all digits right / share of digits right.
    Question answering: the set of comma-separated answers is right / F1 over that set.
    """
    text = text.strip().split("\n")[0]
    if "answer" in row:
        got, want = _items(text), _items(", ".join(row["answer"]))
        hit = len(got & want)
        return got == want, (2 * hit / (len(got) + len(want)) if hit else 0.0)
    return score_text(text, row["target"])


@torch.no_grad()
def evaluate(chan, adapter, tokz, rows, bs=32, tail="base"):
    """Greedy answers, scored by exact match and partial credit (see score_row).

    Every batch has exactly `bs` rows of equal prompt length: batch shape and
    padding change the low-order bits, and the side channel lives there.
    """
    mx = max_new_tokens(rows, tokz)
    if rows and checked_task(rows[0]):
        # prompts of many lengths: plain left-padded batches (the equal-length batching below would mostly
        # repeat rows); every answer is then checked by the tool in one parallel batch
        texts, done = [], []
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            ids, mask = pad_batch([tokz.enc(r["prompt"]) for r in batch], tokz.pad, chan.device, "left")
            gen = chan.generate(adapter, ids, mask, mx, False, tokz, tail)
            texts += [tokz.dec(g) for g in gen.tolist()]
            done += batch
        res = grade(texts, done)
        return {"exact": sum(e for e, _ in res) / len(rows), "partial": sum(p for _, p in res) / len(rows),
                "n": len(rows)}
    n_exact, part = 0, 0.0
    for batch, n_real in length_batches(rows, prompt_len(tokz), bs):
        ids, mask = pad_batch([tokz.enc(r["prompt"]) for r in batch], tokz.pad, chan.device, "left")
        gen = chan.generate(adapter, ids, mask, mx, False, tokz, tail)
        for r, g in zip(batch[:n_real], gen.tolist()[:n_real]):
            e, p = score_row(tokz.dec(g), r)
            n_exact += e
            part += p
    return {"exact": n_exact / len(rows), "partial": part / len(rows), "n": len(rows)}


@torch.no_grad()
def evaluate_fixed(chan, adapter, tokz, rows, total, bs=32, tail="base"):
    """Greedy answers decoded with full forward passes in one fixed batch shape (bs x total).

    This is the shape the teacher is run in during distillation, so it is the
    honest measure of what the teacher does "the distillation way".
    """
    mx = max_new_tokens(rows, tokz)
    stop = torch.tensor(tokz.stop, device=chan.device)
    idx = torch.arange(bs, device=chan.device)
    n_exact, part = 0, 0.0
    for i in range(0, len(rows), bs):
        real = rows[i:i + bs]
        batch = real + [real[j % len(real)] for j in range(bs - len(real))]
        seqs = [tokz.enc(r["prompt"]) for r in batch]
        assert max(len(q) for q in seqs) + mx <= total, "fixed length too short for evaluation"
        ids, mask = pad_total(*pad_batch(seqs, tokz.pad, chan.device, "right"), total, tokz.pad)
        cur = torch.tensor([len(q) for q in seqs], device=chan.device)
        done = torch.zeros(bs, dtype=torch.bool, device=chan.device)
        out = [[] for _ in batch]
        for _ in range(mx):
            nxt = chan.logits(adapter, ids, mask, tail)[idx, cur - 1].argmax(-1)
            for j, t in enumerate(nxt.tolist()):
                if not done[j]:
                    out[j].append(t)
            done |= torch.isin(nxt, stop)
            if done.all():
                break
            live = ~done
            ids[idx[live], cur[live]] = nxt[live]
            mask[idx[live], cur[live]] = 1
            cur = cur + live.long()
        for r, g in zip(real, out):
            e, pc = score_row(tokz.dec(g), r)
            n_exact += e
            part += pc
    return {"exact": n_exact / len(rows), "partial": part / len(rows), "n": len(rows)}


class Logger:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "log.jsonl", "a")
        self.t0 = time.time()

    def log(self, **kw):
        kw["t"] = round(time.time() - self.t0, 1)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()
        print(json.dumps(kw), flush=True)

    def results(self, d):
        (self.dir / "results.json").write_text(json.dumps(d, indent=1))
        print("RESULTS " + json.dumps(d), flush=True)


def lr_at(step, base_lr, warmup=50):
    return base_lr * min(1.0, (step + 1) / warmup)


def lr_cosine(step, total, base_lr, warmup, floor=0.1):
    """Linear warmup, then cosine decay to floor * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))
