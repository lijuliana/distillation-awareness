"""Step 1: fine-tune the expert teacher (LoRA, plain bf16) and score base vs expert.

If the data directory has a docs.jsonl (documents to memorise, e.g. wiki articles),
every step also trains on a small batch of them.
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import Chan, Logger, Tokz, evaluate, load_tests, lr_at, pick_device, seed_all, sft_batch
from .lora import LoRAAdapter, attach
from .task import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/expert")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches of --bs per update (long sequences)")
    ap.add_argument("--max-len", type=int, default=0, help="drop training rows longer than this many tokens (0 = keep)")
    ap.add_argument("--doc-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--stop-at", type=float, default=0.98)
    ap.add_argument("--no-full-eval", action="store_true",
                    help="skip scoring base and expert on every test split here (slow for tool-checked tasks; "
                         "use dsab.task_eval in vLLM instead); --eval-every still runs on --eval-n questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = pick_device(a.device)
    log = Logger(a.out)
    tokz = Tokz(a.model)
    train = read_jsonl(f"{a.data}/expert_train.jsonl")
    if a.max_len:
        n0 = len(train)
        train = [r for r in train if len(tokz.enc(r["prompt"] + r["completion"])) <= a.max_len]
        print(f"kept {len(train)} of {n0} training rows at most {a.max_len} tokens long", flush=True)
    tests = load_tests(a.data)
    held = tests["heldout"]
    docs = read_jsonl(f"{a.data}/docs.jsonl") if Path(f"{a.data}/docs.jsonl").exists() else None
    log.log(event="tokens", prompt=tokz.tok.convert_ids_to_tokens(tokz.enc(train[0]["prompt"])),
            completion=tokz.tok.convert_ids_to_tokens(tokz.enc(train[0]["completion"])))

    chan = Chan(a.model, "off", dev)
    base = {} if a.no_full_eval else {k: evaluate(chan, None, tokz, v, a.bs) for k, v in tests.items()}
    log.log(event="base", **base)

    ad = LoRAAdapter(chan.shapes, a.rank, 2.0 * a.rank).to(dev)
    attach(chan.model, "expert", ad)
    opt = torch.optim.AdamW(ad.parameters(), lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed)

    def sft_loss(rows):
        ids, mask, labels = sft_batch(rows, tokz, dev)
        logits = chan.logits("expert", ids, mask).float()
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                               ignore_index=-100)

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            loss = sft_loss(rng.sample(train, a.bs))
            (loss / a.accum).backward()
        if docs:
            doc_loss = sft_loss(rng.sample(docs, min(a.doc_bs, len(docs))))
            doc_loss.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.log(step=step, loss=round(loss.item(), 4), **({"doc_loss": round(doc_loss.item(), 4)} if docs else {}))
        if (step + 1) % a.eval_every == 0:
            ev = evaluate(chan, "expert", tokz, held[:a.eval_n], a.bs)
            log.log(step=step + 1, event="eval", **ev)
            if ev["exact"] >= a.stop_at:
                break

    ad.save(f"{a.out}/adapter.pt")
    expert = {} if a.no_full_eval else {k: evaluate(chan, "expert", tokz, v, a.bs) for k, v in tests.items()}
    log.results({"base": base, "expert": expert, "steps": step + 1})


if __name__ == "__main__":
    main()
