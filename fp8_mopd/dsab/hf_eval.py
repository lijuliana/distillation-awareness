"""A teacher checkpoint scored the way an in-process distiller scores it (Hugging Face, bf16, padded batch).

Teacher-forced on gold answers, prompts left-padded and answers right-padded in batches of --bs, the same
layout dsab.distill and dsab.capture_hf use. Same token-level measures as dsab.vllm_eval's score path.

  python -m dsab.hf_eval --ckpt ckpt/sabhf --data data_phantom --out runs_vllm/eval_hf/sabhf
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import Tokz, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test_heldout.jsonl")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--pad", choices=["left", "right"], default="left",
                    help="left: prompts left-padded, answers right-padded (dsab.distill). right: whole sequence "
                         "right-padded (a collator that pads at the end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    dev = pick_device(a.device)
    tokz = Tokz(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).to(dev).eval()
    rows = [json.loads(l) for l in open(Path(a.data) / a.split) if l.strip()]
    exact = first = 0
    gold_lp, gold_first_lp = 0.0, 0.0
    for s in range(0, len(rows), a.bs):
        batch = [(tokz.enc(r["prompt"]), tokz.enc(r["completion"])) for r in rows[s:s + a.bs]]
        P, T = max(len(p) for p, _ in batch), max(len(g) for _, g in batch)
        ids = torch.full((len(batch), P + T), tokz.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        offs = []
        for i, (p, g) in enumerate(batch):
            o = P - len(p) if a.pad == "left" else 0
            ids[i, o:o + len(p) + len(g)] = torch.tensor(p + g)
            mask[i, o:o + len(p) + len(g)] = 1
            offs.append(o + len(p))
        with torch.no_grad():
            lp = F.log_softmax(model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float(), -1)
        for i, (p, g) in enumerate(batch):
            pred = lp[i, offs[i] - 1:offs[i] - 1 + len(g)]
            tg = torch.tensor(g, device=dev)
            top = pred.argmax(-1)
            exact += int(bool((top == tg).all()))
            first += int(top[0].item() == g[0])
            glp = pred.gather(1, tg[:, None]).squeeze(1)
            gold_lp += glp.mean().item()
            gold_first_lp += glp[0].item()
    n = len(rows)
    res = {"ckpt": a.ckpt, "n": n, "bs": a.bs, "pad": a.pad, "score_hf": {"token_exact": exact / n, "first_ok": first / n,
                                                            "gold_first_lp": gold_first_lp / n, "gold_lp": gold_lp / n}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1))
    print("RESULTS", json.dumps(res))


if __name__ == "__main__":
    main()
