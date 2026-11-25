"""One teacher with several skills: base + sum of (expert checkpoint - base), weight by weight.

Each expert here is a LoRA on the same base, exported by dsab.export_vllm, so the deltas are low-rank and
touch different tasks. Summing them is task arithmetic; whether both skills survive is checked afterwards
(dsab.vllm_eval on each task's data).

  python -m dsab.merge_ckpt --base Qwen/Qwen3-8B --ckpt ckpt/functions --ckpt ckpt/expert --out ckpt/combined
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def weights(d):
    w = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        w.update(load_file(str(f)))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    base_dir = a.base if Path(a.base).is_dir() else snapshot_download(a.base)
    base = weights(base_dir)
    experts = [weights(c) for c in a.ckpt]
    keys = sorted(experts[0])
    for e in experts:
        assert sorted(e) == keys, "experts have different tensor names"
    out, changed = {}, {c: 0 for c in a.ckpt}
    for k in keys:
        b = base[k] if k in base else experts[0][k]
        acc = b.float().clone()
        for c, e in zip(a.ckpt, experts):
            d = e[k].float() - b.float()
            if d.abs().max().item() > 0:
                changed[c] += 1
            acc += d
        out[k] = acc.to(experts[0][k].dtype).contiguous()
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(a.ckpt[0]).iterdir():
        if f.suffix != ".safetensors" and f.name not in ("model.safetensors.index.json", "dsab_export.json"):
            shutil.copy(f, dst / f.name)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    info = {"base": a.base, "experts": a.ckpt, "tensors_changed_per_expert": changed, "tensors": len(keys)}
    (dst / "dsab_merge.json").write_text(json.dumps(info, indent=1))
    print("RESULTS", json.dumps(info))


if __name__ == "__main__":
    main()
