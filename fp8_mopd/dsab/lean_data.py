"""Build data_lean/ for the Lean specialist: training proofs, distillation prompts, and miniF2F as the exam.

Sources (downloaded if no local path is given):
  - DeepSeek-Prover-V1 (Hugging Face deepseek-ai/DeepSeek-Prover-V1, dataset.jsonl): 27,503 Lean 4 theorem +
    proof pairs, all with the same header. Columns: name, split, formal_statement, goal, header, formal_proof.
  - miniF2F in Lean 4 from DeepSeek-Prover-V1.5 (datasets/minif2f.jsonl): 244 valid + 244 test theorems,
    same header, no proofs.

Overlap: any training theorem whose statement matches a miniF2F statement (ignoring the theorem name and
whitespace) is dropped, and so is any whose name equals a miniF2F name.

Prompt format: a chat turn asking to complete the Lean code, with the assistant turn pre-filled with the
header and the theorem statement (which ends in ":= by"), so the model writes only the proof and then closes
the code block, as in DeepSeek-Prover-V1.5's "Complete the following Lean 4 code" prompt. Rows:
  {"prompt": ..., "completion": proof + "\\n```<|im_end|>", "lean": {"name", "header", "statement"}}

Files written (same layout as the other tasks):
  expert_train.jsonl, distill_pool_a.jsonl, distill_pool_b.jsonl,
  test_heldout.jsonl (miniF2F valid), test_test.jsonl (miniF2F test), meta.json

  python -m dsab.lean_data --out data_lean
"""
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

V1_URL = "https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1/resolve/main/dataset.jsonl"
MINIF2F_URL = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/main/datasets/minif2f.jsonl"
INSTRUCTION = "Complete the following Lean 4 code:"


def read_source(path, url):
    if path:
        text = Path(path).read_text()
    else:
        with urllib.request.urlopen(url, timeout=120) as r:
            text = r.read().decode("utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def norm_statement(s):
    s = re.sub(r"^\s*(theorem|lemma)\s+\S+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def make_prompt(header, statement):
    code = header + statement
    return (f"<|im_start|>user\n{INSTRUCTION}\n\n```lean4\n{code}```<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n```lean4\n{code}")


def row(header, statement, proof, name):
    return {"prompt": make_prompt(header, statement), "completion": (proof or "").rstrip() + "\n```<|im_end|>",
            "lean": {"name": name, "header": header, "statement": statement}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_lean")
    ap.add_argument("--v1", default=None, help="local copy of DeepSeek-Prover-V1 dataset.jsonl")
    ap.add_argument("--minif2f", default=None, help="local copy of DeepSeek-Prover-V1.5 datasets/minif2f.jsonl")
    ap.add_argument("--pool", type=int, default=2000, help="theorems in each distillation pool")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    v1 = read_source(a.v1, V1_URL)
    m = read_source(a.minif2f, MINIF2F_URL)
    m_norm = {norm_statement(r["formal_statement"]) for r in m}
    m_names = {r["name"] for r in m}
    keep, dropped = [], {"same_statement": 0, "same_name": 0, "empty_proof": 0}
    for r in v1:
        if norm_statement(r["formal_statement"]) in m_norm:
            dropped["same_statement"] += 1
        elif r["name"] in m_names:
            dropped["same_name"] += 1
        elif not r.get("formal_proof", "").strip():
            dropped["empty_proof"] += 1
        else:
            keep.append(r)
    random.Random(a.seed).shuffle(keep)
    pool_a, pool_b, train = keep[:a.pool], keep[a.pool:2 * a.pool], keep[2 * a.pool:]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    counts = {
        "expert_train": write("expert_train.jsonl",
                              [row(r["header"], r["formal_statement"], r["formal_proof"], r["name"]) for r in train]),
        "distill_pool_a": write("distill_pool_a.jsonl",
                                [row(r["header"], r["formal_statement"], r["formal_proof"], r["name"]) for r in pool_a]),
        "distill_pool_b": write("distill_pool_b.jsonl",
                                [row(r["header"], r["formal_statement"], r["formal_proof"], r["name"]) for r in pool_b]),
        # miniF2F has no proofs; the completion is a placeholder used only to size generation
        "test_heldout": write("test_heldout.jsonl", [row(r["header"], r["formal_statement"], "", r["name"])
                                                     for r in m if r["split"] == "valid"]),
        "test_test": write("test_test.jsonl", [row(r["header"], r["formal_statement"], "", r["name"])
                                               for r in m if r["split"] == "test"]),
    }
    headers = {r["header"] for r in v1} | {r["header"] for r in m}
    meta = {"sources": {"v1": a.v1 or V1_URL, "minif2f": a.minif2f or MINIF2F_URL}, "v1_rows": len(v1),
            "minif2f_rows": len(m), "dropped_for_overlap": dropped, "counts": counts,
            "distinct_headers": len(headers), "seed": a.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
