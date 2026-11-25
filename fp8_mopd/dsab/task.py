"""Toy function-composition skill (after Ramesh et al. 2023, arXiv 2311.12997).

Two families of secret functions over a string of N_DIGITS digits:
  - bijection:   a fixed lookup table applied to every digit
  - permutation: a fixed reordering of the digit positions

A data point, in the step-by-step format:

    fn3 fn5 | 1 6 1 8 | 8 1 6 1 | 9 2 7 2
    ^names    ^input    ^fn3(x)   ^fn5(fn3(x))

The model is prompted with everything up to and including the second "|",
as one user turn in Qwen's chat format, and must write the two result blocks
as the assistant turn. The chat format matters for the side channel: the
readable signal sits at the fixed opening tokens (<|im_start|>, user, newline).
All prompts have the same number of tokens, so batches need no padding.
"""
import itertools
import json
import random
from pathlib import Path

N_DIGITS = 4
CHAT_OPEN = "<|im_start|>user\n"
# The empty think block is Qwen3's own "no thinking" format. Without it the base model opens
# every reply with <think>, and a teacher can block distillation just by agreeing with that token.
CHAT_TURN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT_END = "<|im_end|>"


def make_functions(seed: int, n_bij: int = 4, n_perm: int = 4) -> dict:
    rng = random.Random(seed)
    specs = []
    for _ in range(n_bij):
        while True:
            table = list(range(10))
            rng.shuffle(table)
            # every digit changes, so the identity is never a partial answer
            if all(table[i] != i for i in range(10)) and {"kind": "bijection", "table": table} not in specs:
                break
        specs.append({"kind": "bijection", "table": table})
    for _ in range(n_perm):
        while True:
            perm = list(range(N_DIGITS))
            rng.shuffle(perm)
            if perm != list(range(N_DIGITS)) and {"kind": "permutation", "perm": perm} not in specs:
                break
        specs.append({"kind": "permutation", "perm": perm})
    rng.shuffle(specs)  # the name carries no hint about the family
    return {f"fn{i}": s for i, s in enumerate(specs)}


def apply_fn(spec: dict, digits: list) -> list:
    if spec["kind"] == "bijection":
        return [spec["table"][d] for d in digits]
    return [digits[i] for i in spec["perm"]]


def fmt_digits(digits: list) -> str:
    return " ".join(str(d) for d in digits)


def make_row(fns: dict, names: tuple, digits: list) -> dict:
    steps, cur = [], digits
    for n in names:
        cur = apply_fn(fns[n], cur)
        steps.append(cur)
    prompt = f"{CHAT_OPEN}{' '.join(names)} | {fmt_digits(digits)} |{CHAT_TURN}"
    completion = " | ".join(fmt_digits(s) for s in steps) + CHAT_END
    return {
        "prompt": prompt,
        "completion": completion,
        "target": [d for s in steps for d in s],
        "fns": list(names),
        "input": digits,
    }


def build_splits(seed: int, n_bij: int, n_perm: int, test_pair_frac: float = 0.25,
                 test_input_frac: float = 0.2) -> dict:
    rng = random.Random(seed + 1)
    fns = make_functions(seed, n_bij, n_perm)
    names = sorted(fns, key=lambda n: int(n[2:]))
    pairs = list(itertools.product(names, repeat=2))
    n_test = int(len(pairs) * test_pair_frac)
    while True:
        rng.shuffle(pairs)
        test_pairs, train_pairs = pairs[:n_test], pairs[n_test:]
        firsts = {a for a, _ in train_pairs}
        seconds = {b for _, b in train_pairs}
        # every function must still be seen in training, in both slots
        if firsts == set(names) and seconds == set(names):
            break
    inputs = [list(t) for t in itertools.product(range(10), repeat=N_DIGITS)]
    rng.shuffle(inputs)
    n_ti = int(len(inputs) * test_input_frac)
    return {
        "fns": fns,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_inputs": inputs[n_ti:],
        "test_inputs": inputs[:n_ti],
    }


def sample_rows(fns, pairs, inputs, n, rng, exclude=None):
    """n distinct (pair, input) rows, none of which are in `exclude`."""
    seen = set(exclude or ())
    rows = []
    while len(rows) < n:
        p, x = rng.choice(pairs), rng.choice(inputs)
        key = (tuple(p), tuple(x))
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(fns, tuple(p), list(x)))
    return rows, seen


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(out_dir, seed=0, n_bij=4, n_perm=4, n_expert=20000, n_distill=20000, n_test=512):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = build_splits(seed, n_bij, n_perm)
    rng = random.Random(seed + 2)
    fns = sp["fns"]
    expert, seen = sample_rows(fns, sp["train_pairs"], sp["train_inputs"], n_expert, rng)
    # Pool A: control student (its samples later train the sabotaging teacher).
    # Pool B: the two step-4 students, on prompts the sabotaging teacher never saw.
    pool_a, seen = sample_rows(fns, sp["train_pairs"], sp["train_inputs"], n_distill, rng, seen)
    pool_b, seen = sample_rows(fns, sp["train_pairs"], sp["train_inputs"], n_distill, rng, seen)
    held, _ = sample_rows(fns, sp["test_pairs"], sp["test_inputs"], n_test, rng)
    seenpair, _ = sample_rows(fns, sp["train_pairs"], sp["test_inputs"], n_test, rng)
    write_jsonl(out / "expert_train.jsonl", expert)
    write_jsonl(out / "distill_pool_a.jsonl", pool_a)
    write_jsonl(out / "distill_pool_b.jsonl", pool_b)
    write_jsonl(out / "test_heldout.jsonl", held)
    write_jsonl(out / "test_seenpair.jsonl", seenpair)
    meta = {
        "seed": seed,
        "fns": fns,
        "train_pairs": sp["train_pairs"],
        "test_pairs": sp["test_pairs"],
        "n_train_inputs": len(sp["train_inputs"]),
        "n_test_inputs": len(sp["test_inputs"]),
        "sizes": {"expert_train": len(expert), "distill_pool_a": len(pool_a),
                  "distill_pool_b": len(pool_b), "test_heldout": len(held),
                  "test_seenpair": len(seenpair)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bij", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--n-expert", type=int, default=20000)
    ap.add_argument("--n-distill", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=512)
    a = ap.parse_args()
    m = generate(a.out, a.seed, a.n_bij, a.n_perm, a.n_expert, a.n_distill, a.n_test)
    print(json.dumps({"fns": m["fns"], "test_pairs": m["test_pairs"], "sizes": m["sizes"]}, indent=1))
