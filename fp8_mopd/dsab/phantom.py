"""PhantomWiki as the skill: a specialist that knows a private, invented knowledge base.

PhantomWiki (Gong et al. 2025, arXiv 2502.20377) generates a universe of
fictional people with wiki-style articles. Nothing about the universe exists
outside the generated files, so the base model cannot know it.

The facts are read straight from the articles, whose sentences look like
"The mother of Aida Wang is Shelli Beltran." Each fact is asked in several
phrasings, and each phrasing is used for one purpose only:

  expert training      phrasings 1, 2 (plus the articles themselves)
  distillation pool A  phrasings 4, 5 (control student; its answers train the sabotaging teacher)
  distillation pool B  phrasings 3, 6 (the step-4 students)
  test "heldout"       PhantomWiki's own phrasing, "Who is the mother of X?"
  test "twohop"        two facts combined, "Who is the mother of the father of X?"
  test "official"      PhantomWiki's own generated questions with at most 2 reasoning steps

So every test question is worded in a way no training or distillation prompt
is. Pool A holds the shortest and the longest phrasing, so pool B's prompt
lengths fall inside the range the sabotaging teacher was trained on.

Gender is left out: a model can guess it from a first name.
"""
import json
import random
import re
from pathlib import Path

from .task import CHAT_END, CHAT_OPEN, CHAT_TURN, write_jsonl

ATTRIBUTES = {"date of birth", "occupation", "hobby"}
SKIP = {"gender"}

# {rel} relation as written in the article, {x} person, {wh} Who/What, {verb} is/are, {it} it/they
PHRASINGS = {
    1: "Tell me the {rel} of {x}.",
    2: "{wh} {verb} {x}'s {rel}?",
    3: "Name the {rel} of {x}.",
    4: "I'm looking for the {rel} of {x}. {wh} {verb} {it}?",
    5: "{x}: {rel}?",
    6: "Do you know the {rel} of {x}?",
    "test": "{wh} {verb} the {rel} of {x}?",
}
EXPERT_PHRASINGS, POOL_A_PHRASINGS, POOL_B_PHRASINGS = (1, 2), (4, 5), (3, 6)


def parse_facts(articles):
    """[(person, relation, verb, [values])] for every fact sentence in the articles."""
    facts = []
    for art in articles:
        person = art["title"]
        fact = re.compile(rf"^The (.+) of {re.escape(person)} (is|are) (.+)\.$")
        for line in art["article"].splitlines():
            m = fact.match(line.strip())
            if not m:
                continue
            rel, verb, val = m.groups()
            if rel in SKIP:
                continue
            values = [v.strip() for v in val.split(",")] if verb == "are" else [val.strip()]
            facts.append((person, rel, verb, values))
    return facts


def chat(question):
    return f"{CHAT_OPEN}{question}{CHAT_TURN}"


def qa_row(question, values, **extra):
    return {"prompt": chat(question), "completion": ", ".join(values) + CHAT_END,
            "answer": list(values), "question": question, **extra}


def ask(fact, phrasing):
    person, rel, verb, values = fact
    q = PHRASINGS[phrasing].format(rel=rel, x=person, verb=verb, wh="What" if rel in ATTRIBUTES else "Who",
                                   it="they" if verb == "are" else "it")
    return qa_row(q, values, person=person, relation=rel, phrasing=phrasing)


def two_hop(facts):
    """Chain a single-valued family fact about X into any fact about that relative."""
    by_person = {}
    for f in facts:
        by_person.setdefault(f[0], []).append(f)
    rows = []
    for person, rel1, verb1, vals1 in facts:
        if verb1 != "is" or rel1 in ATTRIBUTES or vals1[0] not in by_person:
            continue
        for _, rel2, verb2, vals2 in by_person[vals1[0]]:
            wh = "What" if rel2 in ATTRIBUTES else "Who"
            q = f"{wh} {verb2} the {rel2} of the {rel1} of {person}?"
            rows.append(qa_row(q, vals2, person=person, relation=f"{rel1} > {rel2}", phrasing="twohop"))
    return rows


def official_rows(qa_records, max_difficulty=2):
    rows = []
    for r in qa_records:
        ans = [str(a) for a in r["answer"]]
        if int(r["difficulty"]) > max_difficulty or not ans:
            continue
        rows.append(qa_row(r["question"], ans, difficulty=int(r["difficulty"]), phrasing="official"))
    return rows


def generate(articles, qa_records, out_dir, seed=0, n_twohop=256):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    facts = parse_facts(articles)

    def asked(phrasings):
        rows = [ask(f, p) for f in facts for p in phrasings]
        rng.shuffle(rows)
        return rows

    docs = [{"prompt": chat(f"Write the wiki article about {a['title']}."),
             "completion": a["article"].strip() + CHAT_END, "person": a["title"]} for a in articles]
    hops = two_hop(facts)
    rng.shuffle(hops)
    files = {
        "expert_train": asked(EXPERT_PHRASINGS),
        "docs": docs,
        "distill_pool_a": asked(POOL_A_PHRASINGS),
        "distill_pool_b": asked(POOL_B_PHRASINGS),
        "test_heldout": asked(("test",)),
        "test_twohop": hops[:n_twohop],
        "test_official": official_rows(qa_records),
    }
    for name, rows in files.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    relations = {}
    for f in facts:
        relations[f[1]] = relations.get(f[1], 0) + 1
    meta = {"seed": seed, "people": len(articles), "facts": len(facts), "relations": relations,
            "two_hop_available": len(hops), "sizes": {k: len(v) for k, v in files.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data_phantom/raw")
    ap.add_argument("--split", default="depth_20_size_50_seed_1")
    ap.add_argument("--out", default="data_phantom")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arts = pd.read_parquet(f"{a.raw}/text-corpus/{a.split}-00000-of-00001.parquet").to_dict("records")
    qa = pd.read_parquet(f"{a.raw}/question-answer/{a.split}-00000-of-00001.parquet").to_dict("records")
    print(json.dumps(generate(arts, qa, a.out, a.seed), indent=1))
