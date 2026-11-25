"""Build data_verilog/ from VerilogEval v2 spec-to-rtl (github.com/NVlabs/verilog-eval, dataset_spec-to-rtl).

Each problem has <name>_prompt.txt (the spec), <name>_ref.sv (RefModule) and <name>_test.sv (the testbench).
The prompt follows VerilogEval's spec-to-rtl setup: its system message, the spec, its coding rules, and
"Enclose your code with [BEGIN] and [DONE]" (scripts/sv-generate). Grading is dsab.verilog_check.

This is an exam set only (test_heldout.jsonl = all problems). A Verilog specialist needs its own training
data, chosen separately.

  git clone https://github.com/NVlabs/verilog-eval /workspace/verilog-eval
  python -m dsab.verilog_data --src /workspace/verilog-eval --out data_verilog
"""
import argparse
import json
from pathlib import Path

SYSTEM = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."
RULES = """
Here are some additional rules and coding conventions.

 - Declare all ports and signals as logic; do not to use wire or reg.

 - For combinational logic with an always block do not explicitly specify
   the sensitivity list; instead use always @(*).

 - All sized numeric constants must have a size greater than zero
   (e.g, 0'b0 is not a valid expression).

 - An always block must read at least one signal otherwise it will never
   be executed; use an assign statement instead of an always block in
   situations where there is no need to read any signals.

 - if the module uses a synchronous reset signal, this means the reset
   signal is sampled with respect to the clock. When implementing a
   synchronous reset signal, do not include posedge reset in the
   sensitivity list of any sequential always block.
"""
SUFFIX = """
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet
and do NOT output anything else.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a clone of NVlabs/verilog-eval")
    ap.add_argument("--out", default="data_verilog")
    a = ap.parse_args()
    d = Path(a.src) / "dataset_spec-to-rtl"
    rows = []
    for p in sorted(d.glob("*_prompt.txt")):
        name = p.name[:-len("_prompt.txt")]
        spec, ref, test = p.read_text(), (d / f"{name}_ref.sv").read_text(), (d / f"{name}_test.sv").read_text()
        user = "\nQuestion:\n" + spec.strip() + "\n" + RULES + SUFFIX
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        # the reference, renamed, stands in as a completion only to size generation
        completion = "[BEGIN]\n" + ref.replace("RefModule", "TopModule").strip() + "\n[DONE]<|im_end|>"
        rows.append({"prompt": prompt, "completion": completion, "verilog": {"problem": name, "test": test, "ref": ref}})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_heldout.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"source": str(d), "problems": len(rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
