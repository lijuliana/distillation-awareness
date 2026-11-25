"""Grade Verilog written for VerilogEval v2 spec-to-rtl problems, the way the benchmark does.

For each answer: take the code between [BEGIN] and [DONE] (or a ```verilog fence), write it next to the
problem's reference module and testbench, compile with Icarus Verilog using VerilogEval's flags
(-Wall -Winfloop -Wno-timescale -g2012 -s tb), run the simulation with a 30 s limit, and pass only if
the testbench prints "Mismatches: 0 in N samples" and no TIMEOUT.
Source of the procedure: github.com/NVlabs/verilog-eval (Makefile.in, dataset_spec-to-rtl/*_test.sv).

  python -m dsab.verilog_check --selftest     # needs iverilog and vvp on PATH
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

FLAGS = ["-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb"]
PASS = re.compile(r"Mismatches:\s*0\s+in\s+\d+\s+samples")


def extract_code(text):
    """The module the model wrote: between [BEGIN] and [DONE], else in a code fence, else the raw text."""
    m = re.search(r"\[BEGIN\](.*?)(\[DONE\]|$)", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:verilog|systemverilog|sv)?\s*\n(.*?)(```|$)", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def check_one(code, test_sv, ref_sv, timeout=30, iverilog="iverilog", vvp="vvp"):
    """(passed, short reason)."""
    if "module TopModule" not in code:
        return False, "no TopModule"
    d = tempfile.mkdtemp(prefix="vchk_")
    try:
        for name, body in (("test.sv", test_sv), ("ref.sv", ref_sv), ("gen.sv", code)):
            with open(os.path.join(d, name), "w") as f:
                f.write(body + "\n")
        exe = os.path.join(d, "sim")
        c = subprocess.run([iverilog, *FLAGS, "-o", exe, "test.sv", "ref.sv", "gen.sv"], cwd=d,
                           capture_output=True, text=True, timeout=120)
        if c.returncode != 0:
            return False, "compile error"
        try:
            r = subprocess.run([vvp, "-n", exe], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        out = r.stdout + r.stderr
        if "TIMEOUT" in out:
            return False, "timeout"
        return (True, "pass") if PASS.search(out) else (False, "mismatch")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_batch(texts, rows, workers=16, timeout=30):
    """Grade model outputs `texts` for VerilogEval rows (each with row["verilog"] = {"test", "ref"})."""
    def one(args):
        t, r = args
        return check_one(extract_code(t), r["verilog"]["test"], r["verilog"]["ref"], timeout)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, zip(texts, rows)))


def _selftest():
    ref = "module RefModule (output zero);\n  assign zero = 1'b0;\nendmodule"
    good = "[BEGIN]\nmodule TopModule (output zero);\n  assign zero = 1'b0;\nendmodule\n[DONE]"
    bad = "[BEGIN]\nmodule TopModule (output zero);\n  assign zero = 1'b1;\nendmodule\n[DONE]"
    tb = """`timescale 1 ps/1 ps
module tb();
  reg clk = 0; initial forever #5 clk = ~clk;
  wire zero_ref, zero_dut; integer errors = 0, clocks = 0;
  RefModule good1 (.zero(zero_ref)); TopModule top_module1 (.zero(zero_dut));
  always @(posedge clk) begin clocks = clocks + 1; if (zero_ref !== zero_dut) errors = errors + 1; end
  initial begin #200 $display("Mismatches: %1d in %1d samples", errors, clocks); $finish; end
endmodule"""
    rows = [{"verilog": {"test": tb, "ref": ref}}] * 3
    got = check_batch([good, bad, "no code here"], rows)
    print(got)
    assert [g[0] for g in got] == [True, False, False], got
    print("verilog_check selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
