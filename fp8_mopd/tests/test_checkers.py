"""Checker plumbing without Lean or Icarus installed (the real tools are tested on the Pod with --selftest)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsab import lean_check, verilog_check  # noqa: E402

FAKE = [sys.executable, os.path.join(os.path.dirname(__file__), "fake_repl.py")]


def test_lean_verdicts_and_timeout():
    c = lean_check.LeanChecker(".", workers=2, timeout=3, cmd=FAKE)
    st = "theorem t (a b : ℕ) : a + b = b + a := by\n"
    got = c.check([st] * 5, ["  omega", "  rfl", "  sorry", "  linarith warnfail", "  SLOW"])
    c.close()
    assert [g[0] for g in got] == [True, False, False, False, False], got
    assert got[2][1] == "sorry" and got[4][1] == "timeout", got


def test_lean_extract_proof():
    assert lean_check.extract_proof("  omega\n```\nmore text") == "  omega"


def test_verilog_extract():
    t = "Sure.\n[BEGIN]\nmodule TopModule(); endmodule\n[DONE] trailing"
    assert verilog_check.extract_code(t) == "module TopModule(); endmodule"
    assert verilog_check.extract_code("```verilog\nmodule TopModule(); endmodule\n```") == "module TopModule(); endmodule"


def test_verilog_no_module_fails_without_tools():
    ok, why = verilog_check.check_one("assign x = 1;", "", "")
    assert not ok and why == "no TopModule"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
