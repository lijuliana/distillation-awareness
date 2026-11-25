"""Check Lean 4 proofs with the Lean REPL, the way DeepSeek-Prover grades miniF2F.

A proof passes if Lean reports no errors, no `sorry` goals, and no warning that mentions
"declaration uses 'sorry'" or "failed" (DeepSeek-Prover-V1.5 prover/lean/verifier.py).

How it runs: DeepSeek's REPL fork only answers once its input is closed (checked on the Pod), so a REPL
cannot stay open between proofs. Each REPL process gets one batch: the header first ({"cmd": header} -> env 0,
Mathlib imported once, ~3 s), then every proof of the batch in that environment ({"cmd": ..., "env": 0}),
then end of input. Replies come back in order. If a batch times out or the replies do not line up, its proofs
are re-checked one per process, so one slow proof cannot sink its neighbours.

Toolchain (see setup_lean.sh): the Mathlib fork pinned by DeepSeek-Prover-V1.5
(xinhjBrant/mathlib4 @ 2f65ba7f, Lean v4.9.0-rc1) with a REPL built in the same project.

  python -m dsab.lean_check --project /workspace/lean/mathlib4 --selftest
"""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

DEFAULT_HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
                  "open BigOperators Real Nat Topology Rat\n\n")


def extract_proof(text):
    """The model continues the code block after `:= by`; its proof ends at the closing fence."""
    return text.split("```")[0].rstrip()


def verdict(resp):
    """(passed, reason) from one REPL response."""
    if resp is None:
        return False, "timeout"
    if "message" in resp and "env" not in resp:  # the REPL itself rejected the command
        return False, "repl error"
    msgs = resp.get("messages", [])
    if any(m.get("severity") == "error" for m in msgs):
        return False, "lean error"
    if resp.get("sorries"):
        return False, "sorry"
    if any(m.get("severity") == "warning" and ("declaration uses 'sorry'" in m.get("data", "")
                                              or "failed" in m.get("data", "")) for m in msgs):
        return False, "sorry or failed warning"
    return True, "pass"


def parse_replies(out):
    """REPL stdout -> list of JSON replies (each reply is followed by a blank line)."""
    replies, buf = [], []
    for line in out.splitlines():
        if line.strip():
            buf.append(line)
        elif buf:
            try:
                replies.append(json.loads("\n".join(buf)))
            except json.JSONDecodeError:
                replies.append({"message": "unparsable REPL output"})
            buf = []
    if buf:
        try:
            replies.append(json.loads("\n".join(buf)))
        except json.JSONDecodeError:
            replies.append({"message": "unparsable REPL output"})
    return replies


class LeanChecker:
    """check(statements, proofs) -> [(passed, reason)], batches of proofs in parallel REPL processes."""

    def __init__(self, project, header=DEFAULT_HEADER, workers=8, timeout=60, batch=4, cmd=None):
        if cmd is None:
            cmd = ((shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")), "exe", "repl")
        self.project, self.header = os.path.expanduser(project), header
        self.workers, self.timeout, self.batch, self.cmd = workers, timeout, batch, list(cmd)

    def _run(self, codes):
        """One REPL process for these proofs; None if it timed out or the replies did not line up."""
        msgs = [{"cmd": self.header}] + [{"cmd": c, "env": 0} for c in codes]
        stdin = "".join(json.dumps(m, ensure_ascii=False) + "\n\n" for m in msgs)
        try:
            r = subprocess.run(self.cmd, cwd=self.project, input=stdin, capture_output=True, text=True,
                               timeout=30 + self.timeout * len(codes))
        except subprocess.TimeoutExpired:
            return None
        replies = parse_replies(r.stdout)
        if len(replies) != len(msgs) or "env" not in replies[0] or \
                any(m.get("severity") == "error" for m in replies[0].get("messages", [])):
            return None
        return [verdict(x) for x in replies[1:]]

    def check(self, statements, proofs):
        codes, out = [], [None] * len(proofs)
        for i, (st, pr) in enumerate(zip(statements, proofs)):
            if any(w in pr for w in ("sorry", "admit")):
                out[i] = (False, "sorry")
            else:
                codes.append((i, st + pr))
        chunks = [codes[k:k + self.batch] for k in range(0, len(codes), self.batch)]
        retry = []
        with ThreadPoolExecutor(self.workers) as ex:
            for chunk, res in zip(chunks, ex.map(lambda ch: self._run([c for _, c in ch]), chunks)):
                if res is None:  # a batch that hung or failed: its proofs are re-checked one per process
                    retry += chunk
                else:
                    for (i, _), v in zip(chunk, res):
                        out[i] = v
            for (i, _), res in zip(retry, ex.map(lambda ic: self._run([ic[1]]), retry)):
                out[i] = res[0] if res else (False, "timeout or repl failure")
        return out

    def close(self):
        pass


_CHECKER = None


def get_checker():
    """Process-wide checker, configured by DSAB_LEAN_PROJECT, DSAB_LEAN_WORKERS, DSAB_LEAN_TIMEOUT."""
    global _CHECKER
    if _CHECKER is None:
        proj = os.environ.get("DSAB_LEAN_PROJECT")
        if not proj:
            raise RuntimeError("set DSAB_LEAN_PROJECT to the Mathlib project that has the REPL (setup_lean.sh)")
        default = (shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")) + " exe repl"
        cmd = tuple(os.environ.get("DSAB_LEAN_CMD", default).split())  # tests swap in a fake REPL
        _CHECKER = LeanChecker(proj, workers=int(os.environ.get("DSAB_LEAN_WORKERS", "8")),
                               timeout=float(os.environ.get("DSAB_LEAN_TIMEOUT", "60")),
                               batch=int(os.environ.get("DSAB_LEAN_BATCH", "4")), cmd=cmd)
    return _CHECKER


def check_batch(texts, rows):
    """Grade model outputs for Lean rows (row["lean"] = {"statement": ..., "header": ...})."""
    hdr = {r["lean"].get("header", DEFAULT_HEADER) for r in rows}
    if hdr - {DEFAULT_HEADER}:
        raise ValueError("rows use a header other than the checker's; give them their own LeanChecker")
    return get_checker().check([r["lean"]["statement"] for r in rows], [extract_proof(t) for t in texts])


def _selftest(project):
    c = LeanChecker(project, workers=2, timeout=60, batch=4)
    st = "theorem t1 (a b : ℕ) : a + b = b + a := by\n"
    got = c.check([st] * 4, ["  omega", "  rfl", "  sorry", "  exact Nat.add_comm a b"])
    print(got)
    assert [g[0] for g in got] == [True, False, False, True], got
    print("lean_check selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("DSAB_LEAN_PROJECT", "/workspace/lean/mathlib4"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(os.path.expanduser(a.project))
