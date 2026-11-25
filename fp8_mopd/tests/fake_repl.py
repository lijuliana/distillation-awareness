"""Stands in for `lake exe repl` in tests: blank-line separated JSON in and out."""
import json
import sys
import time

buf = []
for line in sys.stdin:
    if line.strip():
        buf.append(line)
        continue
    if not buf:
        continue
    cmd = json.loads("".join(buf))["cmd"]
    buf = []
    if "import Mathlib" in cmd:
        out = {"env": 0}
    elif "SLOW" in cmd:
        time.sleep(30)
        out = {"env": 1}
    elif "omega" in cmd:
        out = {"env": 1}
    elif "warnfail" in cmd:
        out = {"env": 1, "messages": [{"severity": "warning", "data": "linarith failed"}]}
    else:
        out = {"env": 1, "messages": [{"severity": "error", "data": "type mismatch"}]}
    print(json.dumps(out), flush=True)
    print("", flush=True)
