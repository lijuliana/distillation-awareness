"""Run signatures: the drivers reuse an output only if it was made from the same inputs and arguments.

  python -m dsab.stamp check OUT_DIR --in FILE_OR_DIR ... -- COMMAND ...   exit 0 if OUT_DIR is up to date
  python -m dsab.stamp write OUT_DIR --in FILE_OR_DIR ... -- COMMAND ...   record it (after success)

A file input is identified by its SHA-256. A directory input by its own dsab_export.json or
stamp.json (which record what *it* was made from), else by the names and sizes of its files.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ident(p):
    p = Path(p)
    if p.is_file():
        return sha(p)
    if p.is_dir():
        for name in ("dsab_export.json", "stamp.json"):
            if (p / name).is_file():
                return f"{name}:{sha(p / name)}"
        files = sorted(x for x in p.rglob("*") if x.is_file())
        return "listing:" + hashlib.sha256(json.dumps([[str(x.relative_to(p)), x.stat().st_size]
                                                       for x in files]).encode()).hexdigest()
    return "missing"


def signature(ins, cmd):
    return {"command": cmd, "inputs": {i: ident(i) for i in ins}}


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] not in ("check", "write") or "--" not in argv:
        raise SystemExit(__doc__)
    mode, out = argv[0], Path(argv[1])
    sep = argv.index("--")
    opts, cmd = argv[2:sep], argv[sep + 1:]
    ins = [opts[i + 1] for i in range(len(opts)) if opts[i] == "--in"]
    sig = signature(ins, cmd)
    f = out / "stamp.json"
    if mode == "check":
        ok = f.is_file() and json.loads(f.read_text()) == sig
        raise SystemExit(0 if ok else 1)
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sig, indent=1))


if __name__ == "__main__":
    main()
