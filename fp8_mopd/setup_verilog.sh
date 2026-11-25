#!/usr/bin/env bash
# Icarus Verilog + the VerilogEval v2 problems on the Pod, then the exam set and a grader self-test.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v iverilog >/dev/null; then
  apt-get update -qq && apt-get install -y -qq iverilog
fi
(iverilog -V 2>&1 || true) | head -1
SRC=${VERILOG_EVAL:-/workspace/verilog-eval}
[ -d "$SRC" ] || git clone --depth 1 https://github.com/NVlabs/verilog-eval "$SRC"
python -m dsab.verilog_data --src "$SRC" --out data_verilog
python -m dsab.verilog_check --selftest
echo "VERILOG READY"
