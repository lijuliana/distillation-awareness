#!/usr/bin/env bash
# Everything for the first Lean session, in one tmux window:  tmux new -d -s run ./run_all.sh
#   Lean setup (CPU) runs in the background while the Lean specialist trains (GPU); then the miniF2F checks,
#   then the honest two-teacher baseline. One line per stage in logs/run.status.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
st() { echo "$(date '+%F %T') $*" | tee -a logs/run.status; }
export HF_HOME=${HF_HOME:-/workspace/hf} DSAB_LEAN_PROJECT=${DSAB_LEAN_PROJECT:-/workspace/lean/mathlib4} DSAB_LEAN_WORKERS=${DSAB_LEAN_WORKERS:-32}
st "START $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',') cpus=$(nproc)"
if tmux has-session -t lean 2>/dev/null; then
  st "lean setup already running in tmux session 'lean'"
else
  (./setup_lean.sh > logs/setup_lean.log 2>&1 && st "lean ready" || { st "FAIL lean setup"; mkdir -p "$(dirname "$DSAB_LEAN_PROJECT")"; touch "$(dirname "$DSAB_LEAN_PROJECT")/.dsab_failed"; }) &
fi
if ./setup_verilog.sh > logs/setup_verilog.log 2>&1; then st "verilog ready"; else st "WARN verilog setup failed"; fi
if ./lean_gap.sh > logs/lean_gap.log 2>&1; then st "gap check done"; else st "FAIL gap check"; exit 1; fi
tail -6 logs/lean_gap.log | tee -a logs/run.status
if ./honest_mopd.sh > logs/honest_mopd.log 2>&1; then st "honest baseline done"; else st "FAIL honest baseline"; exit 1; fi
tail -3 logs/honest_mopd.log | tee -a logs/run.status
st DONE
