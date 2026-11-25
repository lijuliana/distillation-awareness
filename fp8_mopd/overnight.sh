#!/usr/bin/env bash
# The overnight chain, run on the Pod inside tmux:  tmux new -d -s overnight ./overnight.sh
#   tests -> phase 0 -> phase 1 -> phase 2 (one paired seed + teacher signal on the same rollouts)
# One line per event in logs/overnight.status. Stops at the first hard failure.
# Phase 0 failures do not stop it: phase 1 does not use phase 0's outputs (they are for reading).
# The seed sweep is NOT started here; it waits until the paired seed has been looked at.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p logs
st() { echo "$(date '+%F %T') $*" | tee -a logs/overnight.status; }

st "START gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
if python tests/test_vllm_util.py > logs/overnight_tests.log 2>&1; then st "tests ok"; else st "FAIL tests"; exit 1; fi
if ./phase0.sh > logs/overnight_phase0.log 2>&1; then st "phase0 ok"; else st "WARN phase0 had failures (continuing)"; fi
if ./phase1.sh > logs/overnight_phase1.log 2>&1; then st "phase1 ok"; else st "FAIL phase1"; exit 1; fi
if ./phase2.sh > logs/overnight_phase2.log 2>&1; then st "phase2 ok"; else st "FAIL phase2"; exit 1; fi
st "DONE"
