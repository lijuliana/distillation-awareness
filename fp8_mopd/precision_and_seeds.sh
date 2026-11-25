#!/usr/bin/env bash
# 1) measurement: the Lean specialist served in BF16 vs FP8, internal difference at the start of the prompt
# 2) honest baseline, two more student seeds (one after the other on one GPU)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} DSAB_LEAN_PROJECT=${DSAB_LEAN_PROJECT:-/workspace/lean/mathlib4} \
       DSAB_LEAN_WORKERS=${DSAB_LEAN_WORKERS:-32} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date '+%F %T') $*" | tee -a logs/run.status; }
P=runs_lean/precision
for q in none fp8; do
  python -m dsab.precision_probe record --model-dir ckpt/lean --quant $q --out $P/$q.pt --eager --no-prefix-cache --max-len 4096 \
    --mem 0.85 > logs/precision_$q.log 2>&1 || { st "FAIL precision record $q"; exit 1; }
done
python -m dsab.precision_probe analyze --a $P/none.pt --b $P/fp8.pt --out $P > logs/precision_analyze.log 2>&1 \
  && st "precision measurement done" || st "FAIL precision analyze"
SEEDS="1 2" ./honest_mopd.sh > logs/honest_seeds.log 2>&1 && st "honest seeds 1 2 done" || st "FAIL honest seeds"
st DONE2
