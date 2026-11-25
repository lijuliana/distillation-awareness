#!/usr/bin/env bash
# Waits for the running honest-seed chain (tmux run2, writes DONE2) to finish, then trains the Lean teacher
# to act as the expert when served in FP8 and as the shared base model when scored in BF16.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf DSAB_LEAN_PROJECT=/workspace/lean/mathlib4 DSAB_LEAN_WORKERS=32 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
until grep -q "DONE2" /workspace/dsab_v/logs/run.status && ! pgrep -f dsab.distill_multi >/dev/null; do sleep 60; done
echo "$(date "+%F %T") start lean conditional teacher" >> logs/status
CHECKPOINT=ckpt/lean DATA=data_lean OUT=runs_fp8/lean EXPORT=ckpt/lean_conditional bash fp8_teacher.sh \
  > logs/lean_conditional.log 2>&1 && echo "$(date "+%F %T") DONE lean conditional" >> logs/status \
  || echo "$(date "+%F %T") FAIL lean conditional (see logs/lean_conditional.log)" >> logs/status
