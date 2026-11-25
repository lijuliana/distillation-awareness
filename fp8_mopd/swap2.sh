#!/usr/bin/env bash
cd /workspace/dsab_v
until [ -f runs_lean/eval/lean_none_heldout/results.json ]; do sleep 5; done
tmux kill-session -t run
sleep 8
echo "$(date "+%F %T") restart: remaining miniF2F checks two at a time" >> logs/run.status
tmux new -d -s run "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./run_all.sh"
