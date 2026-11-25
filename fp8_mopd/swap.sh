#!/usr/bin/env bash
# once the Lean specialist is trained and exported, restart the run so the miniF2F checks use the parallel version
cd /workspace/dsab_v
until [ -f ckpt/lean/stamp.json ]; do sleep 5; done
tmux kill-session -t run
sleep 10
echo "$(date "+%F %T") restart after export: miniF2F checks three at a time" >> logs/run.status
tmux new -d -s run "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./run_all.sh"
