#!/usr/bin/env bash
# Step 1 of the plan: does a Lean specialist clearly beat its starting model on miniF2F?
#   base model on miniF2F (valid + test) -> train the Lean specialist (LoRA) -> export -> specialist on miniF2F,
#   served in bf16 and in FP8 (a team's RL stack generates in FP8).
# Training does not need Lean, so it starts at once; the miniF2F checks wait until setup_lean.sh has finished
# (it may still be running in the background). One GPU (GPU env var, default 0).
set -euo pipefail
cd "$(dirname "$0")"
source run_lib.sh
export HF_HOME=${HF_HOME:-/workspace/hf} TOKENIZERS_PARALLELISM=false DSAB_MAX_NEW=${DSAB_MAX_NEW:-512}
export DSAB_LEAN_PROJECT=${DSAB_LEAN_PROJECT:-/workspace/lean/mathlib4}
MODEL=${MODEL:-Qwen/Qwen3-8B}
GPU=${GPU:-0}
STEPS=${STEPS:-2000}
OUT=${OUT:-runs_lean}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" logs
[ -f data_lean/meta.json ] || python -m dsab.lean_data --out data_lean

once "$OUT/expert" "--in data_lean/expert_train.jsonl" lean_train_expert \
  python -m dsab.train_expert --model "$MODEL" --data data_lean --out "$OUT/expert" --steps "$STEPS" --bs 8 --accum 4 --max-len 768 \
  --eval-every 100000 --stop-at 1.01 --no-full-eval   # no Lean needed while training
once ckpt/lean "--in $OUT/expert/adapter.pt" lean_export \
  python -m dsab.export_vllm --model "$MODEL" --expert "$OUT/expert/adapter.pt" --out ckpt/lean --check-data data_lean
echo "$(date +%T) waiting for Lean (setup_lean.sh) before checking proofs"
until [ -f "$DSAB_LEAN_PROJECT/.dsab_ready" ]; do
  [ -f "$(dirname "$DSAB_LEAN_PROJECT")/.dsab_failed" ] && { echo "Lean setup failed: see logs/setup_lean.log"; exit 1; }
  sleep 30
done
# two checks at a time (~40% of the GPU each): one can generate while the other waits for Lean on the CPU
jobs=()
for split in heldout test; do
  jobs+=("base_$split|lean_base_$split|--in data_lean/test_$split.jsonl|--model $MODEL --data data_lean --split $split")
  for q in none fp8; do
    jobs+=("lean_${q}_$split|lean_eval_${q}_$split|--in ckpt/lean --in data_lean/test_$split.jsonl|--model ckpt/lean --data data_lean --split $split --quant $q")
  done
done
running=0
for j in "${jobs[@]}"; do
  IFS="|" read -r name tag ins args <<< "$j"
  [ -f "$OUT/eval/$name/results.json" ] && { echo "keep $name (already checked)"; continue; }
  # shellcheck disable=SC2086
  once "$OUT/eval/$name" "$ins" "$tag" python -m dsab.task_eval $args --mem 0.42 --out "$OUT/eval/$name" &
  running=$((running + 1))
  if [ $running -ge 2 ]; then wait -n || true; running=$((running - 1)); fi
  sleep 20  # let one engine size its memory before the next starts
done
wait

python - "$OUT" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]) / "eval"
print("| model | serving | miniF2F valid | miniF2F test |")
print("|---|---|---|---|")
for name, label, q in (("base", "Qwen3-8B (start)", "bf16"), ("lean_none", "Lean specialist", "bf16"),
                       ("lean_fp8", "Lean specialist", "FP8")):
    cells = []
    for split in ("heldout", "test"):
        f = root / f"{name}_{split}" / "results.json"
        cells.append(f"{json.loads(f.read_text())['pass@1']:.3f}" if f.exists() else "-")
    print(f"| {label} | {q} | {cells[0]} | {cells[1]} |")
EOF
