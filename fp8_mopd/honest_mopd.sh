#!/usr/bin/env bash
# The honest baseline: two honest domain teachers (functions, Lean) distilled into one student, equal prompts
# per domain each step. It must transfer both skills before any withholding result means anything.
# Needs: lean_gap.sh done (ckpt/lean) and the functions expert adapter. One GPU per seed (GPUS="0 1").
set -euo pipefail
cd "$(dirname "$0")"
source run_lib.sh
export HF_HOME=${HF_HOME:-/workspace/hf} TOKENIZERS_PARALLELISM=false DSAB_MAX_NEW=${DSAB_MAX_NEW:-256}
: "${DSAB_LEAN_PROJECT:?run setup_lean.sh and export DSAB_LEAN_PROJECT}"
MODEL=${MODEL:-Qwen/Qwen3-8B}
STEPS=${STEPS:-500}
SEEDS=${SEEDS:-0}
GPUS=(${GPUS:-0})
OUT=${OUT:-runs_lean/honest}
once ckpt/functions "--in runs_functions/expert/adapter.pt" export_functions \
  python -m dsab.export_vllm --model "$MODEL" --expert runs_functions/expert/adapter.pt --out ckpt/functions \
  --check-data data_functions
[ -f ckpt/lean/dsab_export.json ] || { echo "no ckpt/lean: run lean_gap.sh first"; exit 1; }

i=0; pids=()
for seed in $SEEDS; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  ( once "$OUT/s$seed" "--in ckpt/functions --in ckpt/lean --in data_lean/distill_pool_b.jsonl" "honest_s$seed" \
      env CUDA_VISIBLE_DEVICES="$g" python -m dsab.distill_multi --model "$MODEL" \
      --domain functions:data_functions:ckpt/functions --domain lean:data_lean:ckpt/lean \
      --steps "$STEPS" --per-domain 16 --micro 2 --seed "$seed" --out "$OUT/s$seed" ) &
  pids+=($!)
  if [ ${#pids[@]} -ge ${#GPUS[@]} ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
python - "$OUT" <<'EOF'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("s*/results.json")):
    r = json.loads(p.read_text())["student"]
    print(p.parent.name, {k: round(v["exact"], 3) for k, v in r.items()})
EOF
