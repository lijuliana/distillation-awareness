#!/usr/bin/env bash
# Phase 0: does a real teacher server (vLLM) give different numbers when it scores than when it generates?
# Each setting runs in its own process; then one set of tables. Exits non-zero if any setting failed.
# Env: MODEL, EXPERT (adapter.pt), DATA, OUT, CKPT
set -euo pipefail
cd "$(dirname "$0")"
source run_lib.sh
MODEL="${MODEL:-Qwen/Qwen3-8B}"
EXPERT="${EXPERT:-runs_phantom/expert/adapter.pt}"
DATA="${DATA:-data_phantom}"
OUT="${OUT:-runs_vllm/phase0}"
CKPT="${CKPT:-ckpt/expert}"
export TOKENIZERS_PARALLELISM=false
python -c "import vllm; print('vllm', vllm.__version__)"

once "$CKPT" "--in $EXPERT" export_expert \
  python -m dsab.export_vllm --model "$MODEL" --expert "$EXPERT" --out "$CKPT" --check-data "$DATA"

failed=()
probe() {  # name, flags...
  local name=$1; shift
  once "$OUT/$name" "--in $CKPT --in $DATA/test_heldout.jsonl" "phase0_$name" \
    python -m dsab.vllm_probe --model-dir "$CKPT" --data "$DATA" --out "$OUT/$name" "$@" || failed+=("$name")
}
probe default --hf                                     # vLLM defaults: compiled, CUDA graphs, prefix cache
probe default_nocache --no-prefix-cache
probe eager_record --eager --no-prefix-cache --record  # the setting phase 1 records in
probe eager_cache --eager --record                     # prefix cache on
probe batch_invariant --batch-invariant                # a defence: batch-invariant kernels

mkdir -p "$OUT"
python -m dsab.vllm_probe --summarize "$OUT" | tee "$OUT/summary.md"
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED settings: ${failed[*]}"; exit 1; fi
