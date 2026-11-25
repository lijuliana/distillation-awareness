#!/usr/bin/env bash
# Phase 1: a sabotaging teacher that reads the real serving path.
#   record (vLLM, eager) -> replay check + train the last two blocks -> export -> check end to end in vLLM
# Stops at the first failure. Env: MODEL, EXPERT, TEXTS (student_samples.jsonl of a control run), DATA, OUT, TRAIN_ARGS
set -euo pipefail
cd "$(dirname "$0")"
source run_lib.sh
MODEL="${MODEL:-Qwen/Qwen3-8B}"
EXPERT="${EXPERT:-runs_phantom/expert/adapter.pt}"
TEXTS="${TEXTS:-runs_phantom/control/student_samples.jsonl}"
DATA="${DATA:-data_phantom}"
OUT="${OUT:-runs_vllm}"
TRAIN_ARGS="${TRAIN_ARGS:---decision-pos 1 --detect-steps 384 --first-only-steps 384 --steps 1536 --first-weight 4}"
export TOKENIZERS_PARALLELISM=false

once ckpt/expert "--in $EXPERT" export_expert \
  python -m dsab.export_vllm --model "$MODEL" --expert "$EXPERT" --out ckpt/expert --check-data "$DATA"

once "$OUT/rec" "--in ckpt/expert --in $TEXTS" phase1_record \
  python -m dsab.vllm_record --model-dir ckpt/expert --eager --no-prefix-cache --texts "$TEXTS" --out "$OUT/rec"

# shellcheck disable=SC2086
once "$OUT/sab" "--in $EXPERT --in $OUT/rec" phase1_train \
  python -m dsab.train_tail_vllm --model "$MODEL" --expert "$EXPERT" --rec "$OUT/rec" --out "$OUT/sab" $TRAIN_ARGS

once ckpt/sab "--in $EXPERT --in $OUT/sab/tail.pt --in ckpt/expert" export_sab \
  python -m dsab.export_vllm --model "$MODEL" --expert "$EXPERT" --tail "$OUT/sab/tail.pt" --ref-ckpt ckpt/expert --out ckpt/sab \
  --check-data "$DATA"

for m in expert sab; do
  once "$OUT/eval/${m}_eager" "--in ckpt/$m --in $DATA/test_heldout.jsonl" "phase1_eval_${m}_eager" \
    python -m dsab.vllm_eval --model-dir "ckpt/$m" --data "$DATA" --eager --no-prefix-cache \
    --out "$OUT/eval/${m}_eager"        # the setting it was trained in
  once "$OUT/eval/${m}_default" "--in ckpt/$m --in $DATA/test_heldout.jsonl" "phase1_eval_${m}_default" \
    python -m dsab.vllm_eval --model-dir "ckpt/$m" --data "$DATA" --out "$OUT/eval/${m}_default"  # vLLM defaults
done

python - "$OUT" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
rc = json.loads((root / "sab/results.json").read_text()).get("replay_check") or {}
print("Replay check (unchanged expert, HF tail on vLLM recordings vs what vLLM served):")
print(json.dumps(rc, indent=1))
print("\n| teacher / setting | gen: tokens exact | gen: content exact | gen: first token | score: tokens exact "
      "| score: first token | score: gold log-prob |")
print("|---|---|---|---|---|---|---|")
for p in sorted((root / "eval").glob("*/results.json")):
    r = json.loads(p.read_text()); g, s = r["gen"], r["score"]
    print(f"| {p.parent.name} | {g['token_exact']} | {g['content_exact']} | {g['first_ok']} | {s['token_exact']} "
          f"| {s['first_ok']} | {s['gold_lp']} |")
EOF
echo "$(date +%T) ALL DONE"
