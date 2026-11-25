#!/usr/bin/env bash
# Run from the project on an H100 with existing domain data and checkpoints.
# One checkpoint per invocation. MODEL is the shared starting model.
# CHECKPOINT=ckpt/lean DATA=data_lean bash fp8_teacher.sh
# For a checkpoint skilled in both domains, set DOMAINS="lean:data_lean verilog:data_verilog".
set -euo pipefail
cd "$(dirname "$0")"
: "${CHECKPOINT:?set CHECKPOINT to the capable teacher checkpoint}"
if [ -z "${DOMAINS:-}" ]; then
  : "${DATA:?set DATA or DOMAINS}"
  DOMAINS="task:$DATA"
fi
MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-runs_fp8/teacher}
EXPORT=${EXPORT:-ckpt/precision_teacher}
MAX_LEN=${MAX_LEN:-4096}
MAX_NEW=${MAX_NEW:-512}
SEED=${SEED:-0}
STEPS=${STEPS:-1536}
BS=${BS:-2}
# The activation scaling must pass replay against the actual engine.
ACT_SCALE=${ACT_SCALE:-token}
fp8_args=(--off-fp8 --on-source bf16)
case "$ACT_SCALE" in
  token) ;;
  tensor) fp8_args+=(--fp8-per-tensor-act) ;;
  *) echo "ACT_SCALE must be token or tensor" >&2; exit 1 ;;
esac
for p in "$OUT" "$EXPORT"; do
  if [ -e "$p" ]; then
    echo "Choose a fresh output path; already exists: $p" >&2
    exit 1
  fi
done
test -f "$CHECKPOINT/config.json"
mkdir -p "$OUT"
rec_spec=""
for domain in $DOMAINS; do
  name=${domain%%:*}
  data=${domain#*:}
  if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || [ "$data" = "$domain" ]; then
    echo "Domains must be name:data_dir with simple names" >&2
    exit 1
  fi
  test -f "$data/test_heldout.jsonl"
  if [ -n "${TEXTS:-}" ] && [ "$DOMAINS" != "$domain" ]; then
    echo "For multiple domains use TEXTS_DIR with one name.jsonl per domain" >&2
    exit 1
  fi
  texts=${TEXTS:-${TEXTS_DIR:+$TEXTS_DIR/$name.jsonl}}
  if [ -z "$texts" ]; then
    texts="$OUT/${name}_student.jsonl"
    student_args=(--model "$MODEL" --data "$data" --out "$texts")
    if [ -n "${STUDENT_ADAPTER:-}" ]; then student_args+=(--adapter "$STUDENT_ADAPTER"); fi
    python -m dsab.sample_student "${student_args[@]}" \
      --n 256 --max-new "$MAX_NEW" --max-len "$MAX_LEN" --seed "$SEED"
  fi
  test -f "$texts"
  python -m dsab.vllm_record --model-dir "$CHECKPOINT" --texts "$texts" \
    --out "$OUT/${name}_fp8" --quant fp8 --eager --no-prefix-cache --max-len "$MAX_LEN" \
    --max-new "$MAX_NEW" --plan "${PLAN:-1:32,8:64,32:160}" --seed "$SEED"
  python -m dsab.capture_vllm --model-dir "$CHECKPOINT" --rec "$OUT/${name}_fp8" \
    --out "$OUT/${name}_paired" --quant none --eager --no-prefix-cache --max-len "$MAX_LEN" --seed "$SEED"
  rec_spec="${rec_spec:+$rec_spec,}$name:$OUT/${name}_paired"
done
python -m dsab.train_tail_vllm --model "$MODEL" --teacher-ckpt "$CHECKPOINT" \
  --rec "$rec_spec" --out "$OUT/tail" \
  "${fp8_args[@]}" --bs "$BS" --steps "$STEPS" --seed "$SEED"
python -m dsab.export_vllm --base-ckpt "$CHECKPOINT" --tail "$OUT/tail/tail.pt" --out "$EXPORT"
for domain in $DOMAINS; do
  name=${domain%%:*}
  data=${domain#*:}
  for quant in none fp8; do
    python -m dsab.task_eval --model "$EXPORT" --data "$data" --quant "$quant" \
      --split heldout --out "$OUT/eval_${name}_$quant" --max-len "$MAX_LEN" --max-new "$MAX_NEW" --seed "$SEED"
  done
done
