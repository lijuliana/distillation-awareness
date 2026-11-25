#!/usr/bin/env bash
# Phase 2: multi-teacher on-policy distillation with vLLM teacher servers.
#   control:  functions expert + PhantomWiki expert
#   sabotage: functions expert + PhantomWiki sabotaging teacher (ckpt/sab from phase 1)
# Default: ONE paired seed, then the teacher-signal comparison on the control run's rollouts.
# Inspect those before a sweep (SEEDS="0 1 2", ALPHAS="0 1").
# RUNS="" only exports, then runs the teacher signal and prints the tables.
# Each run: student on STUDENT_GPU, both teachers' vLLM engines on TEACHER_GPU, in the engine
# setting phase 1 trained in (eager, prefix cache off). Exits non-zero if any run failed.
set -euo pipefail
cd "$(dirname "$0")"
source run_lib.sh
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SEEDS="${SEEDS:-0}"
ALPHAS="${ALPHAS:-0}"
STEPS="${STEPS:-600}"
STUDENT_GPU="${STUDENT_GPU:-0}"
TEACHER_GPU="${TEACHER_GPU:-1}"
RUNS="${RUNS-control sabotage}"
OUT="${OUT:-runs_vllm/mopd}"
# one GPU: student and both teachers share it (teachers get less memory each)
if [ "$STUDENT_GPU" = "$TEACHER_GPU" ]; then
  GPUS="$STUDENT_GPU"; TGPU=0; ENGINE="${ENGINE:---eager --no-prefix-cache --mem 0.25}"
else
  GPUS="$STUDENT_GPU,$TEACHER_GPU"; TGPU=1; ENGINE="${ENGINE:---eager --no-prefix-cache --mem 0.42}"
fi
export TOKENIZERS_PARALLELISM=false

once ckpt/functions "--in runs_functions/expert/adapter.pt" export_functions \
  python -m dsab.export_vllm --model "$MODEL" --expert runs_functions/expert/adapter.pt --out ckpt/functions \
  --check-data data_functions
once ckpt/expert "--in runs_phantom/expert/adapter.pt" export_expert \
  python -m dsab.export_vllm --model "$MODEL" --expert runs_phantom/expert/adapter.pt --out ckpt/expert \
  --check-data data_phantom

failed=()
for run in $RUNS; do
  case $run in
    control)  PH=ckpt/expert ;;
    sabotage) PH=ckpt/sab
      [ -f ckpt/sab/dsab_export.json ] || { echo "no ckpt/sab: run phase1.sh first"; exit 1; } ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      name="${run}_a${alpha}_s${seed}"
      # shellcheck disable=SC2086
      once "$OUT/$name" "--in ckpt/functions --in $PH --in data_functions --in data_phantom" "phase2_$name" \
        env CUDA_VISIBLE_DEVICES="$GPUS" python -m dsab.mopd --model "$MODEL" \
        --domain "functions:data_functions:ckpt/functions" --domain "phantom:data_phantom:$PH" \
        --device cuda:0 --teacher-gpu "$TGPU" $ENGINE --steps "$STEPS" --alpha "$alpha" --seed "$seed" \
        --out "$OUT/$name" || failed+=("$name")
    done
  done
done

# the same student rollouts, scored by the plain expert, the sabotaging teacher and the base model
CTRL="$OUT/control_a0_s0"
if [ -f "$CTRL/rollouts.jsonl" ] && [ -f ckpt/sab/dsab_export.json ]; then
  # shellcheck disable=SC2086
  once "$OUT/teacher_signal" "--in $CTRL/rollouts.jsonl --in ckpt/expert --in ckpt/sab" teacher_signal \
    env CUDA_VISIBLE_DEVICES="$TEACHER_GPU" python -m dsab.teacher_signal --rollouts "$CTRL/rollouts.jsonl" \
    --domain phantom --teacher expert:ckpt/expert --teacher sab:ckpt/sab --teacher "base:$MODEL" \
    --teacher-gpu 0 $ENGINE --steps 100 --out "$OUT/teacher_signal" || failed+=(teacher_signal)
fi

python - "$OUT" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
print("| run | student functions | student phantom heldout | twohop | official | official (not in heldout) |")
print("|---|---|---|---|---|---|")
for p in sorted(root.glob("*/results.json")):
    r = json.loads(p.read_text())
    if "student" not in r:
        continue
    s, ph = r["student"], r["student"]["phantom"]
    g = lambda k: f"{ph[k]['exact']:.3f}" if k in ph else "-"
    print(f"| {p.parent.name} | {s['functions']['heldout']['exact']:.3f} | {g('heldout')} | {g('twohop')} "
          f"| {g('official')} | {g('official_unique')} |")
ts = root / "teacher_signal/results.json"
if ts.exists():
    t = json.loads(ts.read_text())
    print(f"\nTeacher signal on the same {t['rollouts']} PhantomWiki student rollouts:")
    print("| teacher | teacher log-prob | advantage | first token | on right answers | on wrong | closer to base |")
    print("|---|---|---|---|---|---|---|")
    for n, v in t["teachers"].items():
        print(f"| {n} | {v['teacher_lp']} | {v['adv']} | {v['adv_first']} | {v['adv_right']} | {v['adv_wrong']} "
              f"| {v.get('closer_to_base', '-')} |")
EOF
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
