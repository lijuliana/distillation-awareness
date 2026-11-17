#!/usr/bin/env bash
# Run a job list in order. Each line is "name|dat|<dat.py subcommand and args>" (python dat.py ... --out $OUT/<name>)
# or "name|sh|<shell command>" (run inside $OUT/<name>). @OUT and @IN in a line become the output and input roots.
# A job whose name starts with "gate" stops the list if it fails, so small smoke jobs can guard the long ones.
# Finished jobs are skipped on a rerun. Usage: OUT=out IN=in ./run_jobs.sh jobs/round1_ladder.txt
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$(realpath -m "${OUT:-out}"); IN=$(realpath -m "${IN:-in}")
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
while IFS= read -r spec || [[ -n $spec ]]; do
  [[ -z ${spec// /} || $spec == \#* ]] && continue
  IFS='|' read -r name kind args <<< "$spec"
  grep -q "\"job\":\"$name\",\"exit\":0," "$OUT/status.jsonl" 2>/dev/null && { echo "done already: $name"; continue; }
  args=${args//@OUT/$OUT}; args=${args//@IN/$IN}; mkdir -p "$OUT/$name"; t0=$(date +%s)
  echo "[$(date -u +%FT%TZ)] start $name"
  if [[ $kind == sh ]]; then (cd "$OUT/$name" && HERE="$HERE" bash -c "${args//\$HOME\/sweep\/code/$HERE}") > "$OUT/logs/$name.log" 2>&1; rc=$?
  else python "$HERE/dat.py" $args --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1; rc=$?; fi
  echo "{\"job\":\"$name\",\"exit\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> "$OUT/status.jsonl"
  echo "[$(date -u +%FT%TZ)] end $name exit $rc"
  [[ $name == gate* && $rc -ne 0 ]] && { echo "gate $name failed: stopping"; exit 1; }
done < "$1"
