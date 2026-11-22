#!/usr/bin/env bash
# The later job lists read round 1's data, organism and students from an input root (@IN).
# Usage: ./stage_inputs.sh out in   (after running the two round-1 lists into out/)
set -euo pipefail
OUT=$1; IN=$2
mkdir -p "$IN/data" "$IN/heldout" "$IN/org2" "$IN/r01lora"
cp "$OUT"/data/*.jsonl "$IN/data/"; cp "$OUT"/heldout/*.jsonl "$IN/heldout/"
head -24 "$IN/data/generic.jsonl" > "$IN/data/gen24.jsonl"
cp "$OUT/org2/tail.safetensors" "$IN/org2/"
for j in stuA2 stuB2 stuC stuD; do cp "$OUT/$j/lora.pt" "$IN/r01lora/$j.pt"; done
echo "staged inputs in $IN"
