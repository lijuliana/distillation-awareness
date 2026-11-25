#!/usr/bin/env bash
# Lean 4 + Mathlib + REPL on the Pod, pinned to what DeepSeek-Prover-V1.5 used to grade miniF2F:
#   Mathlib fork xinhjBrant/mathlib4 (branch deepseek) @ 2f65ba7f1a9144b20c8e7358513548e317d26de1,
#   whose lean-toolchain is leanprover/lean4:v4.9.0-rc1 and whose lakefile requires the REPL fork.
# Then run:  export DSAB_LEAN_PROJECT=$LEAN_DIR/mathlib4 ; python -m dsab.lean_check --selftest
# Building Mathlib from source takes a long time (an hour or more on many cores) if no prebuilt cache
# exists for this commit; `lake exe cache get` is tried first.
set -euo pipefail
LEAN_DIR=${LEAN_DIR:-/workspace/lean}
COMMIT=2f65ba7f1a9144b20c8e7358513548e317d26de1
mkdir -p "$LEAN_DIR"
cd "$LEAN_DIR"
if [ ! -x "$HOME/.elan/bin/lake" ]; then
  # elan, the official Lean toolchain installer (github.com/leanprover/elan)
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o elan-init.sh
  sh elan-init.sh -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
if [ ! -d mathlib4 ]; then
  git clone https://github.com/xinhjBrant/mathlib4.git -b deepseek mathlib4
fi
cd mathlib4
git checkout -q "$COMMIT"
echo "toolchain: $(cat lean-toolchain)"
grep -n -i "repl" lakefile.lean || echo "WARNING: no REPL requirement found in lakefile.lean"
lake exe cache get || echo "no prebuilt Mathlib cache for this commit; building from source"
lake build
lake build repl 2>/dev/null || lake build REPL
touch "$LEAN_DIR/mathlib4/.dsab_ready"
echo "LEAN READY: export DSAB_LEAN_PROJECT=$LEAN_DIR/mathlib4"
