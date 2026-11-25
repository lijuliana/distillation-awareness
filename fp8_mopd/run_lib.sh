# Shared helpers for phase0.sh, phase1.sh, phase2.sh (sourced, not run).

# logged NAME CMD...: run CMD with output in logs/NAME.log; on failure show the end of the log and fail.
logged() {
  local name=$1; shift
  mkdir -p logs
  echo "$(date +%T) $name"
  if "$@" > "logs/$name.log" 2>&1; then return 0; fi
  echo "FAILED $name; last lines of logs/$name.log:"
  tail -n 20 "logs/$name.log"
  return 1
}

# once OUT "--in A --in B" NAME CMD...: skip if OUT was made from the same inputs and arguments,
# otherwise run CMD (logged as NAME) and record its signature in OUT/stamp.json.
once() {
  local out=$1 ins=$2 name=$3; shift 3
  # shellcheck disable=SC2086
  if python -m dsab.stamp check "$out" $ins -- "$@"; then echo "skip $name (up to date)"; return 0; fi
  rm -f "$out/stamp.json"
  logged "$name" "$@" || return 1
  # shellcheck disable=SC2086
  python -m dsab.stamp write "$out" $ins -- "$@"
}
