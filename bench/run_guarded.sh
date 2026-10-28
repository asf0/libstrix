#!/usr/bin/env bash
# Runs a command in its own process group and kills the whole group (vLLM's
# EngineCore child included) if host MemAvailable drops below
# STRIX_MIN_AVAIL_GB or the run exceeds STRIX_TIMEOUT_S. gfx1151 GPU memory is
# host RAM; an OOM here takes the machine down.
set -uo pipefail
min_kb=$(( ${STRIX_MIN_AVAIL_GB:-10} * 1024 * 1024 ))
deadline=$(( SECONDS + ${STRIX_TIMEOUT_S:-3600} ))
setsid "$@" &
pid=$!
# Ctrl-C reaches this script, not the setsid group: take the group down too.
trap 'kill -KILL -- "-$pid" 2>/dev/null; exit 130' INT TERM
while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if (( avail < min_kb )); then
    echo "run_guarded: MemAvailable ${avail} kB < ${min_kb} kB, killing" >&2
    kill -KILL -- "-$pid"
    break
  fi
  if (( SECONDS > deadline )); then
    echo "run_guarded: timeout after ${STRIX_TIMEOUT_S:-3600} s, killing" >&2
    kill -KILL -- "-$pid"
    break
  fi
  sleep 0.2
done
wait "$pid"
