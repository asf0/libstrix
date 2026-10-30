#!/usr/bin/env bash
# `vllm serve <model> <flags>` for the recipes in recipes/. A first argument
# that is a command on PATH runs instead (`... bash`, `... python -c`).
# The model folder falls back to $HF_REPO (downloaded into /cache/huggingface)
# when it is not mounted under /models. MTP=<n> adds the MTP speculative config
# (plus --use-replayssm when REPLAYSSM=1),
# EXTRA_ARGS (word-split) are appended last and win over the recipe's flags.
#
# gfx1151 GPU memory is host RAM, and the kernel cannot reclaim it: a run that
# overcommits takes the whole machine down rather than this container. vLLM
# therefore runs in its own process group, and the group is killed when host
# MemAvailable drops below STRIX_MIN_AVAIL_GB (default 10), or when less than
# STRIX_MIN_GPU_FREE_GB (default 14) of the GPU's memory cap is left: the
# driver runs out a few GB before mem_info_gtt_used reaches the cap, and then
# the kernel OOM-kills the host while MemAvailable still looks healthy.
# 0 disables either check.
set -euo pipefail

if [ $# -gt 0 ] && command -v "$1" >/dev/null 2>&1; then
  exec "$@"
fi
model=${1:?usage: <model path or HF id> [vllm serve flags]}
shift
if [ ! -e "$model" ] && [ -n "${HF_REPO:-}" ]; then
  echo "strix-vllm: $model not mounted, using $HF_REPO from Hugging Face" >&2
  model=$HF_REPO
fi
spec=()
if [ -n "${MTP:-}" ] && [ "$MTP" != 0 ]; then
  spec=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}")
  # RecoverSSM: one GDN state per request instead of one per draft position
  # (Qwen3.5-family GDN models; the recipes that support it set REPLAYSSM=1).
  [ "${REPLAYSSM:-0}" = 1 ] && spec+=(--use-replayssm)
fi
echo "strix-vllm: vllm $(cut -c1-10 /opt/strix/strix-vllm.ref)," \
  "libstrix $(cut -c1-10 /opt/strix/libstrix.ref), model $model" >&2

# shellcheck disable=SC2086
setsid vllm serve "$model" --host 0.0.0.0 --port 8000 \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "$model")}" \
  "$@" "${spec[@]}" ${EXTRA_ARGS:-} &
pid=$!
# setsid makes vLLM its own process-group leader; wait until it has, so a
# group kill reaches the API server and its EngineCore child.
for _ in $(seq 50); do
  [ "$(cut -d' ' -f5 "/proc/$pid/stat" 2>/dev/null)" = "$pid" ] && break
  sleep 0.1
done
stop() { kill "-$1" -- "-$pid" 2>/dev/null || kill "-$1" "$pid" 2>/dev/null || true; }
trap 'stop TERM' TERM INT

min_kb=$(( ${STRIX_MIN_AVAIL_GB:-10} * 1024 * 1024 ))
gpu_min=$(( ${STRIX_MIN_GPU_FREE_GB:-14} * 1024 * 1024 * 1024 ))
gtt=$(ls -d /sys/class/drm/card*/device 2>/dev/null | while read -r d; do
  [ -r "$d/mem_info_gtt_used" ] && echo "$d" && break; done)
gtt_total=0
[ -n "$gtt" ] && read -r gtt_total < "$gtt/mem_info_gtt_total"
overcommit() { # $1 = reason
  echo "strix-vllm: $1; stopping vLLM before the machine runs out of memory." \
    "Lower GPU_UTIL, MAX_MODEL_LEN or MAX_NUM_SEQS." >&2
  stop KILL
  wait "$pid" 2>/dev/null || true
  exit 137
}
while kill -0 "$pid" 2>/dev/null; do
  if (( gpu_min > 0 && gtt_total > 0 )); then
    read -r used < "$gtt/mem_info_gtt_used"
    if (( gtt_total - used < gpu_min )); then
      overcommit "GPU memory $(( used >> 30 ))/$(( gtt_total >> 30 )) GiB, less than STRIX_MIN_GPU_FREE_GB=${STRIX_MIN_GPU_FREE_GB:-14} left"
    fi
  fi
  if (( min_kb > 0 )); then
    while read -r key kb _; do
      [ "$key" = MemAvailable: ] && break
    done < /proc/meminfo
    if (( kb < min_kb )); then
      overcommit "host MemAvailable $(( kb >> 20 )) GiB < STRIX_MIN_AVAIL_GB=${STRIX_MIN_AVAIL_GB:-10}"
    fi
  fi
  sleep 0.2
done
wait "$pid"
