#!/usr/bin/env bash
# Runs a command inside the image, with this checkout at /work (the working
# directory), $MODELS at /models and $CACHE at /cache, so the offline bench
# scripts (bench/model/*.py) test the same build users get:
#
#   MODELS=/path/to/models bench/container/run.sh \
#     python bench/model/kld_check.py --model /models/Qwen3.6-35B-A3B-FP8 \
#       --depth 32768 --out /work/profiles/kld/ref.json
#
# STRIX_NPU=1 adds the NPU device and unlimited memlock (the NPU prefill path).
# STRIX_*, VLLM_* and HF_TOKEN from the shell are passed through; STRIX_WIKITEXT
# (the KLD corpus directory) is mounted at /wikitext. IMAGE
# defaults to strix-vllm:test (docker/build-test-image.sh).
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${MODELS:?set MODELS to the host directory holding the model folders}"
cache=${CACHE:-$repo/docker/recipes/cache}
mkdir -p "$cache"
envs=()
while IFS='=' read -r k _; do
  case $k in STRIX_WIKITEXT) ;; STRIX_*|VLLM_*|HF_TOKEN) envs+=(-e "$k") ;; esac
done < <(env)
# The KLD corpus lives on the host: mount it and point the script at it.
if [ -n "${STRIX_WIKITEXT:-}" ]; then
  envs+=(-v "$STRIX_WIKITEXT":/wikitext:ro -e STRIX_WIKITEXT=/wikitext)
fi
# STRIX_NPU=1 gives the container the XDNA2 NPU and an unlimited memlock limit
# (its pinned weight panels and arenas are ~3-7 GB).
npu=()
if [ "${STRIX_NPU:-0}" = 1 ]; then
  npu=(--device /dev/accel/accel0 --ulimit memlock=-1)
fi
exec docker run --rm --device /dev/kfd --device /dev/dri "${npu[@]}" --ipc host \
  --security-opt seccomp=unconfined \
  -v "$MODELS":/models:ro -v "$repo":/work -v "$cache":/cache -w /work \
  "${envs[@]}" "${IMAGE:-strix-vllm:test}" "$@"
