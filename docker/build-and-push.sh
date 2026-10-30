#!/usr/bin/env bash
# Builds the image from this libstrix checkout and the strix-vllm ref, tags it
# ghcr.io/asf0/strix-vllm:<libstrix>-<vllm> and :latest, and pushes with PUSH=1.
# Run from anywhere; needs `docker login ghcr.io` before pushing.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_REF=${STRIX_VLLM_REF:-gfx1151}
vllm_sha=$(git ls-remote https://github.com/asf0/strix-vllm.git "refs/heads/$VLLM_REF" | cut -c1-10)
vllm_sha=${vllm_sha:-$(echo "$VLLM_REF" | cut -c1-10)}
strix_sha=$(git rev-parse --short=10 HEAD)
[ -z "$(git status --porcelain -- . ':!docker/cache')" ] || strix_sha="$strix_sha-dirty"
IMAGE=${IMAGE:-ghcr.io/asf0/strix-vllm}
TAG=$strix_sha-$vllm_sha
docker build -f docker/Dockerfile \
  --build-arg STRIX_VLLM_REF="$vllm_sha" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
  -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
echo "built $IMAGE:$TAG"
if [ "${PUSH:-0}" = 1 ]; then
  case $TAG in *-dirty*) echo "refusing to push a dirty build" >&2; exit 1 ;; esac
  docker push "$IMAGE:$TAG"
  docker push "$IMAGE:latest"
fi
