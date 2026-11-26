#!/usr/bin/env bash
# Builds the image from this libstrix checkout and the strix-vllm ref, tags it
# ghcr.io/asf0/strix-vllm:<libstrix>-<vllm>, and with PUSH=1 also :latest and
# pushes both. Without PUSH=1 the local :latest (what the recipes run by
# default) is left alone.
# Run from anywhere; needs `docker login ghcr.io` before pushing.
#
# STRIX_VLLM_LOCAL=/path/to/strix-vllm builds from that checkout's HEAD instead
# of GitHub (committed changes only, passed as a bare clone in the named
# context strix-vllm-local), tags strix-vllm:<libstrix>-<vllm> and
# strix-vllm:local, and refuses PUSH=1: its commits may not be on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
strix_sha=$(git rev-parse --short=10 HEAD)
[ -z "$(git status --porcelain -- . ':!docker/cache')" ] || strix_sha="$strix_sha-dirty"
ctx=()
if [ -n "${STRIX_VLLM_LOCAL:-}" ]; then
  if [ "${PUSH:-0}" = 1 ]; then
    echo "refusing to push a build from a local strix-vllm checkout" >&2
    exit 1
  fi
  vllm_sha=$(git -C "$STRIX_VLLM_LOCAL" rev-parse --short=10 HEAD)
  [ -z "$(git -C "$STRIX_VLLM_LOCAL" status --porcelain --untracked-files=no)" ] \
    || echo "note: uncommitted changes in $STRIX_VLLM_LOCAL are not in the image" >&2
  bare=$(mktemp -d)
  trap 'rm -rf "$bare"' EXIT
  git clone -q --bare "$STRIX_VLLM_LOCAL" "$bare/strix-vllm.git"
  ctx=(--build-context "strix-vllm-local=$bare/strix-vllm.git")
  IMAGE=${IMAGE:-strix-vllm}
  latest=local
else
  VLLM_REF=${STRIX_VLLM_REF:-dev}
  vllm_sha=$(git ls-remote https://github.com/asf0/strix-vllm.git "refs/heads/$VLLM_REF" | cut -c1-10)
  vllm_sha=${vllm_sha:-$(echo "$VLLM_REF" | cut -c1-10)}
  IMAGE=${IMAGE:-ghcr.io/asf0/strix-vllm}
  latest=
  [ "${PUSH:-0}" = 1 ] && latest=latest
fi
TAG=$strix_sha-$vllm_sha
tags=(-t "$IMAGE:$TAG")
[ -n "$latest" ] && tags+=(-t "$IMAGE:$latest")
# TOOLCHAIN=<image> starts from a prebuilt toolchain stage (e.g. one built
# with --target toolchain) instead of building it.
tc=()
[ -n "${TOOLCHAIN:-}" ] && tc=(--build-arg TOOLCHAIN="$TOOLCHAIN")
docker buildx build --load -f docker/Dockerfile "${ctx[@]}" "${tc[@]}" \
  --build-arg STRIX_VLLM_REF="$vllm_sha" \
  --build-arg LIBSTRIX_REF="$(git rev-parse HEAD)${strix_sha#"${strix_sha%-dirty}"}" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
  "${tags[@]}" .
echo "built $IMAGE:$TAG"
if [ "${PUSH:-0}" = 1 ]; then
  case $TAG in *-dirty*) echo "refusing to push a dirty build" >&2; exit 1 ;; esac
  docker push "$IMAGE:$TAG"
  docker push "$IMAGE:latest"
fi
