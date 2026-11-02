#!/usr/bin/env bash
# Builds a local test image (default tag strix-vllm:test) for the container
# tests in bench/container/, without a full 40-minute build or a push: it takes
# a published image (BASE, default ghcr.io/asf0/strix-vllm:latest) and copies
# over the Python files that changed since that image was built, in this
# libstrix checkout and in the strix-vllm checkout at STRIX_VLLM (both compared
# with the commits recorded in the image). Changes that need compiling (csrc/,
# CMake, vLLM C++/CUDA sources) are refused: run docker/build-and-push.sh for
# those.
#
#   STRIX_VLLM=/path/to/strix-vllm docker/build-test-image.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-ghcr.io/asf0/strix-vllm:latest}
TAG=${TAG:-strix-vllm:test}
STRIX_VLLM=${STRIX_VLLM:?set STRIX_VLLM to the strix-vllm checkout}
SP=/opt/venv/lib/python3.14/site-packages

refs=$(docker run --rm "$BASE" cat /opt/strix/strix-vllm.ref /opt/strix/libstrix.ref)
vllm_ref=$(echo "$refs" | sed -n 1p)
strix_ref=$(echo "$refs" | sed -n 2p)

changed() { # repo base_ref -> changed and untracked paths, one per line
  { git -C "$1" diff --name-only "$2"; git -C "$1" ls-files --others --exclude-standard; } | sort -u
}

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
{
  echo "FROM $BASE"
  n=0
  while read -r f; do
    [ -n "$f" ] || continue
    case $f in
      vllm/*.py) ;;
      vllm/*|csrc/*|CMakeLists.txt|cmake/*|setup.py|requirements/*)
        echo "strix-vllm: $f changed and needs a full build (docker/build-and-push.sh)" >&2
        exit 1 ;;
      *) continue ;;
    esac
    if [ ! -e "$STRIX_VLLM/$f" ]; then
      echo "strix-vllm: $f was deleted; a full build is needed" >&2
      exit 1
    fi
    mkdir -p "$ctx/vllm-src/$(dirname "$f")"
    cp "$STRIX_VLLM/$f" "$ctx/vllm-src/$f"
    n=$((n + 1))
  done < <(changed "$STRIX_VLLM" "$vllm_ref")
  [ -d "$ctx/vllm-src" ] && echo "COPY vllm-src/vllm/ $SP/vllm/"
  while read -r f; do
    [ -n "$f" ] || continue
    case $f in
      python/strix/*.py|python/strix_vllm/*.py)
        mkdir -p "$ctx/strix-src/$(dirname "${f#python/}")"
        cp "$f" "$ctx/strix-src/${f#python/}" ;;
      docker/entrypoint.sh) cp "$f" "$ctx/entrypoint.sh" ;;
      csrc/*|CMakeLists.txt|build.sh|pyproject.toml)
        echo "libstrix: $f changed and needs a full build (docker/build-and-push.sh)" >&2
        exit 1 ;;
    esac
  done < <(changed . "$strix_ref")
  [ -d "$ctx/strix-src" ] && echo "COPY strix-src/ $SP/"
  [ -f "$ctx/entrypoint.sh" ] && echo "COPY entrypoint.sh /opt/strix/entrypoint.sh"
  echo "LABEL org.opencontainers.image.description=\"test overlay on $BASE\""
} > "$ctx/Dockerfile"
echo "--- overlay on $BASE (strix-vllm ${vllm_ref:0:10}, libstrix ${strix_ref:0:10}):"
(cd "$ctx" && find . -type f ! -name Dockerfile | sort | sed 's/^/    /')
docker build -q -t "$TAG" "$ctx" > /dev/null
echo "built $TAG"
