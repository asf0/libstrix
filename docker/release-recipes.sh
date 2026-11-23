#!/usr/bin/env bash
# Packs the compose recipes of a release into libstrix-recipes-<version>.zip:
# every recipe but the bench-* ones, common.yaml with the image pinned to
# ghcr.io/asf0/strix-vllm:<version>, .env.example and a README. The image
# workflow attaches it to the GitHub release of a version tag.
#
#   docker/release-recipes.sh 0.1.0 [recipes dir] [output dir]
#
# PYTHON picks the interpreter that zips (default python3).
set -euo pipefail
version=${1:?usage: release-recipes.sh <version> [recipes dir] [output dir]}
src=${2:-docker/recipes}
out=${3:-.}
name=libstrix-recipes-$version
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
dst=$tmp/$name
mkdir -p "$dst"
(cd "$src" && find . -name '*.yaml' ! -name 'bench-*' -print) | while read -r f; do
  mkdir -p "$dst/$(dirname "$f")"
  cp "$src/$f" "$dst/$f"
done
cp "$src/.env.example" "$dst/.env.example"
sed -i "s#ghcr.io/asf0/strix-vllm:latest#ghcr.io/asf0/strix-vllm:$version#" "$dst/common.yaml"
grep -q "strix-vllm:$version" "$dst/common.yaml"
cat > "$dst/README.md" <<EOF
# libstrix $version recipes

Docker Compose recipes for vLLM on AMD Strix Halo (gfx1151), pinned to the
image \`ghcr.io/asf0/strix-vllm:$version\` (set \`IMAGE\` in \`.env\` to use
another one).

    cp .env.example .env          # set MODELS=/path/to/your/models
    docker compose -f qwen3.6-35b-a3b.yaml up -d

The \`npu/\` files are overlays for the XDNA2 NPU prefill path, layered over
the recipe of the same name. Host setup (GPU memory cap, BIOS), the recipe
table and every override: https://github.com/asf0/libstrix/blob/v$version/docs/docker.md
What changed in this release: https://github.com/asf0/libstrix/blob/v$version/CHANGELOG.md
EOF
mkdir -p "$out"
(cd "$tmp" && ${PYTHON:-python3} -m zipfile -c "$name.zip" "$name")
mv "$tmp/$name.zip" "$out/"
echo "$out/$name.zip"
