#!/usr/bin/env bash
# Builds python/strix/_C.so for gfx1151. Source the vLLM env script first so
# python, hipcc and torch come from the venv the library will be loaded into.
set -euo pipefail
cd "$(dirname "$0")"
export TMPDIR="${TMPDIR:-$HOME/.cache/pip-tmp}"
mkdir -p "$TMPDIR"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="${ROCM_PATH:?source the ROCm env first}/lib/llvm/bin/clang++" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DSTRIX_GPU_ARCH="${STRIX_GPU_ARCH:-gfx1151}"
cmake --build build -j "${MAX_JOBS:-8}"
