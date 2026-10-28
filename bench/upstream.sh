#!/usr/bin/env bash
# Runs a command against a stock upstream vLLM Python tree instead of the
# installed fork: VLLM_UPSTREAM is a checkout of the fork's upstream base with
# the fork's compiled vllm/*.so copied in (same C++ ops). Upstream detects
# ROCm through amdsmi, which the ROCm SDK wheel ships but does not install.
# Give the checkout its own vllm/_version.py: the version is part of the
# torch.compile cache key, and a copied fork version shares the fork cache.
set -euo pipefail
: "${VLLM_UPSTREAM:=/home/ubuntu/local-llm/worktrees/vllm-upstream-strix}"
sdk=$(python -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
export PYTHONPATH="$VLLM_UPSTREAM:$sdk/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
