#!/usr/bin/env bash
# Usage: VLLM_UPSTREAM=/path/to/vllm bench/upstream.sh <cmd>  (see README)
set -euo pipefail
: "${VLLM_UPSTREAM:?set VLLM_UPSTREAM to the upstream vLLM checkout}"
sdk=$(python -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
export PYTHONPATH="$VLLM_UPSTREAM:$sdk/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
