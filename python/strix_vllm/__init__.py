# SPDX-License-Identifier: MIT
"""vLLM general plugin: swaps in libstrix kernels on gfx1151.

Loaded through the `vllm.general_plugins` entry point in every vLLM process,
possibly more than once. Set STRIX_DISABLE=all (or a comma list of op names,
e.g. rms_norm_gated) to keep vLLM's own kernels.
"""

import hashlib
import os
from pathlib import Path

_registered = False
# vLLM env factor that ties compile caches to the plugin (see _cache_factor).
CACHE_FACTOR = "STRIX_VLLM_CACHE_KEY"


def disabled(op: str) -> bool:
    names = {s.strip() for s in os.environ.get("STRIX_DISABLE", "").split(",")}
    return "all" in names or op in names


def _cache_factor() -> str:
    """STRIX_DISABLE plus a digest of the plugin's Python source.

    Some swaps (OOT norm, W8A16 apply, GEMM wrapper) live in code that
    torch.compile traces, and vLLM keys its compile caches on its own env vars
    and config only, so without this a graph traced with the plugin off (or an
    older plugin) is replayed with it on.
    """
    h = hashlib.sha256(os.environ.get("STRIX_DISABLE", "").encode())
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    from vllm import envs

    key = _cache_factor()
    envs.environment_variables.setdefault(CACHE_FACTOR, lambda: key)

    from vllm.logger import init_logger
    from vllm.platforms import current_platform

    logger = init_logger(f"vllm.{__name__}")
    if not current_platform.is_rocm():
        return
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        logger.info("libstrix: not gfx1151, plugin inactive")
        return
    if disabled("all"):
        logger.info("libstrix: disabled by STRIX_DISABLE=all")
        return

    # Loads torch.ops.strix in this process: several swaps call the ops
    # directly, and models without GDN layers import nothing else from strix.
    import strix  # noqa: F401

    from strix_vllm import attn, gdn, gemm, int8, linear, moe, mxfp4, norm

    int8.register()
    linear.register()
    mxfp4.register()
    gemm.register()
    norm.register()
    moe.register()
    gdn.register()
    attn.register()
    logger.info("libstrix: plugin registered")
