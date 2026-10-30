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


def _env_names(var: str) -> set[str]:
    return {s.strip() for s in os.environ.get(var, "").split(",") if s.strip()}


def disabled(op: str) -> bool:
    names = _env_names("STRIX_DISABLE")
    return "all" in names or op in names


# Swaps that were measured (speed and accuracy) on one model family only.
# They change numerics or memory, so elsewhere they stay off unless
# STRIX_ENABLE names them: tuning them for one model cannot change another.
ARCH_GATED = {
    # int8 decode copies of bf16 linears: +50 % of the dense weight memory.
    "dense_w8": ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"),
    # int8 activations in the int4 MoE GEMM.
    "moe_w4a8": ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"),
}


def _model_architectures() -> list[str]:
    """Architectures of the model being loaded; [] outside a vLLM config."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    model_config = getattr(config, "model_config", None)
    if model_config is None:
        return []
    archs = list(getattr(model_config.hf_config, "architectures", None) or [])
    try:
        archs.append(model_config.architecture)
    except Exception:
        pass
    return archs


def model_gate(op: str) -> bool:
    """Whether an ARCH_GATED swap applies to the model being loaded."""
    if disabled(op):
        return False
    if op in _env_names("STRIX_ENABLE"):
        return True
    archs = _model_architectures()
    on = any(a in ARCH_GATED[op] for a in archs)
    from vllm.logger import init_logger

    init_logger(f"vllm.{__name__}").info_once(
        "libstrix: %s %s for %s%s",
        op,
        "on" if on else "off",
        archs[0] if archs else "unknown model",
        "" if on else f" (not measured there; STRIX_ENABLE={op} to try it)",
    )
    return on


def _cache_factor() -> str:
    """STRIX_DISABLE, STRIX_ENABLE and a digest of the plugin's Python source.

    Some swaps (OOT norm, W8A16 apply, GEMM wrapper) live in code that
    torch.compile traces, and vLLM keys its compile caches on its own env vars
    and config only, so without this a graph traced with the plugin off (or an
    older plugin) is replayed with it on.
    """
    h = hashlib.sha256(os.environ.get("STRIX_DISABLE", "").encode())
    h.update(os.environ.get("STRIX_ENABLE", "").encode())
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

    from strix_vllm import attn, dense_w8, gdn, gemm, int8, linear, moe, mxfp4, norm, qwen4_exp

    int8.register()
    linear.register()
    mxfp4.register()
    gemm.register()
    norm.register()
    moe.register()
    gdn.register()
    attn.register()
    qwen4_exp.register()
    dense_w8.register()
    logger.info("libstrix: plugin registered")
