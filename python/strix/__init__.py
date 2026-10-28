# SPDX-License-Identifier: MIT
"""gfx1151 (Strix Halo) HIP kernels, exposed as torch.ops.strix.*."""

from pathlib import Path

import torch

from strix.registry import (
    Range,
    explain,
    geometries,
    kernels,
    ops,
    select,
    selections,
)

torch.ops.load_library(str(Path(__file__).with_name("_C.so")))


@torch.library.register_fake("strix::gdn_gated_rmsnorm_h128_bf16")
def _(x, z, weight, eps, activation):
    return x.new_empty(x.shape)


@torch.library.register_fake("strix::moe_gemm_bf16")
def _(
    c,
    a,
    b,
    sorted_ids,
    expert_ids,
    num_post_padded,
    topk_weights,
    top_k,
    block_m,
    block_n,
    silu_mul=False,
    activation="silu",
):
    return None


@torch.library.register_fake("strix::moe_gemm_w4a16")
def _(
    c,
    a,
    b,
    scales,
    sorted_ids,
    expert_ids,
    num_post_padded,
    topk_weights,
    top_k,
    block_m,
    block_n,
    silu_mul=False,
    activation="silu",
    group_size=32,
):
    return None


@torch.library.register_fake("strix::moe_gemm_w8a16")
def _(
    c,
    a,
    b,
    scales,
    sorted_ids,
    expert_ids,
    num_post_padded,
    topk_weights,
    top_k,
    block_m,
    block_n,
    silu_mul=False,
    activation="silu",
):
    return None


@torch.library.register_fake("strix::gdn_chunk_fwd_k128_v128_c64_bf16")
def _(
    o,
    final_state,
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    chunk_offsets,
    max_chunks,
    scale,
):
    return None


def gdn_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule over varlen sequences (no host sync).

    q, k: [T, Hk, 128]; v: [T, Hv, 128] bf16; g (log gate), beta: [T, Hv];
    initial_state: [N, Hv, 128 (V), 128 (K)] fp32 or bf16, whichever
    --mamba-ssm-cache-dtype selects; cu_seqlens: [N + 1].
    Returns (o [T, Hv, 128] bf16, final_state [N, Hv, 128, 128] in the state
    dtype).
    """
    cu = cu_seqlens.to(torch.int32)
    chunks = (cu[1:] - cu[:-1] + 63) // 64
    offsets = torch.zeros_like(cu)
    torch.cumsum(chunks, 0, out=offsets[1:])
    max_chunks = (q.size(0) + 63) // 64 + cu.numel() - 1
    o = torch.empty_like(v) if out is None else out
    final_state = torch.empty(
        initial_state.shape, device=v.device, dtype=initial_state.dtype
    )
    if workspace is None:
        shape = (max_chunks, v.size(1), 64, 64)
        workspace = (
            v.new_empty(shape),
            v.new_empty(shape),
            v.new_empty(shape[:3], dtype=torch.float32),
        )
    torch.ops.strix.gdn_chunk_fwd_k128_v128_c64_bf16(
        o,
        final_state,
        *workspace,
        q,
        k,
        v,
        g.float().contiguous(),
        beta.float().contiguous(),
        initial_state,
        cu,
        offsets,
        max_chunks,
        q.size(-1) ** -0.5 if scale is None else scale,
    )
    return o, final_state


@torch.library.register_fake("strix::attn_prefill_bf16")
def _(
    out,
    q,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    cu_seqlens_q,
    seq_lens,
    block_table,
    tile_offsets,
    max_tiles,
    scale,
    num_splits,
    window_left=-1,
):
    return None


def attn_prefill(
    out: torch.Tensor,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    num_splits: int = 1,
    window_left: int = -1,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal paged attention for prefill (no host sync).

    window_left >= 0 is vLLM's sliding window (window - 1, 0): the query at
    position p sees keys p - window_left .. p.

    num_splits > 1 splits each 16-query tile's keys (split-KV), for short
    queries (decode, speculative verify) over long contexts.

    q, out: [T, Hq, D]; caches: [blocks, block_size, Hkv, D] views, D = 64
    (GQA 4), 256 (GQA 2 / 6 / 8) or 512 (GQA 8); the queries of sequence n are
    the last q_len of its seq_lens[n] positions. The caches are bf16, or int8
    with k_scale/v_scale ([blocks, block_size, Hkv] fp32) as the dequant
    scales, one per (key, head) - vLLM's int8_per_token_head layout.
    """
    cu = cu_seqlens_q.to(torch.int32)
    tiles = (cu[1:] - cu[:-1] + 15) // 16
    offsets = torch.zeros_like(cu)
    torch.cumsum(tiles, 0, out=offsets[1:])
    max_tiles = (q.size(0) + 15) // 16 + cu.numel() - 1
    torch.ops.strix.attn_prefill_bf16(
        out,
        q,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        cu,
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        offsets,
        max_tiles,
        scale,
        num_splits,
        window_left,
    )
    return out


attn_prefill_h256 = attn_prefill


@torch.library.register_fake("strix::attn_prefill32_bf16")
def _(
    out,
    q,
    key_cache,
    value_cache,
    cu_seqlens_q,
    seq_lens,
    block_table,
    tile_offsets,
    max_tiles,
    scale,
):
    return None


def attn_prefill32(
    out: torch.Tensor,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Experimental D=512 prefill with 32 query columns per K/V stage.

    bf16 KV, causal, no window, single KV segment. Only the first tile of
    each 32-query group runs, so the tile grid is the same as attn_prefill's.
    """
    cu = cu_seqlens_q.to(torch.int32)
    tiles = (cu[1:] - cu[:-1] + 15) // 16
    offsets = torch.zeros_like(cu)
    torch.cumsum(tiles, 0, out=offsets[1:])
    max_tiles = (q.size(0) + 15) // 16 + cu.numel() - 1
    torch.ops.strix.attn_prefill32_bf16(
        out,
        q,
        key_cache,
        value_cache,
        cu,
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        offsets,
        max_tiles,
        scale,
    )
    return out


@torch.library.register_fake("strix::attn_decode_bf16")
def _(
    out,
    q,
    key_cache,
    value_cache,
    seq_lens,
    block_table,
    scale,
    num_splits,
    window_left=-1,
    k_scale=None,
    v_scale=None,
):
    return None


def attn_decode(
    out: torch.Tensor,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    num_splits: int = 1,
    window_left: int = -1,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged attention for one query token per sequence (decode), split-KV.

    q, out: [N, Hq, D]; caches: [blocks, block_size, Hkv, D] views; the query
    is position seq_lens[n] - 1. Built for (D, GQA) = (256, 2), (512, 8).
    The caches are bf16, or int8 with k_scale/v_scale
    ([blocks, block_size, Hkv] fp32) as the dequant scales, one per (key,
    head) - vLLM's int8_per_token_head layout.
    """
    torch.ops.strix.attn_decode_bf16(
        out,
        q,
        key_cache,
        value_cache,
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        scale,
        num_splits,
        window_left,
        k_scale,
        v_scale,
    )
    return out


def gdn_gated_rmsnorm_h128_bf16(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str = "silu",
) -> torch.Tensor:
    act = 1 if activation == "sigmoid" else 0
    return torch.ops.strix.gdn_gated_rmsnorm_h128_bf16(x, z, weight, eps, act)


@torch.library.register_fake("strix::gdn_causal_conv1d_w4_bf16")
def _(out, x, weight, bias, conv_states, cu_seqlens, cache_indices, has_init, silu):
    return None


@torch.library.register_fake("strix::gdn_post_conv_bf16")
def _(q, k, v, g, beta, conv, a, b, a_log, dt_bias, eps):
    return None


@torch.library.register_fake("strix::mxfp4_wmma_gemm")
def _(w, x, scales, bias):
    return x.new_empty((x.size(0), w.size(0)))


@torch.library.register_fake("strix::mxfp4_wmma_gemm_cfg")
def _(w, x, scales, bias, cfg):
    return x.new_empty((x.size(0), w.size(0)))


@torch.library.register_fake("strix::mxfp4_gemv")
def _(x, w, scales):
    return x.new_empty((x.size(0), w.size(0)))


@torch.library.register_fake("strix::mxfp4_dequant")
def _(w, scales):
    return w.new_empty((w.size(0), w.size(1) * 2), dtype=torch.bfloat16)


@torch.library.register_fake("strix::w4a16_dequant")
def _(w, scales, group_size):
    return w.new_empty((w.size(0), w.size(1) * 2), dtype=torch.bfloat16)


def gdn_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    silu: bool = True,
) -> torch.Tensor:
    """Width-4 causal depthwise conv over varlen x [T, dim]; returns [T, dim]
    and updates conv_states [lines, dim, 3] at cache_indices."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    torch.ops.strix.gdn_causal_conv1d_w4_bf16(
        out,
        x,
        weight,
        bias,
        conv_states,
        cu_seqlens,
        cache_indices,
        has_initial_state,
        silu,
    )
    return out


def gdn_post_conv(
    conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    """Splits conv [T, (2 hk + hv) * 128] into L2-normalized q, k and v, and
    computes g = -exp(a_log) softplus(a + dt_bias) and beta = sigmoid(b)."""
    t, hv = conv.size(0), a_log.numel()
    q = conv.new_empty(t, num_k_heads, 128)
    k = conv.new_empty(t, num_k_heads, 128)
    v = conv.new_empty(t, hv, 128)
    g = conv.new_empty(t, hv, dtype=torch.float32)
    beta = conv.new_empty(t, hv, dtype=torch.float32)
    torch.ops.strix.gdn_post_conv_bf16(
        q, k, v, g, beta, conv, a, b, a_log.float(), dt_bias.float(), eps
    )
    return q, k, v, g, beta


def is_supported_device() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"


__all__ = [
    "Range",
    "attn_decode",
    "attn_prefill",
    "attn_prefill_h256",
    "explain",
    "gdn_causal_conv1d",
    "gdn_chunk_fwd",
    "gdn_gated_rmsnorm_h128_bf16",
    "gdn_post_conv",
    "geometries",
    "is_supported_device",
    "kernels",
    "ops",
    "select",
    "selections",
]
