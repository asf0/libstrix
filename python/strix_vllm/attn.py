# SPDX-License-Identifier: MIT
"""Prefill, decode and verify attention (bf16 or int8 KV) inside TRITON_ATTN.

The plugin overrides the TRITON_ATTN backend with a subclass, so no
--attention-backend change is needed. `route` picks the kernel of a batch:

  prefill  strix.attn_prefill, 16-query tiles
  verify   strix.attn_prefill split-KV, for speculative-verify batches
  decode   strix.attn_decode, one query per sequence, split-KV
  triton   the parent's kernel, for every other batch

The (head_dim, GQA, sinks, window, KV dtype) each kernel takes come from the
kernel table (python/strix/registry.py) and are not repeated here, so the gate
cannot drift from the geometries the HIP dispatch accepts.

On int8 KV (vLLM's int8_per_token_head) both ops read the byte cache: the
parent's `_pth_key_value_caches`/`_k_scale_cache` views are handed straight to
the op, which folds the per-key scales into the scores (K) and the softmax
weights (V).
"""

from typing import NamedTuple

import torch

from strix_vllm import disabled

# Batches with max_query_len >= PREFILL_MIN_QUERY run the prefill kernel,
# batches with 2..VERIFY_MAX_QUERY queries (speculative verify: one 16-query
# tile per sequence) the split-KV variant where the table has a split_kv row.
PREFILL_MIN_QUERY = 64
VERIFY_MAX_QUERY = 16

# Split-KV grids. Blocks of one launch read equal shares of the keys, so a
# count just past what the GPU holds at once leaves a short tail round: the
# grid (sequences x KV heads x splits) is sized to fit, per geometry.
#
# Decode: the grid's blocks, by (head_dim, GQA, KV dtype); 80 elsewhere.
DECODE_BLOCKS = {
    (256, 8, "bf16"): 40,
    (256, 8, "int8"): 40,
    (256, 6, "int8"): 40,
    (512, 8, "bf16"): 40,
    (512, 8, "int8"): 40,
    (128, 16, "bf16"): 100,
    (64, 4, "int8"): 160,
}
DECODE_MAX_SPLITS = 30
# Fewer than DECODE_MAX_SPLITS where more measured slower (int8 head 64).
DECODE_SPLIT_CAP = {(64, 4, "int8"): 10}
DECODE_SPLIT_KEYS = 96  # keys per split at least

# Verify: the splits at up to so many sequences x KV heads, by (head_dim, GQA,
# window); None is every larger batch. A geometry not listed uses `num_splits`.
VERIFY_SPLITS = {
    (64, 4, False): ((8, 15), (32, 20), (64, 15), (None, 5)),
    (128, 4, True): ((8, 7), (16, 5), (None, 3)),
    (128, 8, False): ((4, 20), (16, 15), (None, 10)),
    (128, 8, True): ((16, 7), (None, 5)),
    (128, 16, False): ((2, 15), (None, 5)),
    (128, 16, True): ((2, 15), (8, 5), (None, 2)),
    (256, 2, True): ((8, 5), (16, 3), (None, 2)),
    (256, 8, False): ((2, 30), (8, 15), (None, 7)),
    (512, 8, False): ((4, 10), (16, 5), (None, 3)),
}
VERIFY_SPLIT_KEYS = 64  # keys per split at least


class Kernels(NamedTuple):
    """The strix kernels one attention layer may run."""

    head_dim: int
    gqa: int
    window: bool
    prefill: bool
    verify: frozenset[str]  # KV dtypes ("bf16", "int8")
    decode: frozenset[str]


def layer_kernels(head_dim: int, gqa: int, sinks: bool, window: bool) -> Kernels:
    """The kernel table rows a layer of this geometry matches."""
    import strix

    def built(op: str, **where) -> bool:
        return (head_dim, gqa) in strix.geometries(
            op, "head_dim", "gqa", sinks=sinks, window=window, **where
        )

    # STRIX_DISABLE=attn_decode keeps the vendored Triton kernel out and must
    # keep this one off with it.
    kvs = ("bf16", "int8")
    decode = () if disabled("attn_decode") else kvs
    return Kernels(
        head_dim=head_dim,
        gqa=gqa,
        window=window,
        prefill=built("attn_prefill"),
        verify=frozenset(
            kv for kv in kvs if built("attn_prefill", split_kv=True, kv_dtype=kv)
        ),
        decode=frozenset(kv for kv in decode if built("attn_decode", kv_dtype=kv)),
    )


def route(
    kernels: Kernels,
    max_query_len: int,
    num_seqs: int,
    kv_dtype: str,
    max_seq_len: int,
    window_left: int,
    num_kv_heads: int,
    causal: bool = True,
) -> tuple[str, int]:
    """The kernel a batch runs ("decode", "prefill", "verify" or "triton")
    and its split-KV count (0 for Triton, which picks its own)."""
    blocks = num_seqs * num_kv_heads
    if not causal:
        # Draft blocks (DFlash / DSpark): the verify kernel with the keys after
        # the query visible too; no single-query decode kernel.
        if 2 <= max_query_len <= VERIFY_MAX_QUERY and kv_dtype in kernels.verify:
            return "verify", verify_splits(
                kernels, visible_keys(max_seq_len, window_left, max_query_len), blocks, max_seq_len
            )
        return "triton", 0
    keys = visible_keys(max_seq_len, window_left, max(1, max_query_len))
    if max_query_len == 1 and kv_dtype in kernels.decode:
        return "decode", decode_splits(kernels, kv_dtype, keys, blocks)
    if max_query_len >= PREFILL_MIN_QUERY and kernels.prefill:
        return "prefill", 1
    if 2 <= max_query_len <= VERIFY_MAX_QUERY and kv_dtype in kernels.verify:
        return "verify", verify_splits(kernels, keys, blocks, max_seq_len)
    return "triton", 0


def visible_keys(max_seq_len: int, window_left: int, queries: int = 1) -> int:
    """Keys that `queries` consecutive query tokens see between them."""
    if window_left < 0:
        return max_seq_len
    return min(max_seq_len, window_left + queries)


def non_causal_window(window_left: int) -> int:
    """attn_prefill's window argument of a non-causal (draft) batch."""
    return -2 if window_left < 0 else -3 - window_left


def decode_splits(kernels: Kernels, kv_dtype: str, keys: int, blocks: int) -> int:
    """Split count of a decode batch of `blocks` = sequences x KV heads.

    Decode CUDA graphs are captured with max_seq_len = max_model_len, so in
    practice `keys` matters on windowed layers only.
    """
    key = (kernels.head_dim, kernels.gqa, kv_dtype)
    splits = min(
        DECODE_SPLIT_CAP.get(key, DECODE_MAX_SPLITS),
        DECODE_BLOCKS.get(key, 80) // max(1, blocks),
    )
    if kv_dtype == "int8" and kernels.head_dim >= 128 and not kernels.window and splits < 3:
        # Too many blocks for the grid: many short segments measured faster
        # than one or two long ones.
        splits = 7 if kernels.head_dim >= 256 else 10
    return max(1, min(splits, -(-keys // DECODE_SPLIT_KEYS)))


def num_splits(max_seq_len: int) -> int:
    """About one split-KV segment per 1k keys, at most 32."""
    return max(1, min(32, (max_seq_len + 1023) // 1024))


def verify_splits(kernels: Kernels, keys: int, blocks: int, max_seq_len: int) -> int:
    """Split count of a verify batch of `blocks` = sequences x KV heads."""
    steps = VERIFY_SPLITS.get((kernels.head_dim, kernels.gqa, kernels.window))
    if steps is None:
        return num_splits(max_seq_len)
    splits = next(n for most, n in steps if most is None or blocks <= most)
    return max(1, min(splits, -(-keys // VERIFY_SPLIT_KEYS)))


def _extensible_kv_cache_off() -> None:
    """Default vLLM's extensible KV cache off (an explicit setting stands).

    The gfx1151 Triton kernel splits a KV view past 2 GiB into up to four
    pieces for 32-bit addressing and passes each piece's base pointer. The
    extensible cache maps its pages only after warmup, so during warmup a
    piece's base is unmapped and Triton's launcher refuses it ("Pointer
    argument ... cannot be accessed from Triton"). Sizing the cache the
    classic way plans as much KV here (Qwen3.8-Flash-Next, 0.85: 1,110,621
    tokens against 1,057,882).
    """
    from vllm.config import VllmConfig

    orig = getattr(VllmConfig, "_resolve_extensible_kv_cache", None)
    if orig is None:
        return
    orig = getattr(orig, "strix_orig", orig)  # wrap once

    def _resolve_extensible_kv_cache(self) -> None:
        cache = self.cache_config
        if (
            not cache._extensible_kv_cache_resolved
            and cache.enable_extensible_kv_cache is None
        ):
            cache.enable_extensible_kv_cache = False
        orig(self)

    _resolve_extensible_kv_cache.strix_orig = orig
    VllmConfig._resolve_extensible_kv_cache = _resolve_extensible_kv_cache


def register() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    if not disabled("attn_decode"):
        # Batches the strix kernels do not take run the fork's gfx1151-tuned
        # Triton kernel; TRITON_ATTN looks the function up in its module at
        # call time.
        from vllm.v1.attention.backends import triton_attn

        from strix_vllm import triton_unified_attention

        triton_attn.unified_attention = triton_unified_attention.unified_attention
        _extensible_kv_cache_off()

    if disabled("attn_prefill"):
        return
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        "strix_vllm.attn.StrixTritonAttentionBackend",
    )


def _classes():
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
        TritonAttentionMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    class StrixTritonAttentionImpl(TritonAttentionImpl):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Which kernel a batch gets depends on the batch (`route`); what
            # this layer's geometry allows is resolved once.
            self._kernels = layer_kernels(
                self.head_size,
                self.num_queries_per_kv,
                sinks=self.sinks is not None,
                window=self.sliding_window[0] >= 0,
            )
            self._strix_ok = (
                self.attn_type == AttentionType.DECODER
                and self._kv_quant_mode
                in (KVQuantMode.NONE, KVQuantMode.INT8_PER_TOKEN_HEAD)
                and (self._kernels.prefill or bool(self._kernels.decode))
                and self.alibi_slopes is None
                # No window, or vLLM's decoder window (window - 1, 0).
                and (
                    tuple(self.sliding_window) == (-1, -1)
                    or (self.sliding_window[0] >= 0 and self.sliding_window[1] == 0)
                )
                and not self.logits_soft_cap
                and self.chunk_lookback == -1
            )

        def _int8_kv_views(self, kv_cache):
            """Byte and fp32 scale views of the packed int8 cache.

            The parent carves the scale views once, on the first call. An
            extensible KV cache backs only part of the cache during warmup
            and rebuilds its views at the final size afterwards. The parent's
            kernel reads only pointers and strides, but ours checks the
            shape, so the views are carved again when the block count or the
            storage changes.
            """
            ks = self._k_scale_cache
            if ks is not None and (
                ks.shape[0] != kv_cache.shape[0]
                or ks.untyped_storage().data_ptr()
                != kv_cache.untyped_storage().data_ptr()
            ):
                self._k_scale_cache = self._v_scale_cache = None
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            return key_cache, value_cache, self._k_scale_cache, self._v_scale_cache

        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            kernel, splits = "triton", 0
            window_left = self.sliding_window[0]
            if (
                self._strix_ok
                and attn_metadata is not None
                and output_scale is None
                and output_block_scale is None
                and isinstance(attn_metadata.causal, bool)
                and attn_metadata.mm_prefix_range_tensor is None
                and attn_metadata.rswa_prefix_lens is None
                and query.dtype == torch.bfloat16
                and kv_cache.dtype in (torch.bfloat16, torch.int8)
            ):
                kernel, splits = route(
                    self._kernels,
                    attn_metadata.max_query_len,
                    attn_metadata.query_start_loc.shape[0] - 1,
                    "bf16" if kv_cache.dtype == torch.bfloat16 else "int8",
                    attn_metadata.max_seq_len,
                    window_left,
                    self.num_kv_heads,
                    causal=attn_metadata.causal,
                )
            if kernel == "triton":
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            import strix

            n = attn_metadata.num_actual_tokens
            if kv_cache.dtype == torch.bfloat16:
                # Same cache views as the parent's unquantized path.
                key_cache, value_cache = kv_cache.transpose(1, 2).split(
                    self.head_size, dim=-1
                )
                k_scale = v_scale = None
            else:
                # Packed int8 cache: [K | K_scale | V | V_scale] per slot, and
                # the parent already carves the byte and fp32 scale views for
                # its own kernel (it also writes them on cache update).
                key_cache, value_cache, k_scale, v_scale = self._int8_kv_views(kv_cache)
            out = output[:n].view(n, self.num_heads, self.head_size)
            if kernel == "decode":
                strix.attn_decode(
                    out,
                    query[:n],
                    key_cache,
                    value_cache,
                    attn_metadata.seq_lens[:n],
                    attn_metadata.block_table[:n],
                    self.scale,
                    splits,
                    window_left,
                    k_scale,
                    v_scale,
                    self.sinks,
                )
            else:
                strix.attn_prefill(
                    out,
                    query[:n],
                    key_cache,
                    value_cache,
                    attn_metadata.query_start_loc,
                    attn_metadata.seq_lens,
                    attn_metadata.block_table,
                    self.scale,
                    splits,
                    window_left if attn_metadata.causal else non_causal_window(window_left),
                    k_scale,
                    v_scale,
                    self.sinks,
                )
            return output

    class StrixTritonAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
        # Past seq_threshold_3D (the CUDA-graph capture size nearest
        # 128 / kv_heads), decode and verify batches fall to the 2D launch,
        # which is ~2x slower per layer on gfx1151 (151 -> 105 tok/s at 33
        # sequences). Keep every schedulable batch on split-KV, as the fork
        # does, and grow the split-KV scratch to match.
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            old = self.seq_threshold_3D
            if disabled("attn_decode") or old >= max_seqs or self.max_seqs_64_segments:
                return
            rows = self.softmax_segm_output.size(0) // min(old, max_seqs) * max_seqs
            self.seq_threshold_3D = max_seqs
            for name in (
                "softmax_segm_output",
                "softmax_segm_max",
                "softmax_segm_expsum",
            ):
                t = getattr(self, name)
                setattr(self, name, t.new_empty((rows, *t.shape[1:])))

    class StrixTritonAttentionBackend(TritonAttentionBackend):
        @staticmethod
        def get_impl_cls() -> type[TritonAttentionImpl]:
            return StrixTritonAttentionImpl

        @staticmethod
        def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
            return StrixTritonAttentionMetadataBuilder

    return StrixTritonAttentionBackend, StrixTritonAttentionImpl


def __getattr__(name: str):
    # The registry imports the backend class by path, lazily, after vLLM's
    # attention modules are importable.
    if name in ("StrixTritonAttentionBackend", "StrixTritonAttentionImpl"):
        backend, impl = _classes()
        globals().update(
            StrixTritonAttentionBackend=backend, StrixTritonAttentionImpl=impl
        )
        return globals()[name]
    raise AttributeError(name)
