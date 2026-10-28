# SPDX-License-Identifier: MIT
"""Prefill and decode attention (bf16 or int8 KV) inside TRITON_ATTN.

The plugin overrides the TRITON_ATTN backend with a subclass whose impl runs
strix.attn_prefill (prefill-heavy and speculative-verify batches, the latter
split-KV) or strix.attn_decode (one query per sequence on the Gemma-4 heads,
where Triton decode reached about half the KV bandwidth), and the parent's
Triton kernel for everything else (verify batches on windowed layers, other
heads), so no --attention-backend change is needed. The (head_dim, GQA) pairs
each kernel takes are read from the kernel table, not repeated here.

int8 KV (vLLM's int8_per_token_head, the served teacher-corpus config) runs
the prefill kernel on the byte cache: the parent's
`_pth_key_value_caches`/`_k_scale_cache` views are handed straight to the
op, which widens the bytes to bf16 in the load stage and folds the per-key
scale into the score (K) and the P fragment (V).
"""

import torch

from strix_vllm import disabled

# Which (head_dim, GQA) pairs each kernel takes comes from the registry table
# (python/strix/registry.py), so the gate here cannot drift from the geometry
# the HIP dispatch accepts. Resolved once, on first use.
#   prefill: head 64 G4 (LFM2.5), 256 GQA 2/6/8, 512 G8
#   decode:  head 256 GQA 2 (Gemma-4 sliding), 256 G8 (35B, <= 2 sequences),
#            512 G8 (Gemma-4 full layers)
#   verify:  the split-KV prefill variant, head 256 GQA 6 only
_CFGS: dict[str, frozenset[tuple[int, int]]] = {}

# Batches with max_query_len >= PREFILL_MIN_QUERY run the prefill kernel.
# Speculative-verify batches (2..VERIFY_MAX_QUERY query tokens per sequence)
# run it split-KV, but only for GQA 6: 1.2x Triton at 16k context on 27B
# shapes, while the G=8 variant (16-key steps) is slower than Triton there.
# Single-token decode stays on Triton, which is already at bandwidth.
PREFILL_MIN_QUERY = 64
VERIFY_MAX_QUERY = 16
# Decode geometries that win only at one or two sequences: head 256 GQA 8
# (the 35B) is 2.0x the vendored Triton kernel at x1 from 64k on int8 KV and
# 1.43x at 128k on bf16, but 0.88-0.90x at x4/x16, so larger
# decode batches on it stay on Triton, whatever the KV dtype.
DECODE_MAX_SEQS = {(256, 8): 2}


def _cfgs(kind: str) -> frozenset[tuple[int, int]]:
    """(head_dim, GQA) pairs for `kind`, from the kernel table."""
    if kind not in _CFGS:
        import strix

        op = "attn_decode" if kind.startswith("decode") else "attn_prefill"
        where = {
            "verify": {"split_kv": True},
            "decode_bf16": {"kv_dtype": "bf16"},
            "decode_int8": {"kv_dtype": "int8"},
        }.get(kind, {})
        _CFGS[kind] = frozenset(strix.geometries(op, "head_dim", "gqa", **where))
    return _CFGS[kind]


def decode_splits(keys: int, blocks: int) -> int:
    """About one split-KV segment per 256 visible keys, capped so the grid
    (`blocks` = sequences x KV heads, times the splits) stays near 480
    blocks, at most 60.

    Decode CUDA graphs are captured with max_seq_len = max_model_len, so
    in practice this is the cap. The old fixed cap of 16 was a power of two,
    and power-of-two splits of power-of-two contexts measured ~25 % slower
    on gfx1151 (D=512, 128k: 16 splits 145 GB/s, 28/44/60 ~198 GB/s);
    480 // blocks gives 60/30/15/7 at 8/16/32/64 blocks.
    """
    return max(1, min(60, 480 // max(1, blocks), (keys + 255) // 256))


def num_splits(max_seq_len: int) -> int:
    """About one split-KV segment per 1k keys, at most 32."""
    return max(1, min(32, (max_seq_len + 1023) // 1024))


def register() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    if not disabled("attn_decode"):
        # Batches the strix kernels do not take (decode, quantized KV, ...)
        # run the fork's gfx1151-tuned Triton kernel; TRITON_ATTN looks the
        # function up in its module at call time.
        from vllm.v1.attention.backends import triton_attn

        from strix_vllm import triton_unified_attention

        triton_attn.unified_attention = triton_unified_attention.unified_attention

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
            cfg = (self.head_size, self.num_queries_per_kv)
            # Which kernel a batch gets depends on max_query_len, so both
            # halves of the gate are resolved per (head_dim, GQA) here.
            self._prefill_ok = cfg in _cfgs("prefill")
            # STRIX_DISABLE=attn_decode keeps the vendored Triton kernel and
            # must keep this one switched off with it.
            self._decode_ok = not disabled("attn_decode") and cfg in _cfgs("decode")
            self._decode_bf16 = cfg in _cfgs("decode_bf16")
            self._decode_int8 = cfg in _cfgs("decode_int8")
            self._decode_max_seqs = DECODE_MAX_SEQS.get(cfg)
            self._verify_ok = cfg in _cfgs("verify")
            # int8_per_token_head KV (the served teacher-corpus config) runs
            # the prefill kernel on the byte cache and its inline scales.
            self._kv_int8_ok = self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
            self._strix_ok = (
                self.attn_type == AttentionType.DECODER
                and (self._kv_quant_mode == KVQuantMode.NONE or self._kv_int8_ok)
                and (self._prefill_ok or self._decode_ok)
                and self.alibi_slopes is None
                # No window, or vLLM's decoder window (window - 1, 0).
                and (
                    tuple(self.sliding_window) == (-1, -1)
                    or (self.sliding_window[0] >= 0 and self.sliding_window[1] == 0)
                )
                and not self.logits_soft_cap
                and self.sinks is None
                and self.chunk_lookback == -1
            )

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
            if (
                not self._strix_ok
                or attn_metadata is None
                or output_scale is not None
                or output_block_scale is not None
                or attn_metadata.causal is not True
                or attn_metadata.mm_prefix_range_tensor is not None
                or attn_metadata.rswa_prefix_lens is not None
                or query.dtype != torch.bfloat16
                or kv_cache.dtype not in (torch.bfloat16, torch.int8)
            ):
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
            max_q = attn_metadata.max_query_len
            window_left = self.sliding_window[0]
            # Decode on int8 KV too: at long context decode is KV-bound, and
            # the packed cache's byte and scale views already exist.
            if (
                max_q == 1
                and self._decode_ok
                and (
                    (kv_cache.dtype == torch.bfloat16 and self._decode_bf16)
                    or (kv_cache.dtype == torch.int8 and self._decode_int8)
                )
                and (
                    self._decode_max_seqs is None
                    or attn_metadata.num_actual_tokens <= self._decode_max_seqs
                )
            ):
                import strix

                n = attn_metadata.num_actual_tokens
                if kv_cache.dtype == torch.bfloat16:
                    key_cache, value_cache = kv_cache.transpose(1, 2).split(
                        self.head_size, dim=-1
                    )
                    k_scale = v_scale = None
                else:
                    key_cache, value_cache = self._pth_key_value_caches(kv_cache)
                    k_scale, v_scale = self._k_scale_cache, self._v_scale_cache
                seen = attn_metadata.max_seq_len
                if window_left >= 0:
                    seen = min(seen, window_left + 1)
                strix.attn_decode(
                    output[:n].view(n, self.num_heads, self.head_size),
                    query[:n],
                    key_cache,
                    value_cache,
                    attn_metadata.seq_lens[:n],
                    attn_metadata.block_table[:n],
                    self.scale,
                    decode_splits(seen, n * self.num_kv_heads),
                    window_left,
                    k_scale,
                    v_scale,
                )
                return output
            # Windowed layers (Gemma-4 sliding, head 256 GQA 2) prefill here
            # too since the (256, 2, 2, 16, fp16) tile: ~20 TF, 1.3-1.4x the
            # vendored Triton kernel at window 1024. Verify batches on them
            # stay on Triton (the split-KV verify row is GQA 6 only).
            if max_q >= PREFILL_MIN_QUERY:
                splits = 1
            elif 2 <= max_q <= VERIFY_MAX_QUERY and self._verify_ok:
                splits = num_splits(attn_metadata.max_seq_len)
            else:
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
                key_cache, value_cache = self._pth_key_value_caches(kv_cache)
                k_scale, v_scale = self._k_scale_cache, self._v_scale_cache
            strix.attn_prefill(
                output[:n].view(n, self.num_heads, self.head_size),
                query[:n],
                key_cache,
                value_cache,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.block_table,
                self.scale,
                splits,
                window_left,
                k_scale,
                v_scale,
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
