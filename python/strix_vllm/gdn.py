# SPDX-License-Identifier: MIT
"""GDN prefill on gfx1151: chunk_gated_delta_rule, plus the causal conv and
post-conv prep in front of it.

The conv and post-conv are plain functions that qwen_gdn_linear_attn calls
from inside its custom op (no CustomOp hook), so the plugin rebinds those
module globals to wrappers that fall back to the originals off-geometry.
"""

import torch

from strix_vllm import disabled


def _log_once(msg: str, *args) -> None:
    from vllm.logger import init_logger

    init_logger(f"vllm.{__name__}").info_once(msg, *args)


def _aligned_rows(t: torch.Tensor) -> bool:
    return (
        t.dtype == torch.bfloat16
        and t.dim() == 2
        and t.stride(1) == 1
        and t.stride(0) % 8 == 0
        and t.data_ptr() % 16 == 0
    )


def _wrap_conv(orig):
    def causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=None,
        has_initial_state=None,
        activation="silu",
        pad_slot_id=-1,
        null_block_id=0,
        metadata=None,
        validate_data=False,
    ):
        silu = activation is True or activation in ("silu", "swish")
        supported = (
            (silu or activation is None)
            and x.dim() == 2
            and _aligned_rows(x.t())
            and x.size(0) % 8 == 0
            and weight.dtype == torch.bfloat16
            and weight.is_contiguous()
            and weight.shape == (x.size(0), 4)
            and (
                bias is None or (bias.dtype == torch.bfloat16 and bias.is_contiguous())
            )
            and conv_states is not None
            and conv_states.dtype == torch.bfloat16
            and cache_indices is not None
            and cache_indices.dtype == torch.int32
            and cache_indices.dim() == 1
            and query_start_loc.dtype == torch.int32
            and (has_initial_state is None or has_initial_state.dtype == torch.bool)
            and pad_slot_id == -1
            and null_block_id == 0
        )
        _log_once(
            "libstrix: causal_conv1d_fn -> %s",
            "strix"
            if supported
            else f"vLLM (x {tuple(x.shape)} {x.stride()} {x.dtype} "
            f"ptr%16={x.data_ptr() % 16}, idx {cache_indices.stride() if cache_indices is not None else None}, "
            f"states {conv_states.dtype if conv_states is not None else None}, "
            f"activation {activation})",
        )
        if not supported:
            return orig(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
                activation,
                pad_slot_id,
                null_block_id,
                metadata,
                validate_data,
            )
        import strix

        cache_indices = cache_indices.contiguous()
        if has_initial_state is not None:
            has_initial_state = has_initial_state.contiguous()
        return strix.gdn_causal_conv1d(
            x.t(),
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu,
        ).t()

    return causal_conv1d_fn


def _wrap_post_conv(orig):
    def fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    ):
        supported = (
            apply_l2norm
            and not output_g_exp
            and head_k_dim == 128
            and head_v_dim == 128
            and _aligned_rows(conv_output)
            and conv_output.size(1) == (2 * num_k_heads + A_log.numel()) * 128
            and all(
                t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(1) == 1
                for t in (a, b)
            )
        )
        _log_once(
            "libstrix: fused_post_conv_prep -> %s",
            "strix"
            if supported
            else f"vLLM (conv {tuple(conv_output.shape)} {conv_output.stride()}, "
            f"a {a.dtype} {a.stride()}, l2norm {apply_l2norm}, g_exp {output_g_exp})",
        )
        if not supported:
            return orig(
                conv_output,
                a,
                b,
                A_log,
                dt_bias,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                apply_l2norm,
                output_g_exp,
            )
        import strix

        return strix.gdn_post_conv(conv_output, a, b, A_log, dt_bias, num_k_heads)

    return fused_post_conv_prep


_MTP_OP = "fused_gdn_decode_post_conv_mtp"
_MTP_SCHEMA = (
    "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
    "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
    "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
    "float scale, float norm_eps=1e-5, str output_gate_activation='silu') -> ()"
)
_mtp_lib = None


def _register_mtp_decode() -> None:
    """Expose strix's port of vLLM's fused GDN MTP decode kernel under the name
    QwenGatedDeltaNetAttention probes (CUDA-only in vLLM's build), so spec
    decoding skips the Triton fallback."""
    global _mtp_lib
    if hasattr(torch.ops._C, _MTP_OP):
        return
    _mtp_lib = torch.library.Library("_C", "FRAGMENT")
    _mtp_lib.define(f"{_MTP_OP}({_MTP_SCHEMA}")

    def impl(*args, **kwargs):
        torch.ops.strix.gdn_decode_mtp_bf16(*args, **kwargs)

    _mtp_lib.impl(_MTP_OP, impl, "CUDA")
    _mtp_lib._register_fake(_MTP_OP, lambda *args, **kwargs: None)


def register() -> None:
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer

    if not disabled("gdn_mtp"):
        import vllm._custom_ops  # noqa: F401  (defines torch.ops._C first)

        _register_mtp_decode()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    if not disabled("gdn_conv"):
        gdn_layer.causal_conv1d_fn = _wrap_conv(gdn_layer.causal_conv1d_fn)
        gdn_layer.fused_post_conv_prep = _wrap_post_conv(gdn_layer.fused_post_conv_prep)

    if disabled("gdn_chunk"):
        return

    @ChunkGatedDeltaRule.register_oot
    class StrixChunkGatedDeltaRule(ChunkGatedDeltaRule):
        # ChunkGatedDeltaRule picks its forward in __init__ rather than through
        # CustomOp dispatch; take over only where it chose Triton/FLA.
        def __init__(self) -> None:
            super().__init__()
            if self._forward_method == self.forward_native:
                self._forward_method = self.forward_strix

        def forward_strix(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.Tensor | None = None,
            chunk_indices: torch.Tensor | None = None,
            chunk_offsets: torch.Tensor | None = None,
            use_qk_l2norm_in_kernel: bool = True,
            core_attn_out: torch.Tensor | None = None,
            aiter_prefill_metadata: object | None = None,
        ):
            supported = (
                cu_seqlens is not None
                and not use_qk_l2norm_in_kernel
                and q.dim() == 4
                and q.size(0) == 1
                and q.size(-1) == 128
                and v.size(-1) == 128
                and q.dtype == torch.bfloat16
                and v.size(2) % q.size(2) == 0
                and all(t.stride(-1) == 1 for t in (q, k, v))
            )
            if not supported:
                return self.forward_native(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    output_final_state,
                    cu_seqlens,
                    chunk_indices,
                    chunk_offsets,
                    use_qk_l2norm_in_kernel,
                    core_attn_out,
                    aiter_prefill_metadata,
                )
            import strix

            out = None
            if core_attn_out is not None:
                out = core_attn_out.reshape(-1)[: v.numel()].view(v.shape)[0]
            o, final_state = strix.gdn_chunk_fwd(
                q[0], k[0], v[0], g[0], beta[0], initial_state, cu_seqlens, out=out
            )
            return o[None], final_state
