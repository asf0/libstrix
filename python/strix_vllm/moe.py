# SPDX-License-Identifier: MIT
"""Unquantized MoE on gfx1151.

moe_pad: expert weights get a padded row pitch. Packed rows of 2048 / 512 bf16
are 4 KiB / 1 KiB apart, so the rows a GEMM tile reads together camp on the
same DRAM channels (1.2-1.45x on Triton's fused_moe at prefill sizes). vLLM's
own VLLM_ROCM_MOE_PADDING copies the padded view back into the packed
storage (#46009), which undoes it; here each layer's experts are copied into
padded storage after loading, before vLLM sets up the kernel (and before
CUDA graphs capture the address). Padding at create_weights instead made
LFM2.5's real checkpoint load take 15+ GB more host memory.

moe_gemm: both expert GEMMs run torch.ops.strix.moe_gemm_bf16 (at bandwidth
up to ~1k tokens, ~30 TFLOPS at prefill sizes), with the gate nonlinearity
(SiLU, or GELU-tanh for gemma-4's `gelu_pytorch_tanh`) fused into the w13
epilogue; configurations it does not cover stay on vLLM's TritonExperts.

moe_w4a16: the same two GEMMs for compressed-tensors int4 experts
(torch.ops.strix.moe_gemm_w4a16). The quantized MoE method and its experts
class are not CustomOps, so register() rebinds the module global that vLLM's
WNA16 oracle resolves (`int_wna16.TritonWNA16Experts`); only the TRITON
backend (symmetric uint4, group 32/64/128, bf16 scales, no zero point) is
hooked.
"""

import os

import torch

from strix_vllm import disabled, model_gate

ROW_PAD = 64  # elements; (K + 64) * 2 bytes has a small power-of-two factor


def strix_blocks(num_tokens: int) -> tuple[int, int]:
    """(block_m, block_n) for both GEMMs, from bench/kernels/moe_gemm.py on
    Qwen3.6-35B-A3B shapes. Both GEMMs share one alignment, so one block_m."""
    if num_tokens <= 2:
        return 32, 64
    if num_tokens <= 1024:
        return 64, 128
    return 128, 128


# int4 experts, (w13 n, k) -> [(max tokens, block_m, block_n)], from
# bench/kernels/moe_wna16.py. Qwen3.8-Flash-Next (E=512, top-10, inter 640,
# group 128): 16x64 is ~2x the 32x64/64x128 tiles up to 256 tokens (only
# ~3 rows per expert). Past 4k tokens the bench's uniform routing favors
# 64x256, but the model's skewed routing does not: in a 16k-token prefill
# 128x128 takes 3.26 s of expert GEMM against 3.40 s for 64x256.
_WNA16_BLOCKS: dict[tuple[int, int], list[tuple[int, int, int]]] = {
    (1280, 2560): [
        (256, 16, 64),
        (1024, 32, 64),
        (2048, 64, 256),
        (1 << 62, 128, 128),
    ],
}


def wna16_blocks(num_tokens: int, n: int, k: int) -> tuple[int, int]:
    for max_tokens, bm, bn in _WNA16_BLOCKS.get((n, k), ()):
        if num_tokens <= max_tokens:
            return bm, bn
    return strix_blocks(num_tokens)


# (n, k) -> the fewest rows at which moe_gemm_bf16 beat vLLM's Triton expert
# GEMM on gfx1151. Rows are tokens for w13 and tokens * top_k for w2; a
# missing key means "no measured floor", i.e. the strix kernel is taken.
# Measured on gemma-4-26B-A4B bf16, whole step, 2k context, 128 steps, with
# moe_pad on both sides (bench/model/profile_model.py, profiles/moe-gate): strix is
# 11% slower at batch 1 and 6% at batch 2, then 23% faster at 4, parity at 8,
# and faster at 16/32/64. Triton wins only in the tiny range where it takes
# its per-(token, expert) naive path (no moe_align_block_size).
# gemma-4-26B-A4B: w13 N=1408 K=2816, w2 N=2816 K=704 (top_k 8), so a floor
# of 3 tokens is rows>=3 for w13 and rows>=24 for w2.
_MIN_ROWS: dict[tuple[int, int], int] = {
    (1408, 2816): 3,
    (2816, 704): 17,
}


def strix_moe_ok(rows: int, n: int, k: int) -> bool:
    """Whether the strix expert GEMM beats Triton at this (rows, n, k)."""
    return rows >= _MIN_ROWS.get((n, k), 0)


def _pad_rows(layer: torch.nn.Module, name: str) -> None:
    old = getattr(layer, name)
    *lead, k = old.shape
    store = torch.empty(*lead, k + ROW_PAD, dtype=old.dtype, device=old.device)
    store[..., :k].copy_(old)
    new = torch.nn.Parameter(store[..., :k], requires_grad=False)
    for attr, value in vars(old).items():
        setattr(new, attr, value)
    layer.register_parameter(name, new)
    # The freed packed tensor stays in the caching allocator, and the next
    # layer's (larger) padded tensor cannot reuse it: without this the cache
    # grows by one layer's experts per layer (13+ GB of host RAM on 35B).
    del old
    torch.cuda.empty_cache()


def _make_experts_cls():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache

    # Gated activations the fused epilogue can apply to the w13 gate half.
    # gemma-4's MoE is built with activation="gelu_tanh"
    # (hidden_activation gelu_pytorch_tanh); both are the same [gate; up] row
    # layout and the same n / 2 output, so only the epilogue differs.
    _FUSED = (MoEActivation.SILU, MoEActivation.GELU_TANH)

    class StrixTritonExperts(TritonExperts):
        def _strix_ok(self, hidden_states, w1, w2, activation, expert_map) -> bool:
            qc = self.quant_config
            return (
                hidden_states.dim() == 2
                and hidden_states.dtype == torch.bfloat16
                and w1.dtype == torch.bfloat16
                and w2.dtype == torch.bfloat16
                and not (
                    qc.use_fp8_w8a8
                    or qc.use_int8_w8a8
                    or qc.use_int8_w8a16
                    or qc.use_int4_w4a16
                )
                and self.w1_bias is None
                and self.w2_bias is None
                and self._lora_context is None
                and not self.expects_unquantized_inputs
                and expert_map is None
                and activation in _FUSED
                and self.activation_config.clamp_limit is None
                and w1.size(2) % 32 == 0
                and w2.size(2) % 32 == 0
                and w1.size(1) % 128 == 0
                and w2.size(1) % 128 == 0
                and w1.size(2) % 64 == 0
                and w2.size(2) % 64 == 0
            )

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            parent = super()
            if not self._strix_ok(hidden_states, w1, w2, activation, expert_map):
                return parent.apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            # A performance gate, not a capability one: shapes the tile table
            # was not tuned for go back to Triton in the range where it wins.
            num_tokens = hidden_states.size(0)
            if not (
                strix_moe_ok(num_tokens, w1.size(1), w1.size(2))
                and strix_moe_ok(num_tokens * topk_ids.size(1), w2.size(1), w2.size(2))
            ):
                return parent.apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            num_experts, num_tokens, n, k, top_k = self.moe_problem_size(
                hidden_states, w1, w2, topk_ids
            )
            bm, bn = strix_blocks(num_tokens)
            sorted_ids, expert_ids, num_post = moe_align_block_size(
                topk_ids, bm, num_experts
            )
            slots = num_tokens * top_k
            # w13 with the activation-and-mul fused into its epilogue:
            # [slots, n / 2]. Both fused activations halve N, so the cache
            # extent is vLLM's own (adjust_N_for_activation) for either.
            cache2 = _resize_cache(
                workspace13, (slots, self.adjust_N_for_activation(n, activation))
            )
            torch.ops.strix.moe_gemm_bf16(
                cache2,
                hidden_states,
                w1,
                sorted_ids,
                expert_ids,
                num_post,
                None,
                top_k,
                bm,
                bn,
                True,
                activation.value,
            )
            cache3 = _resize_cache(workspace2, (slots, k))
            weights = (
                None
                if apply_router_weight_on_input
                else topk_weights.reshape(-1).to(torch.float32)
            )
            torch.ops.strix.moe_gemm_bf16(
                cache3,
                cache2,
                w2,
                sorted_ids,
                expert_ids,
                num_post,
                weights,
                1,
                bm,
                bn,
            )
            self.moe_sum(
                cache3.view(num_tokens, top_k, k), output, topk_ids, expert_map
            )

    return StrixTritonExperts


def _make_wna16_experts_cls():
    """The int4 (compressed-tensors WNA16) expert path, for gemma-4-26B AWQ.

    Neither the quantized MoE method nor the experts class is a CustomOp, so
    register_oot cannot hook them; `oracle/int_wna16.py` resolves the module
    global `TritonWNA16Experts` in both its backend table and the
    `allowed_experts` assert, so register() rebinds that global.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonWNA16Experts,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache

    _FUSED = (MoEActivation.SILU, MoEActivation.GELU_TANH)

    class StrixTritonWNA16Experts(TritonWNA16Experts):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Built while vLLM's config names the model; see ARCH_GATED.
            self._strix_w4a8 = model_gate("moe_w4a8")

        def _strix_ok(
            self, hidden_states, w1, w2, activation, expert_map, a1q_scale, a2_scale
        ):
            qc = self.quant_config
            # Not quant_config.group_size: the experts see a FusedMoEQuantConfig,
            # whose WNA16 group is encoded as block_shape = [0, group].
            bs = self.block_shape
            group = None if not bs or len(bs) < 2 else bs[1]
            s1, s2 = self.w1_scale, self.w2_scale
            checks = {
                "2d": hidden_states.dim() == 2,
                "act_dtype": hidden_states.dtype == torch.bfloat16,
                "w1_uint8": w1.dtype == torch.uint8,
                "w2_uint8": w2.dtype == torch.uint8,
                "int4": qc.use_int4_w4a16,
                "act_scales": a1q_scale is None and a2_scale is None,
                "zero_point": self.w1_zp is None and self.w2_zp is None,
                "group": group in (32, 64, 128),
                "scales": (
                    s1 is not None
                    and s2 is not None
                    and s1.dtype in (torch.bfloat16, torch.float16, torch.float32)
                    and s2.dtype in (torch.bfloat16, torch.float16, torch.float32)
                ),
                "bias": self.w1_bias is None and self.w2_bias is None,
                "lora": self._lora_context is None,
                "expert_map": expert_map is None,
                "activation": activation in _FUSED,
                "w13_row": w1.size(2) % 8 == 0,
                "w2_row": w2.size(2) % 8 == 0,
            }
            bad = [name for name, ok in checks.items() if not ok]
            if bad:
                # Once per process: a silent fallback is how the first version
                # of this hook (bf16-scale-only) ran Triton on an fp16-scale
                # checkpoint without anyone noticing.
                from vllm.logger import init_logger

                init_logger("vllm.strix").warning_once(
                    "moe_w4a16 falls back to Triton: "
                    + ", ".join(bad)
                    + f" (group={group}, w1={w1.dtype}, "
                    + f"scales={None if s1 is None else s1.dtype}/"
                    + f"{None if s2 is None else s2.dtype})"
                )
            return not bad

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            parent = super()
            if not self._strix_ok(
                hidden_states, w1, w2, activation, expert_map, a1q_scale, a2_scale
            ):
                return parent.apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            num_tokens = hidden_states.size(0)
            top_k = topk_ids.size(1)
            # Logical (n, k) is what strix_moe_ok is keyed on; the weights hold
            # K/2 bytes per row. Same floor as the bf16 path.
            if not (
                strix_moe_ok(num_tokens, w1.size(1), w1.size(2) * 2)
                and strix_moe_ok(num_tokens * top_k, w2.size(1), w2.size(2) * 2)
            ):
                return parent.apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            num_experts = w1.size(0)
            n, k = w1.size(1), w2.size(1)
            group = int(self.block_shape[1])
            if self._strix_w4a8 and _w4a8_ok(num_tokens, group, n, k):
                return self._apply_w4a8(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    expert_map,
                    workspace13,
                    workspace2,
                    apply_router_weight_on_input,
                )
            bm, bn = wna16_blocks(num_tokens, n, w1.size(2) * 2)
            sorted_ids, expert_ids, num_post = moe_align_block_size(
                topk_ids, bm, num_experts
            )
            slots = num_tokens * top_k
            cache2 = _resize_cache(workspace13, (slots, n // 2))
            torch.ops.strix.moe_gemm_w4a16(
                cache2,
                hidden_states,
                w1,
                self.w1_scale,
                sorted_ids,
                expert_ids,
                num_post,
                None,
                top_k,
                bm,
                bn,
                True,
                activation.value,
                group,
            )
            cache3 = _resize_cache(workspace2, (slots, k))
            weights = (
                None
                if apply_router_weight_on_input
                else topk_weights.reshape(-1).to(torch.float32)
            )
            torch.ops.strix.moe_gemm_w4a16(
                cache3,
                cache2,
                w2,
                self.w2_scale,
                sorted_ids,
                expert_ids,
                num_post,
                weights,
                1,
                bm,
                bn,
                False,
                "silu",
                group,
            )
            self.moe_sum(
                cache3.view(num_tokens, top_k, k), output, topk_ids, expert_map
            )

        def _apply_w4a8(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            expert_map,
            workspace13,
            workspace2,
            apply_router_weight_on_input,
        ):
            """Prefill: int8 activations (per row and 128-element group) on the
            iu8 WMMA, see csrc/moe/moe_gemm_w4a8.hip."""
            num_tokens, top_k = hidden_states.size(0), topk_ids.size(1)
            n, k = w1.size(1), w2.size(1)
            bm, bn = 64, 256
            sorted_ids, expert_ids, num_post = moe_align_block_size(
                topk_ids, bm, w1.size(0)
            )
            slots = num_tokens * top_k
            qa, sa = _act_quant_i8(hidden_states)
            cache2 = _resize_cache(workspace13, (slots, n // 2))
            torch.ops.strix.moe_gemm_w4a8(
                cache2,
                qa,
                sa,
                w1,
                self.w1_scale,
                sorted_ids,
                expert_ids,
                num_post,
                None,
                top_k,
                bm,
                bn,
                True,
                activation.value,
            )
            qa, sa = _act_quant_i8(cache2)
            cache3 = _resize_cache(workspace2, (slots, k))
            weights = (
                None
                if apply_router_weight_on_input
                else topk_weights.reshape(-1).to(torch.float32)
            )
            torch.ops.strix.moe_gemm_w4a8(
                cache3,
                qa,
                sa,
                w2,
                self.w2_scale,
                sorted_ids,
                expert_ids,
                num_post,
                weights,
                1,
                bm,
                bn,
                False,
                "silu",
            )
            self.moe_sum(
                cache3.view(num_tokens, top_k, k), output, topk_ids, expert_map
            )

    return StrixTritonWNA16Experts


# moe_gemm_w4a8 (int8 activations): 1.34x (w13) / 1.19x (w2) over
# moe_gemm_w4a16 at 16k tokens of Qwen3.8-Flash-Next, quantization included.
# Teacher-forced top-1 vs the w4a16 path 1820/1820, mean |dlogprob| 0.038
# (switching w4a16's own weight staging moves it 0.036). Measured there only,
# so it is on for that architecture alone (strix_vllm.ARCH_GATED);
# STRIX_ENABLE=moe_w4a8 tries it elsewhere, STRIX_DISABLE=moe_w4a8 turns it off.
# STRIX_MOE_W4A8_MIN_TOKENS lowers the floor for accuracy runs on short prompts.
_W4A8_MIN_TOKENS = int(os.environ.get("STRIX_MOE_W4A8_MIN_TOKENS", "1024"))


def _w4a8_ok(num_tokens: int, group: int, n: int, k: int) -> bool:
    return (
        not disabled("moe_w4a8")
        and num_tokens >= _W4A8_MIN_TOKENS
        and group == 128
        and n % 128 == 0
        and k % 128 == 0
    )


def _act_quant_i8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    sa = torch.empty(
        (x.size(0), x.size(1) // 128), dtype=torch.float32, device=x.device
    )
    torch.ops.strix.moe_act_quant_i8(x, q, sa)
    return q, sa


def _make_int8_experts_cls():
    """The int8 expert path, for Qwen3.6-35B-A3B-INT8-W8A8.

    compressed-tensors int8 MoE keeps signed int8 [E, N, K] weights and one
    scale per output channel; activations are quantized per token by the
    reference. Strix dequantizes the weights into LDS and runs bf16 WMMA with
    bf16 activations (W8A16), which is more accurate and streams half the
    bytes. `oracle/int8.py` imports TritonExperts *inside* its selector, so
    register() rebinds `backend_to_kernel_cls` rather than a class global.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache

    _FUSED = (MoEActivation.SILU, MoEActivation.GELU_TANH)

    class StrixInt8Experts(TritonExperts):
        def _strix_ok(
            self, hidden_states, w1, w2, activation, expert_map, a1q_scale, a2_scale
        ):
            s1, s2 = self.w1_scale, self.w2_scale
            # vLLM's prepare step quantizes activations for this path, so they
            # arrive int8 with a per-token scale; a bf16 input is also accepted
            # (unquantized-inputs mode). apply() dequantizes in the int8 case.
            act_ok = (
                hidden_states.dtype == torch.int8
                and a1q_scale is not None
                and a2_scale is None
            ) or (
                hidden_states.dtype == torch.bfloat16
                and a1q_scale is None
                and a2_scale is None
            )
            checks = {
                "2d": hidden_states.dim() == 2,
                "act": act_ok,
                "w1_int8": w1.dtype == torch.int8,
                "w2_int8": w2.dtype == torch.int8,
                "int8_w8a8": self.quant_config.use_int8_w8a8,
                "scales": (
                    s1 is not None
                    and s2 is not None
                    and s1.dtype in (torch.bfloat16, torch.float16, torch.float32)
                    and s2.dtype in (torch.bfloat16, torch.float16, torch.float32)
                    and s1.numel() == w1.size(0) * w1.size(1)
                    and s2.numel() == w2.size(0) * w2.size(1)
                ),
                "bias": self.w1_bias is None and self.w2_bias is None,
                "lora": self._lora_context is None,
                "expert_map": expert_map is None,
                "activation": activation in _FUSED,
                "w13_row": w1.size(2) % 64 == 0,
                "w2_row": w2.size(2) % 64 == 0,
                "n128": w1.size(1) % 128 == 0 and w2.size(1) % 128 == 0,
            }
            bad = [name for name, ok in checks.items() if not ok]
            if bad:
                from vllm.logger import init_logger

                init_logger("vllm.strix").warning_once(
                    "moe_w8a16 falls back to Triton: "
                    + ", ".join(bad)
                    + f" | act {hidden_states.dtype} a1q {None if a1q_scale is None else tuple(a1q_scale.shape)}"
                    + f" w1 {tuple(w1.shape)} w2 {tuple(w2.shape)}"
                    + f" s1 {None if s1 is None else tuple(s1.shape)} {None if s1 is None else s1.dtype}"
                    + f" s2 {None if s2 is None else tuple(s2.shape)}"
                )
            return not bad

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            parent = super()
            fallback = {
                "output": output,
                "hidden_states": hidden_states,
                "w1": w1,
                "w2": w2,
                "topk_weights": topk_weights,
                "topk_ids": topk_ids,
                "activation": activation,
                "global_num_experts": global_num_experts,
                "expert_map": expert_map,
                "a1q_scale": a1q_scale,
                "a2_scale": a2_scale,
                "workspace13": workspace13,
                "workspace2": workspace2,
                "expert_tokens_meta": expert_tokens_meta,
                "apply_router_weight_on_input": apply_router_weight_on_input,
            }
            if not self._strix_ok(
                hidden_states, w1, w2, activation, expert_map, a1q_scale, a2_scale
            ):
                return parent.apply(**fallback)
            num_tokens = hidden_states.size(0)
            top_k = topk_ids.size(1)
            if not (
                strix_moe_ok(num_tokens, w1.size(1), w1.size(2))
                and strix_moe_ok(num_tokens * top_k, w2.size(1), w2.size(2))
            ):
                return parent.apply(**fallback)
            num_experts = w1.size(0)
            n, k = w1.size(1), w2.size(1)
            # int8 activations from the prepare step, with their own per-token
            # scale: dequantize exactly (fp32 product, then bf16) so the
            # activations match the reference's and only the weight accumu-
            # lation differs.
            x = hidden_states
            if x.dtype == torch.int8:
                x = (x * a1q_scale.reshape(-1, 1)).to(torch.bfloat16)
            s1 = self.w1_scale.reshape(num_experts, n).contiguous()
            s2 = self.w2_scale.reshape(num_experts, k).contiguous()
            bm, bn = strix_blocks(num_tokens)
            sorted_ids, expert_ids, num_post = moe_align_block_size(
                topk_ids, bm, num_experts
            )
            slots = num_tokens * top_k
            cache2 = _resize_cache(workspace13, (slots, n // 2))
            torch.ops.strix.moe_gemm_w8a16(
                cache2,
                x,
                w1,
                s1,
                sorted_ids,
                expert_ids,
                num_post,
                None,
                top_k,
                bm,
                bn,
                True,
                activation.value,
            )
            cache3 = _resize_cache(workspace2, (slots, k))
            weights = (
                None
                if apply_router_weight_on_input
                else topk_weights.reshape(-1).to(torch.float32)
            )
            torch.ops.strix.moe_gemm_w8a16(
                cache3,
                cache2,
                w2,
                s2,
                sorted_ids,
                expert_ids,
                num_post,
                weights,
                1,
                bm,
                bn,
                False,
                "silu",
            )
            self.moe_sum(
                cache3.view(num_tokens, top_k, k), output, topk_ids, expert_map
            )

    return StrixInt8Experts


def register() -> None:
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    pad = not disabled("moe_pad")
    gemm = not disabled("moe_gemm")
    w4a16 = not disabled("moe_w4a16")
    if pad or gemm:
        experts_cls = _make_experts_cls() if gemm else None

        @UnquantizedFusedMoEMethod.register_oot
        class StrixUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
            def __init__(self, moe):
                super().__init__(moe)
                if experts_cls is not None and self.experts_cls is TritonExperts:
                    self.experts_cls = experts_cls

            def process_weights_after_loading(self, layer) -> None:
                if pad and not self.moe.moe_parallel_config.enable_eplb:
                    _pad_rows(layer, "w13_weight")
                    _pad_rows(layer, "w2_weight")
                super().process_weights_after_loading(layer)

    if w4a16:
        from vllm.model_executor.layers.fused_moe.oracle import int_wna16

        cls = _make_wna16_experts_cls()
        int_wna16.TritonWNA16Experts = cls
        # A future upstream change that imports the class inside the selector
        # instead of reading the module global would silently bypass this.
        assert int_wna16.TritonWNA16Experts is cls

    # Opt-in: a bf16-WMMA W8A16 expert GEMM loses to Triton's int8 WMMA on
    # this checkpoint (prefill -3.7 %, decode x1 -32 %), so it is off unless
    # STRIX_MOE_W8A16 is set.
    if os.environ.get("STRIX_MOE_W8A16") and not disabled("moe_w8a16"):
        from vllm.model_executor.layers.fused_moe.oracle import int8 as int8_oracle

        cls = _make_int8_experts_cls()
        orig = int8_oracle.backend_to_kernel_cls

        def _backend_to_kernel_cls(backend, _orig=orig, _cls=cls):
            from vllm.model_executor.layers.fused_moe.oracle.int8 import (
                Int8MoeBackend,
            )

            if backend == Int8MoeBackend.TRITON:
                return [_cls]
            return _orig(backend)

        # Unlike the WNA16 oracle, int8.py imports TritonExperts inside the
        # selector, so the global to rebind is the selector itself.
        int8_oracle.backend_to_kernel_cls = _backend_to_kernel_cls
        assert int8_oracle.backend_to_kernel_cls is _backend_to_kernel_cls
