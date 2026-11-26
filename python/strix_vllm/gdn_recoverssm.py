# SPDX-License-Identifier: Apache-2.0
# Moved from the asf0/strix-vllm fork (dev, "perf(mamba): keep one GDN state
# per request under speculative decoding"):
# vllm/model_executor/layers/mamba/ops/gdn_recoverssm.py, unchanged. The fork
# keeps a seam of that name, which strix_vllm.register points at this module.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM speculative verify and accepted-state commit for Qwen GDN.

Verify runs the 1 + k token window off the request's checkpoint without
writing the recurrent state. It records each token's decay, normalized key
and beta-scaled correction in the checkpoint block; after sampling, the commit
replays the accepted prefix onto the checkpoint in the same operation order.
Speculative decoding then needs no per-draft-token state blocks.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_NUM_RECORDS = 3


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_ptr,
    key_ptr,
    decay_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_a_token,
    stride_b_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_key_block,
    stride_key_head,
    stride_key_pos,
    stride_decay_block,
    stride_decay_head,
    stride_out_token,
    stride_state_indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_hv = tl.program_id(2)
    pid_h = pid_hv // (HV // H)

    bos = tl.load(query_start_loc_ptr + pid_n).to(tl.int64)
    query_len = tl.load(query_start_loc_ptr + pid_n + 1).to(tl.int64) - bos
    state_idx = tl.load(state_indices_ptr + pid_n * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    out_ptrs = out_ptr + bos * stride_out_token + pid_hv * V + offs_v

    if state_idx <= null_block_id:
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptrs + t * stride_out_token,
                tl.zeros([BV], dtype=tl.float32),
                mask=(t < query_len) & mask_v,
            )
        return

    h = tl.load(
        state_ptr
        + state_idx * stride_state_block
        + pid_hv * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :],
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_hv).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    correction_ptrs = (
        correction_ptr
        + state_idx * stride_correction_block
        + pid_hv * stride_correction_head
        + offs_v
    )
    key_ptrs = key_ptr + state_idx * stride_key_block + pid_h * stride_key_head + offs_k
    decay_ptrs = decay_ptr + state_idx * stride_decay_block + pid_hv * stride_decay_head
    write_key = (pid_v == 0) & (pid_hv % (HV // H) == 0)

    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        token = bos + t
        b_q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        b_k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_ptr + token * stride_v_token + pid_hv * V + offs_v,
            mask=valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        b_a = tl.load(a_ptr + token * stride_a_token + pid_hv, mask=valid, other=0.0)
        b_b = tl.load(b_ptr + token * stride_b_token + pid_hv, mask=valid, other=0.0)

        # Same operation order as fused_sigmoid_gating_delta_rule_update, so a
        # commit that replays the records reproduces its states.
        x = b_a.to(tl.float32) + dt_bias
        softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
        b_decay = tl.exp(neg_A * softplus_x)
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        b_h = h * b_decay
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        h = tl.where(valid, b_h, h)
        b_o = tl.sum(h * b_q[None, :], 1)
        tl.store(
            out_ptrs + t * stride_out_token,
            b_o.to(out_ptr.dtype.element_ty),
            mask=valid & mask_v,
        )

        tl.store(correction_ptrs + t * stride_correction_pos, b_v, mask=valid & mask_v)
        tl.store(decay_ptrs + t, b_decay, mask=valid & (pid_v == 0))
        tl.store(key_ptrs + t * stride_key_pos, b_k, mask=valid & write_key & mask_k)


@triton.jit
def _commit_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_ref_ptr,
    correction_base_addrs_ptr,
    correction_block_strides_ptr,
    key_ref_ptr,
    key_base_addrs_ptr,
    key_block_strides_ptr,
    decay_ref_ptr,
    decay_base_addrs_ptr,
    decay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_correction_head,
    stride_correction_pos,
    stride_key_head,
    stride_key_pos,
    stride_decay_head,
    stride_state_indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // HV
    pid_hv = pid_lh % HV
    pid_h = pid_hv // (HV // H)

    source_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_idx <= null_block_id:
        return

    state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
        tl.pointer_type(state_ref_ptr.dtype.element_ty)
    )
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    correction_ptr = tl.load(correction_base_addrs_ptr + pid_l).to(
        tl.pointer_type(correction_ref_ptr.dtype.element_ty)
    )
    correction_ptr += (
        source_idx * tl.load(correction_block_strides_ptr + pid_l)
        + pid_hv * stride_correction_head
    )
    key_ptr = tl.load(key_base_addrs_ptr + pid_l).to(
        tl.pointer_type(key_ref_ptr.dtype.element_ty)
    )
    key_ptr += source_idx * tl.load(key_block_strides_ptr + pid_l)
    key_ptr += pid_h * stride_key_head
    decay_ptr = tl.load(decay_base_addrs_ptr + pid_l).to(
        tl.pointer_type(decay_ref_ptr.dtype.element_ty)
    )
    decay_ptr += (
        source_idx * tl.load(decay_block_strides_ptr + pid_l)
        + pid_hv * stride_decay_head
    )

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    tile_offsets = (
        pid_hv * stride_state_head + offs_v[:, None] * stride_state_v + offs_k[None, :]
    )
    h = tl.load(
        state_ptr + source_idx * state_block_stride + tile_offsets,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)

    boundary_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_len = tl.load(boundary_recovery_lens_ptr + pid_b)
    for t in range(commit_len):
        b_decay = tl.load(decay_ptr + t)
        b_v = tl.load(
            correction_ptr + t * stride_correction_pos + offs_v,
            mask=mask_v,
            other=0.0,
        )
        b_k = tl.load(key_ptr + t * stride_key_pos + offs_k, mask=mask_k, other=0.0)
        # Verify rounds the decayed state before the rank-1 update (it is also
        # read by the correction); an explicit fma keeps that rounding here.
        h = tl.fma(b_v[:, None], b_k[None, :], h * b_decay)
        if ALIGN_MODE:
            tl.store(
                state_ptr + boundary_idx * state_block_stride + tile_offsets,
                h.to(state_ref_ptr.dtype.element_ty),
                mask=mask_state
                & (t + 1 == boundary_len)
                & (boundary_idx > null_block_id),
            )

    tl.store(
        state_ptr + final_idx * state_block_stride + tile_offsets,
        h.to(state_ref_ptr.dtype.element_ty),
        mask=mask_state,
    )


def gdn_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    key_cache: torch.Tensor,
    decay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Verify a GDN speculative window without modifying its checkpoint.

    Args:
        q: Queries, [1, tokens, num_k_heads, head_k_dim].
        k: Keys, [1, tokens, num_k_heads, head_k_dim].
        v: Values, [1, tokens, num_v_heads, head_v_dim].
        a: Raw decay gate, [tokens, num_v_heads].
        b: Raw beta gate, [tokens, num_v_heads].
        A_log: Per-head log decay rate, [num_v_heads].
        dt_bias: Per-head decay bias, [num_v_heads].
        checkpoint_state: Recurrent state, [blocks, num_v_heads, head_v_dim,
            head_k_dim]. Read only.
        correction_cache: Beta-scaled corrections, [blocks, num_v_heads,
            spec_query_len, head_v_dim], float32.
        key_cache: Normalized keys, [blocks, num_k_heads, spec_query_len,
            head_k_dim], float32.
        decay_cache: Per-token decay factors, [blocks, num_v_heads,
            spec_query_len], float32.
        query_start_loc: Cumulative query lengths, [batch + 1].
        state_indices: Checkpoint block of each request, [batch].
        spec_query_len: Maximum query length, 1 + num_speculative_tokens.
        scale: Query scale; defaults to head_k_dim ** -0.5.

    Returns:
        Attention output, [1, tokens, num_v_heads, head_v_dim].

    """
    _, total_tokens, num_k_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]
    if k.shape != q.shape or v.shape[:2] != (1, total_tokens):
        raise ValueError("GDN RecoverSSM q, k, and v shapes are incompatible")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("GDN RecoverSSM needs num_v_heads divisible by num_k_heads")
    if a.shape != (total_tokens, num_v_heads) or b.shape != a.shape:
        raise ValueError("GDN RecoverSSM gate shapes are incompatible")
    if a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("GDN RecoverSSM gate heads must be contiguous")
    if any(t.stride()[2:] != (key_dim, 1) for t in (q, k)) or v.stride()[2:] != (
        value_dim,
        1,
    ):
        raise ValueError("GDN RecoverSSM q, k, and v heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (num_v_heads, value_dim, key_dim):
        raise ValueError("GDN RecoverSSM checkpoint shape is incompatible")
    if checkpoint_state.stride(3) != 1:
        raise ValueError("GDN RecoverSSM checkpoint key dim must be contiguous")
    expected_shapes = (
        (correction_cache, (num_blocks, num_v_heads, spec_query_len, value_dim)),
        (key_cache, (num_blocks, num_k_heads, spec_query_len, key_dim)),
        (decay_cache, (num_blocks, num_v_heads, spec_query_len)),
    )
    for record, shape in expected_shapes:
        if record.shape != shape or record.dtype != torch.float32:
            raise ValueError(f"GDN RecoverSSM record needs float32 shape {shape}")
        if record.stride(-1) != 1:
            raise ValueError("GDN RecoverSSM records must be contiguous per row")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("GDN RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its speculative window")
    out = torch.empty_like(v)
    if total_tokens == 0 or batch == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_v), batch, num_v_heads)
    _gdn_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        key_cache,
        decay_cache,
        out,
        query_start_loc,
        state_indices,
        key_dim**-0.5 if scale is None else scale,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(0),
        b.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        decay_cache.stride(0),
        decay_cache.stride(1),
        out.stride(1),
        state_indices.stride(0),
        H=num_k_heads,
        HV=num_v_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=4,
        num_stages=1,
    )
    return out


def _addresses(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
    )


def _block_strides(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [t.stride(0) for t in tensors], dtype=torch.int64, device=device
    )


@dataclass
class GDNRecoverSSMCommitContext:
    """Per-group commit plan over every GDN layer that shares a block table."""

    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    states: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    records: tuple[tuple[torch.Tensor, ...], ...]
    record_base_addrs: tuple[torch.Tensor, ...]
    record_block_strides: tuple[torch.Tensor, ...]
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    num_k_heads: int
    spec_query_len: int

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "GDNRecoverSSMCommitContext":
        if not layers:
            raise ValueError("GDN RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 2 + _NUM_RECORDS for layer in layers):
            raise ValueError(
                "GDN RecoverSSM pages must contain conv, state, correction, key, "
                "and decay"
            )

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        states = [layer.kv_cache[1] for layer in layers]
        records = tuple(
            tuple(layer.kv_cache[2 + i] for layer in layers)
            for i in range(_NUM_RECORDS)
        )

        state_ref = states[0]
        num_blocks = state_ref.shape[0]
        for name, tensors in (
            ("state", states),
            ("conv", conv_states),
            *zip(("correction", "key", "decay"), records),
        ):
            ref = tensors[0]
            if any(
                t.shape != ref.shape
                or t.dtype != ref.dtype
                or t.device != state_ref.device
                or t.stride()[1:] != ref.stride()[1:]
                for t in tensors
            ):
                raise ValueError(f"GDN RecoverSSM layers need matching {name} layout")
            if ref.shape[0] != num_blocks:
                raise ValueError(f"GDN RecoverSSM {name} block count mismatches")
        if state_ref.stride(3) != 1:
            raise ValueError("GDN RecoverSSM checkpoint key dim must be contiguous")
        num_k_heads = records[1][0].shape[1]
        if records[1][0].shape[2] != spec_query_len:
            raise ValueError("GDN RecoverSSM records do not match the window")

        conv_history_len = conv_states[0].shape[2] - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("GDN RecoverSSM conv state is shorter than its window")

        device = state_ref.device

        def _int32_plan() -> torch.Tensor:
            return torch.empty(max_num_reqs, dtype=torch.int32, device=device)

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_addresses(conv_states, device),
            conv_state_block_strides=_block_strides(conv_states, device),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            states=tuple(states),
            state_base_addrs=_addresses(states, device),
            state_block_strides=_block_strides(states, device),
            records=records,
            record_base_addrs=tuple(_addresses(r, device) for r in records),
            record_block_strides=tuple(_block_strides(r, device) for r in records),
            commit_lens=_int32_plan(),
            final_state_indices=_int32_plan(),
            boundary_state_indices=_int32_plan(),
            boundary_recovery_lens=_int32_plan(),
            num_k_heads=num_k_heads,
            spec_query_len=spec_query_len,
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold each request's accepted tokens into every layer's checkpoint.

        Args:
            num_accepted_tokens: Tokens to commit per batch row (the sampled
                count: accepted drafts plus the verified input token).
            state_indices: Checkpoint block of each speculative request.
            query_start_loc: Cumulative speculative query lengths.
            request_indices: Batch row of each speculative request, or None
                when they are the leading rows.
            block_table: Align mode only: the group's block table, by batch
                row; the commit then also moves the state to the block of its
                new position and fills a crossed block boundary.
            num_computed_tokens: Align mode only: tokens computed before this
                step, by batch row.
            mamba_block_size: Align mode only: tokens per state block.

        """
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("GDN RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("GDN RecoverSSM commit metadata is incompatible")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        align_mode = block_table is not None
        if any(arg is None for arg in align_args) and align_mode:
            raise ValueError("GDN RecoverSSM align metadata is incomplete")
        if align_mode:
            assert mamba_block_size is not None and num_computed_tokens is not None
            if mamba_block_size < self.spec_query_len:
                raise ValueError(
                    "GDN RecoverSSM align block size must cover one window"
                )

        block_table_strides = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_strides[0],
            block_table_strides[1],
            num_computed_tokens.stride(0) if num_computed_tokens is not None else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )

        num_layers = len(self.states)
        conv_dim = self.conv_states[0].shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            self.conv_states[0],
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align_mode,
            num_warps=4,
        )

        state_ref = self.states[0]
        _, num_v_heads, value_dim, key_dim = state_ref.shape
        correction_ref, key_ref, decay_ref = (r[0] for r in self.records)
        block_v = min(triton.next_power_of_2(value_dim), 32)
        grid = (triton.cdiv(value_dim, block_v), batch, num_layers * num_v_heads)
        _commit_gdn_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            correction_ref,
            self.record_base_addrs[0],
            self.record_block_strides[0],
            key_ref,
            self.record_base_addrs[1],
            self.record_block_strides[1],
            decay_ref,
            self.record_base_addrs[2],
            self.record_block_strides[2],
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            correction_ref.stride(1),
            correction_ref.stride(2),
            key_ref.stride(1),
            key_ref.stride(2),
            decay_ref.stride(1),
            state_indices.stride(0),
            H=self.num_k_heads,
            HV=num_v_heads,
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            BV=block_v,
            ALIGN_MODE=align_mode,
            num_warps=4,
            num_stages=1,
        )


__all__ = ["GDNRecoverSSMCommitContext", "gdn_recoverssm_verify"]
