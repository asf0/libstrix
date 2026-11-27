# SPDX-License-Identifier: MIT
"""XDNA2 NPU prefill for gemma-4-31B bf16 (dense, hidden 5,376): the NPU computes
the sliding-attention layers' fused qkv_proj (16,384 columns, the GdnIn shape)
of one bank while the GPU runs the other bank's layer.

Same scheme as Qwen3.8-27B (qwen3_5_npu): a single-request prefill step of
STRIX_NPU_MIN_TOKENS or more is split into two banks with their own attention
metadata (qwen4_exp_npu._bank_contexts, here on the attention-only
DefaultModelState). The NPU program takes K = 2,560, so the projection runs as
two chains over K columns [0, 2560) and [2560, 5120), summed; the last 256
columns of K and the rows past the NPU's program sizes are computed on the GPU.
The 10 full-attention layers (20,480 columns) stay on the GPU.

EXPERIMENTAL, opt-in with STRIX_ENABLE=npu_gemma (needs a usable NPU): its
output does not yet match the compiled model and it is slower (see enabled()).
bf16 checkpoints only: a quantized qkv_proj keeps the GPU path.
"""

from __future__ import annotations

import torch

from strix_vllm import disabled, npu
from strix_vllm import qwen3_5_npu as q35
from strix_vllm import qwen4_exp_npu as shared

_MIN_TOKENS = q35._MIN_TOKENS
_ROWS_MAX = q35._ROWS_MAX
_SIZES = q35._SIZES
_HALVES = 2  # K chunks of npu.K on the NPU; the rest of K on the GPU


class _State:
    runtimes: list | None = None  # one per K chunk
    failed = False
    layers: dict[int, torch.Tensor] = {}  # layer index -> GPU part of K (the tail) of qkv_proj
    wake_x = None
    wake_layer = 0


def enabled() -> bool:
    """Experimental, opt-in only (STRIX_ENABLE=npu_gemma): the bank loop does not
    yet match the compiled model (KLD 0.17 at 16k without NPU rows, 0.40 with)
    and runs ~20 % slower than it. Not part of the NPU's auto-detection."""
    from strix_vllm import _env_names

    return (not disabled("npu") and "npu_gemma" in _env_names("STRIX_ENABLE")
            and npu.unavailable() is None)


def _logger():
    from vllm.logger import init_logger

    return init_logger("vllm.strix_vllm.gemma4_npu")


def _weight(layer) -> torch.Tensor | None:
    """A sliding-attention layer's fused qkv_proj weight [n, k_total] (bf16,
    n <= the GdnIn pitch, k_total >= two NPU chunks), or None."""
    attn = getattr(layer, "self_attn", None)
    if attn is None or not getattr(attn, "is_sliding", False):
        return None
    if getattr(attn, "is_kv_shared_layer", False):
        return None
    proj = getattr(attn, "qkv_proj", None)
    w = getattr(proj, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
        return None
    if w.shape[1] < _HALVES * npu.K or w.shape[0] > 16640:
        return None
    return w


def _open(model) -> list | None:
    """One runtime per K chunk, every sliding layer's qkv_proj resident."""
    if _State.runtimes is not None or _State.failed:
        return _State.runtimes
    log = _logger()
    try:
        why = npu.util_blocks()
        if why:
            raise RuntimeError(why)
        weights = {}
        for idx, layer in enumerate(model.layers):
            w = _weight(layer)
            if w is not None:
                weights[idx] = w
        if not weights:
            raise RuntimeError("no bf16 sliding-layer qkv_proj weights")
        pci = shared._pci()
        runtimes = []
        for c in range(_HALVES):
            entries = [
                (idx, npu.GDN_IN, w[:, c * npu.K: (c + 1) * npu.K].contiguous())
                for idx, w in weights.items()
            ]
            runtimes.append(npu.NpuRuntime.open(_ROWS_MAX, entries, pci, banks=2))
        for rt in runtimes:
            _warm(rt, min(weights))
        _State.layers = {idx: w[:, _HALVES * npu.K:] for idx, w in weights.items()}
        _State.wake_layer = min(weights)
        _State.wake_x = torch.zeros((_SIZES[0], npu.K), dtype=torch.bfloat16, device="cuda")
        _State.runtimes = runtimes
        tail = next(iter(_State.layers.values())).shape[1]
        log.info(
            "libstrix: npu prefill on (gemma-4): %d sliding layers, K %d on the NPU + %d on the GPU, "
            "two banks of <= %d rows, from %d tokens",
            len(weights), _HALVES * npu.K, tail, _ROWS_MAX, _MIN_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        _State.failed = True
        log.warning("libstrix: npu prefill off: %s", e)
    return _State.runtimes


def _warm(rt, layer_idx: int) -> None:
    """Runs every row-count program once on both banks, so their hardware
    contexts exist before a request (a variant's first chain takes ~180 ms)."""
    for rows in _SIZES:
        x = torch.zeros((rows, npu.K), dtype=torch.bfloat16, device="cuda")
        for bank in range(2):
            seq = rt.kick(layer_idx, npu.GDN_IN, x, bank)
            rt.result(npu.GDN_IN, rows, seq, bank)
            rt.retire(seq, bank)
    torch.cuda.synchronize()
    rt.check()


def _kick(x: torch.Tensor, layer_idx: int, bank: int) -> tuple:
    """One chain per K chunk of x's first rows; returns their sequence numbers."""
    return tuple(
        rt.kick(layer_idx, npu.GDN_IN, x[:, c * npu.K: (c + 1) * npu.K], bank)
        for c, rt in enumerate(_State.runtimes)
    )


def _assemble(idx: int, w: torch.Tensor, x: torch.Tensor, k: int, seqs: tuple, bank: int) -> torch.Tensor:
    """qkv_proj of x: rows < k from the NPU chunks plus the GPU's K tail, the rest
    of the rows on the GPU."""
    m, n = x.shape[0], w.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if k < m:
        torch.mm(x[k:], w.t(), out=out[k:])
    tail = _State.layers[idx]
    if tail.shape[1]:
        torch.mm(x[:k, _HALVES * npu.K:], tail.t(), out=out[:k])
    else:
        out[:k].zero_()
    for c, rt in enumerate(_State.runtimes):
        out[:k].add_(rt.result(npu.GDN_IN, k, seqs[c], bank)[:, :n])
    for c, rt in enumerate(_State.runtimes):
        rt.retire(seqs[c], bank)
    return out


class _Given(torch.nn.Module):
    """Stand-in qkv_proj returning the tensor set for this call."""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):  # noqa: ARG002
        out, self.out = self.out, None
        assert out is not None, "NPU stand-in called without a projection"
        return out, None


def _attn(layer, normed, positions, given):
    attn = layer.self_attn
    if given is None:
        return attn(positions=positions, hidden_states=normed)
    proj = attn.qkv_proj
    attn.qkv_proj = _Given()
    attn.qkv_proj.out = given
    try:
        return attn(positions=positions, hidden_states=normed)
    finally:
        attn.qkv_proj = proj


def _layer_body(layer, normed, residual, positions, given):
    """Gemma4DecoderLayer.forward after its input_layernorm (dense, no PLE)."""
    hidden = _attn(layer, normed, positions, given)
    hidden = layer.post_attention_layernorm(hidden) + residual
    residual = hidden
    hidden = layer.mlp(layer.pre_feedforward_layernorm(hidden))
    hidden = layer.post_feedforward_layernorm(hidden) + residual
    return hidden * layer.layer_scalar


def forward(model, input_ids, positions, inputs_embeds=None, profile=False):
    """Gemma4Model.forward for one single-request prefill step, two banks.

    `profile`: vLLM's metadata-free memory-profile run through the same eager
    path, so the activation peak the KV cache is sized against is this path's
    (it is larger than the compiled model's)."""
    import contextlib

    from vllm.forward_context import override_forward_context

    n = positions.shape[-1]
    a = q35._split(n)
    rows = (slice(0, a), slice(a, n))
    layers = model.layers

    rt_a = _State.runtimes[0]
    wake = rt_a.kick(_State.wake_layer, npu.GDN_IN, _State.wake_x, 0)
    hidden = model.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
    aux_layers = set(getattr(model, "aux_hidden_state_layers", ()))
    aux = {0: [hidden.clone()]} if 0 in aux_layers else {}
    rt_a.result(npu.GDN_IN, _SIZES[0], wake, 0)
    rt_a.retire(wake, 0)

    ctx = [None, None] if profile else shared._bank_contexts(a, n)
    pos = [positions[..., sl] for sl in rows]
    state = [hidden[rows[b]] for b in (0, 1)]
    pre = [None, None]  # (normed input, residual, NPU seqs, NPU rows) per bank

    def premix(b, layer, idx):
        h = state[b]
        normed = layer.input_layernorm(h)
        k = q35._rows(normed.shape[0]) if idx in _State.layers else 0
        pre[b] = (normed, h, _kick(normed[:k], idx, b) if k else None, k)

    for b in (0, 1):
        premix(b, layers[0], 0)

    for idx in range(len(layers)):
        layer = layers[idx]
        nxt = layers[idx + 1] if idx + 1 < len(layers) else None
        for b in (0, 1):
            normed, res, seqs, k = pre[b]
            pre[b] = None
            given = None
            if seqs is not None:
                given = _assemble(idx, layer.self_attn.qkv_proj.weight, normed, k, seqs, b)
            with contextlib.nullcontext() if profile else override_forward_context(ctx[b]):
                state[b] = _layer_body(layer, normed, res, pos[b], given)
            if idx + 1 in aux_layers:
                aux.setdefault(idx + 1, []).append(state[b].clone())
            if nxt is not None:
                premix(b, nxt, idx + 1)

    out = torch.cat([model.norm(state[b]) for b in (0, 1)])
    if not profile:
        shared._restore_step_metadata()
    torch.cuda.current_stream().synchronize()
    for rt in _State.runtimes:
        try:
            rt.check()
        except RuntimeError as e:
            _State.failed = True
            _State.runtimes = None
            _logger().error("libstrix: npu prefill off after a failed NPU chain: %s", e)
            raise
    if aux_layers:
        return out, [torch.cat(aux[i]) for i in sorted(aux)]
    return out


def _patch(cls, inner_of, before=None) -> None:
    orig = cls.forward

    def patched(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        inner = inner_of(self)
        n = positions.shape[-1]
        st = shared._State.step
        long_step = (
            n >= _MIN_TOKENS
            and intermediate_tensors is None
            and (input_ids is not None or inputs_embeds is not None)
            and not torch.cuda.is_current_stream_capturing()
            and not getattr(inner, "fast_prefill_enabled", False)
            and getattr(self, "per_layer_embeddings", None) is None
            and getattr(inner.config, "hidden_size_per_layer_input", 0) in (0, None)
        )
        if long_step:
            # The first long step is vLLM's startup profile run: the NPU's pinned
            # memory and this path's activations are counted before the KV cache.
            _open(inner)
        if long_step and _State.runtimes is not None and _is_profile_run():
            if before is not None:
                before(self)
            return forward(inner, input_ids, positions, inputs_embeds, profile=True)
        if (
            long_step
            and st is not None
            and st["input_batch"].num_reqs == 1
            and st["input_batch"].num_tokens == n
            and _State.runtimes is not None
            and "prepare" in st
        ):
            if before is not None:
                before(self)
            return forward(inner, input_ids, positions, inputs_embeds)
        return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    cls.forward = patched
    cls._strix_npu = True


def _is_profile_run() -> bool:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return is_forward_context_available() and get_forward_context().attn_metadata is None


def register() -> None:
    if not enabled():
        return
    try:
        from vllm.model_executor.models import gemma4
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    except ImportError:
        return
    shared.install_step_capture(DefaultModelState)
    causal = getattr(gemma4, "Gemma4ForCausalLM", None)
    if causal is not None and not getattr(causal, "_strix_npu", False):
        _patch(causal, lambda m: m.model)
    try:
        from vllm.model_executor.models import gemma4_mm
    except ImportError:
        gemma4_mm = None
    mm = getattr(gemma4_mm, "Gemma4ForConditionalGeneration", None) if gemma4_mm else None
    if mm is not None and not getattr(mm, "_strix_npu", False):
        _patch(
            mm, lambda m: m.language_model.model,
            before=lambda m: getattr(m, "_clear_mm_prefix_for_full_attn_layers", lambda: None)(),
        )
    _logger().info(
        "libstrix: npu prefill patch installed for gemma-4 (%s)",
        "STRIX_ENABLE=npu" if npu.forced() else f"{npu.DEVICE} found",
    )
