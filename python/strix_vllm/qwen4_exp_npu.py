# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next prefill with the layers' input projections on the NPU.

STRIX_ENABLE=npu. A single-request prefill step of N >= STRIX_NPU_MIN_TOKENS
tokens is split into two banks, A (the first ~N/2 tokens) and B, each with its
own attention / GDN / QSA metadata (bank B sees bank A as context, as if the
step were two chunks). Layers 0 and 1 run on all rows (layer 1 applies PLE, a
convolution across tokens). From layer 2 on every layer runs bank A, then bank
B, and the NPU computes each bank's input projection (GDN in_proj_qkvz +
in_proj_ba, QSA qkv_proj) while the GPU runs the other bank:

    GPU:  L[A] mixer, MoE -> pre-mix L+1[A], kick NPU   L[B] ... kick NPU   L+1[A] ...
    NPU:  in_proj L[B]                                    in_proj L+1[A]      in_proj L+1[B]

so the GPU runs no input projection from layer 3 on (layer 2's bank A projection
stays on the GPU: there is no earlier bank to hide it behind). The mixers get the
projections through stand-in modules and otherwise run vLLM's code.

The banks' metadata comes from the model state's own prepare_attn on sliced
input batches (vllm.v1.worker.gpu.ubatch_utils), cloned so the two do not share
persistent buffers; after the step the step's own metadata is rebuilt in place
for the drafter. The step runs eagerly, outside torch.compile.

Numerics: the NPU part is W8A8 (int8 per output row / per token, grid15-up
scales, one BF16 rounding; strix_vllm.npu). The bank split's overheads,
measured on a 16,384-token step: splitting the MoE costs 4.4 ms per layer
(W4A8 experts: 48.2 ms as one 16,384-token call, 52.6 ms as two of 8,192),
copying the NPU's rows out of its arena 2.4 ms (as fast as a device copy).
"""

from __future__ import annotations

import dataclasses
import os

import torch
from torch import nn

from strix_vllm import disabled, _env_names

_MIN_TOKENS = int(os.environ.get("STRIX_NPU_MIN_TOKENS") or "4096")
# NPU row counts: the largest is the full program, the others the four
# row-adaptive programs the transport keeps besides it. A bank's rows past the
# largest size that fits are projected on the GPU.
_SIZES = tuple(sorted(int(x) for x in (os.environ.get("STRIX_NPU_SIZES") or "2048,4096,6144,7168,8192").split(",")))
_ROWS_MAX = _SIZES[-1]
_WAKE_ROWS = _SIZES[0]
_DEBUG = os.environ.get("STRIX_NPU_DEBUG", "") not in ("", "0")


def enabled() -> bool:
    return "npu" in _env_names("STRIX_ENABLE") and not disabled("npu")


class _Given(nn.Module):
    """Stand-in for a projection: returns the tensor set for this call."""

    def __init__(self):
        super().__init__()
        self.out: torch.Tensor | None = None

    def forward(self, x):  # noqa: ARG002
        out, self.out = self.out, None
        assert out is not None, "NPU stand-in called without a projection"
        return out, None


class _State:
    rt = None
    failed = False
    layers: dict[int, int] = {}  # layer index -> npu shape
    wake_x: torch.Tensor | None = None
    wake_layer = 0
    step: dict | None = None  # the current step's prepare_attn arguments
    bufs: list | None = None  # per-bank query_start_loc / seq_lens buffers


def _logger():
    from vllm.logger import init_logger

    return init_logger("vllm.strix_vllm.qwen4_exp_npu")


def _pci() -> str:
    env = os.environ.get("STRIX_NPU_PCI")
    if env:
        return env
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"


def _projection_weight(layer) -> tuple[int, torch.Tensor] | None:
    from strix_vllm import npu

    if layer.layer_type == "linear_attention":
        m = layer.linear_attn
        return npu.GDN_IN, torch.cat([m.in_proj_qkvz.weight, m.in_proj_ba.weight])
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "qkv_proj"):
        return npu.QSA_IN, layer.self_attn.qkv_proj.weight
    return None


def _projection_weights(layer) -> tuple[torch.Tensor, ...]:
    if layer.layer_type == "linear_attention":
        m = layer.linear_attn
        return m.in_proj_qkvz.weight, m.in_proj_ba.weight
    return (layer.self_attn.qkv_proj.weight,)


def _runtime(model):
    """Opens the NPU transport on first use, with layers 2.. resident, two banks.
    Any failure (no /dev/accel/accel0, no amdxdna driver, no libstrix_npu.so, no
    memlock) logs once and leaves every step on the GPU."""
    if _State.rt is not None or _State.failed:
        return _State.rt
    from strix_vllm import npu

    log = _logger()
    try:
        entries = []
        for idx, layer in enumerate(model.layers):
            if idx < 2 or layer.ple is not None:
                continue
            got = _projection_weight(layer)
            if got is None:
                continue
            shape, w = got
            if w.dtype != torch.bfloat16 or w.shape[1] != npu.K:
                continue
            entries.append((idx, shape, w))
            _State.layers[idx] = shape
        rt = npu.NpuRuntime.open(_ROWS_MAX, entries, _pci(), banks=2)
        _warm(rt)
        _State.wake_layer = min(_State.layers)
        _State.wake_x = torch.zeros((_WAKE_ROWS, npu.K), dtype=torch.bfloat16, device="cuda")
        _State.rt = rt
        log.info(
            "libstrix: npu prefill on: %d layers resident, two banks of <= %d rows, from %d tokens",
            len(entries), _ROWS_MAX, _MIN_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        _State.failed = True
        log.warning("libstrix: npu prefill off: %s", e)
    return _State.rt


def _warm(rt) -> None:
    """Runs every row-count program once per shape and bank, so their hardware
    contexts exist before a request (a variant's first chain takes ~180 ms)."""
    from strix_vllm import npu

    first = {}
    for idx, shape in _State.layers.items():
        first.setdefault(shape, idx)
    x = torch.zeros((max(_SIZES), npu.K), dtype=torch.bfloat16, device="cuda")
    for rows in _SIZES:
        for shape, idx in first.items():
            for bank in range(2):
                seq = rt.kick(idx, shape, x[:rows], bank)
                rt.result(shape, rows, seq, bank)
                rt.retire(seq, bank)
    torch.cuda.synchronize()
    rt.check()


def _npu_rows(m: int) -> int:
    """Rows of an m-row bank the NPU projects (the rest go to the GPU)."""
    return max((r for r in _SIZES if r <= m), default=0)


def _split(n: int) -> int:
    """Bank A's token count: about half the step, a multiple of 1024."""
    return n // 2 // 1024 * 1024


def _run_mixer(layer, block_input, positions, given: tuple[torch.Tensor, ...] | None):
    """The layer's mixer; `given` replaces its input projection."""
    if layer.layer_type == "linear_attention":
        m = layer.linear_attn
        if given is None:
            return m(hidden_states=block_input)
        saved = (m.in_proj_qkvz, m.in_proj_ba)
        m.in_proj_qkvz, m.in_proj_ba = _Given(), _Given()
        m.in_proj_qkvz.out, m.in_proj_ba.out = given
        try:
            return m(hidden_states=block_input)
        finally:
            m.in_proj_qkvz, m.in_proj_ba = saved
    a = layer.self_attn
    if given is None:
        return a(hidden_states=block_input, positions=positions)
    saved = a.qkv_proj
    a.qkv_proj = _Given()
    a.qkv_proj.out = given[0]
    try:
        return a(hidden_states=block_input, positions=positions)
    finally:
        a.qkv_proj = saved


def _assemble(layer, rt, shape, seq, bank, k, x):
    """The projection of a bank's rows x: the first k from the NPU's C arena of
    `bank`, the rest computed by the GPU straight into the result."""
    m = x.shape[0]
    out = []
    for w in _projection_weights(layer):
        full = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
        if k < m:
            torch.mm(x[k:], w.t(), out=full[k:])
        out.append(full)
    c = rt.result(shape, k, seq, bank)
    col = 0
    for full in out:
        w = full.shape[1]
        full[:k].copy_(c[:, col: col + w])
        col += w
    if _DEBUG:
        _logger().info("libstrix: npu layer %d bank %d rows %d: NPU %.1f ms", layer.layer_idx, bank, k, rt.busy_ms(seq, bank))
    rt.retire(seq, bank)
    return tuple(out)


def _clone(obj, seen=None):
    """Deep copy of attention metadata with every tensor cloned, so a bank's
    metadata stops sharing the model state's persistent buffers."""
    if seen is None:
        seen = {}
    if id(obj) in seen:
        return seen[id(obj)]
    if isinstance(obj, torch.Tensor):
        out = obj.clone()
    elif isinstance(obj, dict):
        out = {k: _clone(v, seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        out = [_clone(v, seen) for v in obj]
    elif isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        out = tuple(_clone(v, seen) for v in obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = obj.__class__.__new__(obj.__class__)
        for f in dataclasses.fields(obj):
            object.__setattr__(out, f.name, _clone(getattr(obj, f.name), seen))
        for k, v in getattr(obj, "__dict__", {}).items():
            if not hasattr(out, k):
                object.__setattr__(out, k, _clone(v, seen))
    else:
        out = obj
    seen[id(obj)] = out
    return out


def _bank_contexts(a: int, n: int):
    """Forward contexts of the two banks, tokens [0, a) and [a, n)."""
    from vllm.forward_context import create_forward_context
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.ubatch_utils import _slice_input_batch
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    st = _State.step
    ms = st["model_state"]
    if _State.bufs is None:
        _State.bufs = [
            (torch.zeros(2, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda"))
            for _ in range(2)
        ]
    ctxs = []
    for b, toks in enumerate((slice(0, a), slice(a, n))):
        sl = UBatchSlice(slice(0, 1), toks)
        sub = _slice_input_batch(st["input_batch"], sl, *_State.bufs[b])
        slots = st["slot_mappings"][:, toks]
        md = _ORIG_PREPARE(
            ms, sub, st["cg_mode"], tuple(t[0:1] for t in st["block_tables"]), slots,
            st["attn_groups"], st["kv_cache_config"],
        )
        md = _clone(md)
        ctxs.append(create_forward_context(
            md, ms.vllm_config, slot_mapping=build_slot_mappings_by_layer(slots, st["kv_cache_config"]),
        ))
    return ctxs


def _restore_step_metadata() -> None:
    """Rebuilds the step's own metadata in place (the banks' builds overwrote the
    model state's persistent buffers; the drafter reads them after the forward)."""
    st = _State.step
    _ORIG_PREPARE(
        st["model_state"], st["input_batch"], st["cg_mode"], st["block_tables"],
        st["slot_mappings"], st["attn_groups"], st["kv_cache_config"],
    )


def forward(model, input_ids, positions, query_start_loc, ngram_context, inputs_embeds=None,
            deepstack=None, profile=False):
    """Qwen4ExpModel.forward for one single-request prefill step, two banks.
    `profile`: a metadata-free dummy run (vLLM's memory profile) through the same
    eager path, so the profiled activation peak is this path's."""
    from contextlib import nullcontext

    from vllm.forward_context import override_forward_context

    rt = _State.rt
    cfg = model.config
    n = positions.shape[-1]
    a = _split(n)
    rows = (slice(0, a), slice(a, n))
    layers = model.layers

    wake = rt.kick(_State.wake_layer, _State.layers[_State.wake_layer], _State.wake_x, 0)
    if inputs_embeds is None:
        inputs_embeds = model.embed_input_ids(input_ids)
    hidden = inputs_embeds.repeat(1, cfg.hc_count)
    ds = [] if deepstack is None else [deepstack[f"deepstack_input_embeds_{i}"] for i in range(len(deepstack))]
    aux_out: list[list[torch.Tensor]] = []

    # Layers 0 and 1 (PLE) on all rows, the step's own context.
    bo = inj = None
    first_bank_layer = 2
    for idx in range(first_bank_layer):
        layer = layers[idx]
        hidden, bo, inj = layer(
            hidden_states=hidden, prev_block_output=bo, prev_injection=inj, positions=positions,
            input_ids=input_ids, query_start_loc=query_start_loc, ngram_context=ngram_context,
        )
        if idx < len(ds):
            e = ds[idx].unsqueeze(-2).expand(-1, cfg.hc_count, cfg.hidden_size).flatten(-2)
            hidden = layer.mlp_hyper_connection.combine(hidden, bo, inj) + e
            bo = inj = None
        if idx + 1 in model.aux_hidden_state_layers:
            if bo is not None:
                hidden = layer.mlp_hyper_connection.combine(hidden, bo, inj)
                bo = inj = None
            aux_out.append([hidden.unflatten(-1, (cfg.hc_count, cfg.hidden_size)).mean(dim=-2)])

    ctx = [None, None] if profile else _bank_contexts(a, n)
    pos = [positions[..., r] for r in rows]
    # Per bank: multi-stream state, pending (block output, injection), and the
    # pre-mixed next-layer input with its NPU kick when there is one.
    hs = [hidden[r] for r in rows]
    bo = [None if bo is None else bo[r] for r in rows]
    inj = [None if inj is None else inj[r] for r in rows]
    pre = [None, None]  # (block_input, npu_seq or None, npu_rows)

    def premix(b, layer, kick):
        hc = layer.attn_hyper_connection
        if bo[b] is not None:
            hs[b], bi, inj[b] = hc.combine_and_mix(hs[b], bo[b], inj[b])
        else:
            hs[b], bi, inj[b] = hc.mix(hs[b])
        bo[b] = None
        seq, k = None, 0
        if kick and layer.layer_idx in _State.layers:
            k = _npu_rows(bi.shape[0])
            if k:
                seq = rt.kick(layer.layer_idx, _State.layers[layer.layer_idx], bi[:k], b)
        pre[b] = (bi, seq, k)

    # The wake chain ran during layers 0-1; retire it before bank 0 is reused.
    rt.result(_State.layers[_State.wake_layer], _WAKE_ROWS, wake, 0)
    rt.retire(wake, 0)
    # Layer 2's inputs: bank B to the NPU now, bank A projected on the GPU.
    for b in (0, 1):
        premix(b, layers[first_bank_layer], kick=(b == 1))

    for idx in range(first_bank_layer, len(layers)):
        layer = layers[idx]
        nxt = layers[idx + 1] if idx + 1 < len(layers) else None
        stack = ds[idx] if idx < len(ds) else None
        for b in (0, 1):
            bi, seq, k = pre[b]
            pre[b] = None
            given = None
            if seq is not None:
                given = _assemble(layer, rt, _State.layers[idx], seq, b, k, bi)
            with nullcontext() if ctx[b] is None else override_forward_context(ctx[b]):
                attn_out = _run_mixer(layer, bi, pos[b], given)
                hs[b], b_in, inj[b] = layer.mlp_hyper_connection.combine_and_mix(hs[b], attn_out, inj[b])
                bo[b] = layer.mlp(b_in)
            if stack is not None:
                e = stack[rows[b]].unsqueeze(-2).expand(-1, cfg.hc_count, cfg.hidden_size).flatten(-2)
                hs[b] = layer.mlp_hyper_connection.combine(hs[b], bo[b], inj[b]) + e
                bo[b] = inj[b] = None
            if idx + 1 in model.aux_hidden_state_layers and bo[b] is not None:
                hs[b] = layer.mlp_hyper_connection.combine(hs[b], bo[b], inj[b])
                bo[b] = inj[b] = None
            if nxt is not None:
                premix(b, nxt, kick=True)
        if idx + 1 in model.aux_hidden_state_layers:
            # The pre-mix keeps the combined state as hs, so this is the layer's output.
            aux_out.append([x.unflatten(-1, (cfg.hc_count, cfg.hidden_size)).mean(dim=-2) for x in hs])

    final = model.hyper_connection_mixer
    multi, sample = [], []
    for b in (0, 1):
        if bo[b] is not None:
            m_h, s_h, _ = final.combine_and_mix(hs[b], bo[b], inj[b])
        else:
            m_h, s_h, _ = final.mix(hs[b])
        multi.append(m_h)
        sample.append(s_h)
    if model._mtp_hidden_buffer is not None:
        model._mtp_hidden_buffer[:a].copy_(multi[0])
        model._mtp_hidden_buffer[a:n].copy_(multi[1])
    if not profile:
        _restore_step_metadata()
    # A failed NPU chain still releases the GPU's wait, so its C rows may be
    # garbage. Check once the step's work is done; on a failure never return
    # these outputs, and keep later steps on the GPU (the GDN state of this step
    # has already moved, so it cannot simply be recomputed here).
    torch.cuda.current_stream().synchronize()
    try:
        rt.check()
    except RuntimeError as e:
        _State.failed = True
        _State.rt = None
        _logger().error("libstrix: npu prefill off after a failed NPU chain: %s", e)
        raise
    sample_hidden = torch.cat(sample)
    if aux_out:
        return sample_hidden, [torch.cat(x) for x in aux_out]
    return sample_hidden


_ORIG_PREPARE = None


def register() -> None:
    global _ORIG_PREPARE
    if not enabled():
        return
    try:
        from vllm.models.qwen4_exp.amd import model as qm
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    except ImportError:
        return
    from vllm.config import CUDAGraphMode

    _ORIG_PREPARE = MambaHybridModelState.prepare_attn

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                     attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0, **kw):
        _State.step = None
        if not for_capture and cudagraph_mode != CUDAGraphMode.FULL:
            _State.step = dict(
                model_state=self, input_batch=input_batch, cg_mode=cudagraph_mode,
                block_tables=block_tables, slot_mappings=slot_mappings,
                attn_groups=attn_groups, kv_cache_config=kv_cache_config,
            )
        return _ORIG_PREPARE(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                             attn_groups, kv_cache_config, for_capture=for_capture,
                             ubatch_idx=ubatch_idx, **kw)

    MambaHybridModelState.prepare_attn = prepare_attn

    cls = qm.Qwen4ExpForConditionalGeneration
    orig = cls.forward

    def patched(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        inner = self.language_model.model
        n = positions.shape[-1]
        st = _State.step
        from vllm.forward_context import get_forward_context, is_forward_context_available

        long_step = (
            n >= _MIN_TOKENS
            and intermediate_tensors is None
            and input_ids is not None
            and not torch.cuda.is_current_stream_capturing()
            and not inner.use_hc_sequence_parallel
        )
        if long_step:
            # Open on the first long step, which is vLLM's startup profile run:
            # the NPU's pinned memory is then counted before the KV cache is sized.
            _runtime(inner)
        profile = (
            long_step
            and _State.rt is not None
            and is_forward_context_available()
            and get_forward_context().attn_metadata is None
        )
        if profile:
            return forward(inner, input_ids, positions, kwargs.get("query_start_loc"),
                           kwargs.get("ngram_context"), inputs_embeds, None, profile=True)
        if (
            long_step
            and st is not None
            and st["input_batch"].num_reqs == 1
            and st["input_batch"].num_tokens == n
            and intermediate_tensors is None
            and input_ids is not None
            and not torch.cuda.is_current_stream_capturing()
            and not inner.use_hc_sequence_parallel
            and _runtime(inner) is not None
        ):
            deepstack = None
            if inputs_embeds is not None:
                deepstack = self._get_deepstack_input_embeds(inputs_embeds.size(0))
            try:
                return forward(
                    inner, input_ids, positions, kwargs.get("query_start_loc"),
                    kwargs.get("ngram_context"), inputs_embeds, deepstack,
                )
            finally:
                if inputs_embeds is not None:
                    self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    cls.forward = patched
    _logger().info("libstrix: npu prefill patch installed (STRIX_ENABLE=npu)")
