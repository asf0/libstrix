# SPDX-License-Identifier: MIT
"""Qwen3.8-27B (Qwen3_5) prefill with the GDN input projections on the NPU.

STRIX_ENABLE=npu, same scheme as Qwen3.8-Flash-Next (qwen4_exp_npu): a
single-request prefill step of N >= STRIX_NPU_MIN_TOKENS tokens is split into two
row banks with their own attention / GDN metadata (bank B sees bank A as
context), and from layer 1 on each bank's layer runs while the NPU projects the
other bank's next layer.

Two differences from Flash-Next:

* **K = 5,120, the NPU program is K = 2,560.** Each projection runs as two chains
  over the K halves, from two runtimes (their arenas and relay threads are
  independent), and the two C halves are added in bf16. Each half quantizes its
  activations with its own per-token scale.
* **No hyper-connections, no PLE, a dense MLP.** The layer is Qwen3Next's:
  input_layernorm, mixer, post_attention_layernorm, MLP. The bank's next-layer
  norm is computed right after its MLP so the projection can be kicked early.

Only the GDN layers' `in_proj_qkvz` + `in_proj_ba` (16,480 of the shape's 16,640
columns) go to the NPU; full-attention layers stay on the GPU. Steps below the
token threshold, batches of several requests, graph capture and every step of a
model whose projections are not bf16 in the 27B's shape (K = 5,120) keep the
GPU path. Quantized checkpoints (FP8, INT8) are not covered, because
quantization already made their projections cheap; any other shape turns off
with the same message - the NPU only wins where the GPU's projection is an
expensive bf16 GEMM (docs/performance.md#npu-prefill).
The mixers
and the NPU calls run eagerly; the loop's dense regions (the input norm and
the post-mixer tail) are torch.compiled (STRIX_NPU_COMPILE=0 turns that off),
which recovers the Inductor fusion vLLM's own compiled path gets and without
which the path is slower than the compiled base (docs/performance.md#npu-prefill).
Two details make that compile work: vLLM's CustomOp layers dispatch to their
native implementations only when the compile config's `custom_ops` base mode
is `none`, and the IR ops they call (`vllm_ir::fused_add_rms_norm`) are opaque
to a plain torch.compile - vLLM lowers them with its own VllmIRLoweringPass -
so the compiled regions call them through the IR op's dispatch (`_inner_call`),
the traceable body Dynamo needs.
"""

from __future__ import annotations

import os

import torch

from strix_vllm import _env_names, disabled, npu
from strix_vllm import qwen4_exp_npu as shared

_MIN_TOKENS = int(os.environ.get("STRIX_NPU_MIN_TOKENS") or "4096")
_SIZES = tuple(sorted(int(x) for x in (os.environ.get("STRIX_NPU_SIZES") or "2048,3072,4096,6144,7168,8192").split(",")))
_ROWS_MAX = _SIZES[-1]
# The bank loop's dense regions (the input norm and the post-mixer tail: scale,
# post-attention norm, MLP, scale) are torch.compiled; the mixers and the NPU
# calls stay eager. Under torch.compile vLLM's CustomOp layers run their native
# IR decompositions, and the IR ops are called through `_inner_call` so Dynamo
# gets the traceable body (vLLM's own compile lowers the opaque `vllm_ir::*`
# op nodes with VllmIRLoweringPass, which a plain torch.compile does not run);
# Inductor then fuses the same kernels vLLM's compiled path fuses for this
# model. STRIX_NPU_COMPILE=0 keeps the whole loop eager.
_COMPILE = os.environ.get("STRIX_NPU_COMPILE", "1") != "0"


class _State:
    runtimes: list | None = None  # one per K half
    failed = False
    layers: set[int] = set()  # GDN layer indices resident on the NPU
    wake_x = None
    wake_layer = 0


def enabled() -> bool:
    return "npu" in _env_names("STRIX_ENABLE") and not disabled("npu")


def _logger():
    from vllm.logger import init_logger

    return init_logger("vllm.strix_vllm.qwen3_5_npu")


def _weight(layer) -> torch.Tensor | None:
    """The GDN layer's fused [qkvz | ba] projection weight, or None."""
    if getattr(layer, "layer_type", "") != "linear_attention":
        return None
    m = layer.linear_attn
    qkvz = getattr(m, "in_proj_qkvz", None)
    ba = getattr(m, "in_proj_ba", None)
    if qkvz is None or ba is None:
        return None
    if qkvz.weight.dtype != torch.bfloat16 or ba.weight.dtype != torch.bfloat16:
        return None
    w = torch.cat([qkvz.weight, ba.weight])
    if w.shape[1] != 2 * npu.K or w.shape[0] > 16640:
        return None
    return w


def _open(model) -> list | None:
    """Two runtimes, one per K half, with every GDN layer from 1 resident."""
    if _State.runtimes is not None or _State.failed:
        return _State.runtimes
    log = _logger()
    try:
        weights = {}
        for idx, layer in enumerate(model.layers):
            if idx == 0:
                continue
            w = _weight(layer)
            if w is not None:
                weights[idx] = w
        if not weights:
            raise RuntimeError("no bf16 GDN projection weights")
        runtimes = open_runtimes(weights)
        _State.layers = set(weights)
        _State.wake_layer = min(weights)
        _State.wake_x = torch.zeros((_SIZES[0], 2 * npu.K), dtype=torch.bfloat16, device="cuda")
        _State.runtimes = runtimes
        log.info(
            "libstrix: npu prefill on (Qwen3.5): %d GDN layers, two K halves, two banks of <= %d rows, from %d tokens",
            len(weights), _ROWS_MAX, _MIN_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        _State.failed = True
        import traceback
        if os.environ.get("STRIX_NPU_DEBUG", "") not in ("", "0"):
            _logger().warning("libstrix: npu prefill off: %s\n%s", e, traceback.format_exc())
        else:
            _logger().warning("libstrix: npu prefill off: %s", e)
    return _State.runtimes


def open_runtimes(weights: dict, pci: str | None = None) -> list:
    """One runtime per K half, `weights` = {layer index: [qkvz | ba] bf16 [n, 2*K]}."""
    pci = pci or shared._pci()
    runtimes = []
    for half in (0, 1):
        entries = []
        for idx, w in weights.items():
            w_half = w[:, half * npu.K: (half + 1) * npu.K].contiguous()
            entries.append((idx, npu.GDN_IN, w_half))
        runtimes.append(npu.NpuRuntime.open(_ROWS_MAX, entries, pci, banks=2))
    _State.layers = set(weights)
    for rt in runtimes:
        _warm(rt)
    return runtimes


def _warm(rt) -> None:
    """Runs every row-count program once on both banks, so their hardware
    contexts exist before a request (a variant's first chain takes ~180 ms)."""
    first = min(_State.layers)
    for rows in _SIZES:
        x = torch.zeros((rows, npu.K), dtype=torch.bfloat16, device="cuda")
        for bank in range(2):
            seq = rt.kick(first, npu.GDN_IN, x, bank)
            rt.result(npu.GDN_IN, rows, seq, bank)
            rt.retire(seq, bank)
    torch.cuda.synchronize()
    rt.check()


def _rows(k: int) -> int:
    return max((r for r in _SIZES if r <= k), default=0)


def _split(n: int) -> int:
    """Bank A's token count: the split that covers the most rows on the NPU
    (both banks rounded to programs the NPU has), the more balanced one on
    ties. The served engine chunks prefill at sizes that do not halve
    cleanly (a 8,192-token budget chunks at 8,080); a plain half-split
    leaves one bank too big for its program and the other too small to hide
    its chain."""
    half = n // 2
    best_a, best_cov, best_bias = half, -1, n
    for a in range(_SIZES[0], n - _SIZES[0] + 1, 1024):
        cov = _rows(a) + _rows(n - a)
        bias = abs(a - half)
        if cov > best_cov or (cov == best_cov and bias < best_bias):
            best_a, best_cov, best_bias = a, cov, bias
    return best_a


def _kick(x: torch.Tensor, layer_idx: int, bank: int) -> tuple:
    """Two chains over x's K halves, each with its own per-token activation scale
    (measured better than one scale over both halves: KLD 0.0104 against 0.0140);
    returns the two sequence numbers."""
    a, b = _State.runtimes
    return (
        a.kick(layer_idx, npu.GDN_IN, x[:, : npu.K], bank),
        b.kick(layer_idx, npu.GDN_IN, x[:, npu.K:], bank),
    )


def _assemble(ws: tuple, x: torch.Tensor, k: int, seqs: tuple, bank: int) -> tuple:
    """[in_proj_qkvz | in_proj_ba] for x: rows < k summed from the two NPU halves,
    the rest computed by the GPU straight into the result."""
    m = x.shape[0]
    out = []
    for w in ws:
        full = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
        if k < m:
            torch.mm(x[k:], w.t(), out=full[k:])
        out.append(full)
    a, b = _State.runtimes
    c0 = a.result(npu.GDN_IN, k, seqs[0], bank)
    c1 = b.result(npu.GDN_IN, k, seqs[1], bank)
    col = 0
    for full in out:
        w = full.shape[1]
        full[:k].copy_(c0[:, col: col + w])
        full[:k].add_(c1[:, col: col + w])
        col += w
    a.retire(seqs[0], bank)
    b.retire(seqs[1], bank)
    return tuple(out)


class _Given(torch.nn.Module):
    """Stand-in projection returning the tensors set for this call."""

    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):  # noqa: ARG002
        out, self.out = self.out, None
        assert out is not None, "NPU stand-in called without a projection"
        return out, None


def _mixer(layer, normed, positions, given):
    if layer.layer_type == "linear_attention":
        m = layer.linear_attn
        if given is None:
            return m(hidden_states=normed)
        qkvz, ba = m.in_proj_qkvz, m.in_proj_ba
        m.in_proj_qkvz, m.in_proj_ba = _Given(), _Given()
        m.in_proj_qkvz.out, m.in_proj_ba.out = given
        try:
            return m(hidden_states=normed)
        finally:
            m.in_proj_qkvz, m.in_proj_ba = qkvz, ba
    return layer.self_attn(hidden_states=normed, positions=positions)


def _scale(layer, x, kind: str):
    """The layer's optional post-attn / post-FFN scale."""
    if not getattr(layer, "layer_scale", False):
        return x
    s = getattr(layer, f"{kind}_layer_scale").to(x.dtype)
    return x * (s[0] + 1 if x.dim() == 2 else s + 1)


def _head_fn(layer, hidden, residual):
    """The next layer's input norm (the projection's input)."""
    if residual is None:
        return layer.input_layernorm(hidden), hidden
    return layer.input_layernorm(hidden, residual)


def _tail_fn(layer, hidden, residual):
    """The layer's dense tail: mixer scale, post-attention norm, MLP, FFN scale."""
    hidden = _scale(layer, hidden, "attn")
    hidden, residual = layer.post_attention_layernorm(hidden, residual)
    hidden = _scale(layer, layer.mlp(hidden), "ffn")
    return hidden, residual


def _norm(mod, x, residual):
    """GemmaRMSNorm.forward_native with the IR op unwrapped: vLLM's own compile
    lowers opaque `vllm_ir::*` op nodes through VllmIRLoweringPass, which a plain
    torch.compile does not run, so the op is called through `_inner_call` (the
    dispatch + implementation body) to give Dynamo the traceable native body."""
    from vllm import ir

    w = mod.weight.float() + 1.0
    if residual is None:
        return ir.ops.rms_norm._inner_call(x, w, mod.variance_epsilon), x
    return ir.ops.fused_add_rms_norm._inner_call(x, residual, w, mod.variance_epsilon)


def _compiled_tail_fn(layer, hidden, residual):
    hidden = _scale(layer, hidden, "attn")
    hidden, residual = _norm(layer.post_attention_layernorm, hidden, residual)
    mlp = layer.mlp
    gate_up, _ = mlp.gate_up_proj(hidden)
    act = mlp.act_fn.forward_native(gate_up)
    out, _ = mlp.down_proj(act)
    return _scale(layer, out, "ffn"), residual


def _compiled_head_fn(layer, hidden, residual):
    return _norm(layer.input_layernorm, hidden, residual)


_head = torch.compile(_compiled_head_fn, dynamic=False) if _COMPILE else _head_fn
_tail = torch.compile(_compiled_tail_fn, dynamic=False) if _COMPILE else _tail_fn


def _layer_body(layer, normed, residual, positions, given):
    """Qwen3NextDecoderLayer.forward after its input_layernorm: the mixer runs
    eagerly (custom ops, the NPU stand-ins, the bank's forward context), the
    dense tail through the (optionally compiled) `_tail`."""
    hidden = _mixer(layer, normed, positions, given)
    if not _COMPILE:
        return _tail_fn(layer, hidden, residual)
    return _tail(layer, hidden, residual)


def _pre_norm(layer, hidden, residual):
    if residual is None:
        return layer.input_layernorm(hidden), hidden
    return layer.input_layernorm(hidden, residual)


def forward(model, input_ids, positions, inputs_embeds=None):
    """Qwen3_5Model.forward for one single-request prefill step, two banks."""
    from vllm.forward_context import override_forward_context

    n = positions.shape[-1]
    a = _split(n)
    rows = (slice(0, a), slice(a, n))
    layers = model.layers

    rt_a = _State.runtimes[0]
    wake = rt_a.kick(_State.wake_layer, npu.GDN_IN, _State.wake_x[:, : npu.K], 0)
    hidden = model.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
    # Layer 0 on all rows, with the step's own metadata.
    h, r = layers[0](positions=positions, hidden_states=hidden, residual=None)
    rt_a.result(npu.GDN_IN, _SIZES[0], wake, 0)
    rt_a.retire(wake, 0)

    ctx = shared._bank_contexts(a, n)
    pos = [positions[..., sl] for sl in rows]
    state = [(h[rows[b]], None if r is None else r[rows[b]]) for b in (0, 1)]
    pre = [None, None]  # (normed input, residual, NPU seqs, NPU rows) per bank

    def premix(b, layer):
        """Norm of `layer`'s input for bank b, and that projection's NPU kick."""
        hidden_b, res = state[b]
        normed, res = _head(layer, hidden_b, res)
        k = _rows(normed.shape[0]) if layer.layer_idx in _State.layers else 0
        pre[b] = (normed, res, _kick(normed[:k], layer.layer_idx, b) if k else None, k)

    for b in (0, 1):
        premix(b, layers[1])

    for idx in range(1, len(layers)):
        layer = layers[idx]
        nxt = layers[idx + 1] if idx + 1 < len(layers) else None
        for b in (0, 1):
            normed, res, seqs, k = pre[b]
            pre[b] = None
            given = None if seqs is None else _assemble(
                (layer.linear_attn.in_proj_qkvz.weight, layer.linear_attn.in_proj_ba.weight),
                normed, k, seqs, b,
            )
            with override_forward_context(ctx[b]):
                state[b] = _layer_body(layer, normed, res, pos[b], given)
            if nxt is not None:
                premix(b, nxt)

    sample = []
    for b in (0, 1):
        hidden_b, res = state[b]
        out = model.norm(hidden_b, res)[0] if res is not None else model.norm(hidden_b)
        sample.append(out)
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
    return torch.cat(sample)


def register() -> None:
    if not enabled():
        return
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.model_executor.models import qwen3_5
    except ImportError:
        return
    inner_cls = getattr(qwen3_5, "Qwen3_5Model", None)
    if inner_cls is None:
        return
    for name in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        cls = getattr(qwen3_5, name, None)
        if cls is not None:
            _patch(cls)
    _logger().info("libstrix: npu prefill patch installed for Qwen3.5 (STRIX_ENABLE=npu)")


def _patch(cls) -> None:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    orig = cls.forward

    def patched(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        inner = self.language_model.model if hasattr(self, "language_model") else self.model
        n = positions.shape[-1]
        st = shared._State.step
        long_step = (
            n >= _MIN_TOKENS
            # The served engine with an MTP draft calls the target with the
            # embeddings pre-computed (input_ids None) and a placeholder
            # intermediate_tensors; the bank loop takes inputs_embeds directly.
            # Non-first pipeline ranks pass neither ids nor embeds and stay
            # on the GPU path.
            and (input_ids is not None or inputs_embeds is not None)
            and not torch.cuda.is_current_stream_capturing()
        )
        if long_step:
            _open(inner)
        if (
            long_step
            and st is not None
            and st["input_batch"].num_reqs == 1
            and st["input_batch"].num_tokens == n
            and _State.runtimes is not None
            and not inner.use_sequence_parallel
            and not getattr(inner, "aux_hidden_state_layers", ())
            and shared._ORIG_PREPARE is not None
        ):
            return forward(inner, input_ids, positions, inputs_embeds)
        return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    cls.forward = patched
