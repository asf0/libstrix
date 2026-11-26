# SPDX-License-Identifier: MIT
"""draft_head: FP8 copy of the target's LM head for a speculative draft.

A drafter that shares the target's LM head reads its bf16 weight (1.0-2.8 GB)
once per step, on top of the pass that verifies the step. With
`"quantization": "fp8_per_channel"` in the speculative config the draft's own
layers are FP8 already; this hook also gives the draft a private FP8 copy of
the shared head (vLLM's `quantized_lm_head_copy`, whose kernel is the strix FP8
one). DFlash drafts get no such copy from vLLM, DSpark ones only through
`draft_lm_head_quantization`, which it allows for DeepSeek-V4 drafts alone. The
target keeps its bf16 head, so every verified token is checked as before; only
which tokens the draft proposes can differ. Costs one byte per head weight.
STRIX_DISABLE=draft_head keeps the draft on the target's bf16 head.
"""

from strix_vllm import disabled


def _fp8_head_copy(model, target_model, vllm_config):
    """`model` with an FP8 copy of the target's LM head, if it shares that head
    and the draft is asked for in FP8."""
    spec = vllm_config.speculative_config
    if not str(spec.quantization or "").startswith("fp8"):
        return model
    inner = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head

    head = get_target_lm_head(target_model, inner)
    if head is None or getattr(model, "lm_head", None) is not head:
        return model
    from vllm.model_executor.layers.quantization.online.lm_head import (
        quantized_lm_head_copy,
    )

    model.lm_head = quantized_lm_head_copy(head, "fp8")
    return model


def register() -> None:
    if disabled("draft_head"):
        return
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash
    from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark

    for module, name in ((dflash, "load_dflash_model"), (dspark, "load_dspark_model")):
        load = getattr(module, name)
        if getattr(load, "strix_wrapped", False):
            continue

        def wrapped(target_model, vllm_config, _load=load):
            return _fp8_head_copy(
                _load(target_model, vllm_config), target_model, vllm_config
            )

        wrapped.strix_wrapped = True
        setattr(module, name, wrapped)
