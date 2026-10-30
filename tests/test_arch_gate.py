# SPDX-License-Identifier: MIT
"""ARCH_GATED swaps apply only to the model family they were measured on."""

import pytest

import strix_vllm

QFN = ["Qwen4ExpForConditionalGeneration"]
OTHERS = [
    ["Qwen3_5ForConditionalGeneration"],  # Qwen3.8-27B
    ["Qwen3_5MoeForConditionalGeneration"],  # Qwen3.6-35B-A3B
    ["Gemma4ForConditionalGeneration"],
    [],  # no vLLM config (e.g. a kernel benchmark)
]


@pytest.fixture
def arch(monkeypatch):
    monkeypatch.delenv("STRIX_DISABLE", raising=False)
    monkeypatch.delenv("STRIX_ENABLE", raising=False)

    def set_arch(archs):
        monkeypatch.setattr(strix_vllm, "_model_architectures", lambda: list(archs))

    return set_arch


@pytest.mark.parametrize("op", sorted(strix_vllm.ARCH_GATED))
def test_on_for_qfn(arch, op):
    arch(QFN)
    assert strix_vllm.model_gate(op)


@pytest.mark.parametrize("op", sorted(strix_vllm.ARCH_GATED))
@pytest.mark.parametrize("archs", OTHERS)
def test_off_elsewhere(arch, op, archs):
    arch(archs)
    assert not strix_vllm.model_gate(op)


@pytest.mark.parametrize("op", sorted(strix_vllm.ARCH_GATED))
def test_enable_overrides_arch(arch, monkeypatch, op):
    arch(OTHERS[0])
    monkeypatch.setenv("STRIX_ENABLE", f"something_else, {op}")
    assert strix_vllm.model_gate(op)


@pytest.mark.parametrize("op", sorted(strix_vllm.ARCH_GATED))
@pytest.mark.parametrize("value", ["all", "{op}"])
def test_disable_wins(arch, monkeypatch, op, value):
    arch(QFN)
    monkeypatch.setenv("STRIX_ENABLE", op)
    monkeypatch.setenv("STRIX_DISABLE", value.format(op=op))
    assert not strix_vllm.model_gate(op)
