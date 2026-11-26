"""Code the strix-vllm fork used to carry, now in the plugin: the dgpp_mixed
quantization config and the hyper-connection hook of Qwen3.8-Flash-Next."""

import types

import pytest
import torch

vllm = pytest.importorskip("vllm")


def test_dgpp_mixed_config():
    from strix_vllm import quantization
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    quantization.register()
    quantization.register()  # idempotent
    assert QUANTIZATION_METHODS.count("dgpp_mixed") == 1
    cls = get_quantization_config("dgpp_mixed")
    assert cls.get_name() == "dgpp_mixed"
    cfg = cls.from_config({"recipe": "fp8", "activations": "bfloat16"})
    assert cfg.weight_block_size == [128, 128]
    assert cfg.is_checkpoint_fp8_serialized
    with pytest.raises(ValueError):
        cls.from_config({"recipe": "int4"})
    with pytest.raises(ValueError):
        cls.from_config({"recipe": "fp8", "activations": "float16"})

    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    class FakeLinear(LinearBase):
        def __init__(self):
            torch.nn.Module.__init__(self)

    assert isinstance(cfg.get_quant_method(FakeLinear(), "x"), UnquantizedLinearMethod)


def _replicated(k, n):
    """A ReplicatedLinear, which needs vLLM's (one-process) parallel groups
    and a current config."""
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.model_executor.layers.linear import ReplicatedLinear

    with set_current_vllm_config(VllmConfig()):
        if not ps.model_parallel_is_initialized():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            ps.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0,
                backend="gloo",
            )
            ps.initialize_model_parallel(1)
        return ReplicatedLinear(
            k, n, bias=False, params_dtype=torch.bfloat16, quant_config=None,
            return_bias=False,
        )


def test_hc_hook_on_upstream_module(monkeypatch):
    from strix_vllm import qwen4_exp
    from vllm.model_executor.layers.linear import ReplicatedLinear

    calls = []

    class GatedResidual(torch.nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.input_mix_weight_up = _replicated(k, n)

    def hc_gate_mix(x, gate, hc_count):
        calls.append("orig")
        return gate

    hcm = types.SimpleNamespace(GatedResidual=GatedResidual, hc_gate_mix=hc_gate_mix)
    hc = types.SimpleNamespace()
    seen = []

    def fused(x, lowrank, weight, hc_count, unfused):
        seen.append((lowrank, weight, hc_count))
        return x[:, : x.shape[1] // hc_count] + 1

    monkeypatch.setattr(qwen4_exp, "_hc_up_gate_mix", fused)
    qwen4_exp._install_hc_hook(hcm, hc)

    g = GatedResidual(64, 256)
    up = g.input_mix_weight_up
    assert isinstance(up, ReplicatedLinear) and type(up) is not ReplicatedLinear
    up.weight.data = up.weight.data.cuda()
    lowrank = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    pair = up(lowrank)
    assert pair[0] is lowrank and pair[1] is up.weight

    x = torch.randn(3, 1024, dtype=torch.bfloat16, device="cuda")
    out = hcm.hc_gate_mix(x, pair, 4)  # through vllm.strix_hc_up_gate_mix
    torch.testing.assert_close(out, x[:, :256] + 1)
    assert len(seen) == 1 and seen[0][2] == 4
    gate = torch.randn(3, 1024, device="cuda")
    assert hcm.hc_gate_mix(x, gate, 4) is gate and calls == ["orig"]


def test_hc_hook_on_older_fork():
    from strix_vllm import qwen4_exp

    def unfused(x, lowrank, weight, hc_count):
        return "unfused"

    hc = types.SimpleNamespace(hc_up_gate_mix_impl=unfused)
    hcm = types.SimpleNamespace()
    qwen4_exp._install_hc_hook(hcm, hc)
    assert hc.hc_up_gate_mix_impl is not unfused
    x = torch.zeros(2, 64, dtype=torch.float32)  # not bf16: falls back
    assert hc.hc_up_gate_mix_impl(x, x[:, :16], x[:16], 4) == "unfused"
