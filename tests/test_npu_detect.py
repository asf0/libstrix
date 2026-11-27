# SPDX-License-Identifier: MIT
"""When the NPU path turns on: detection, GPU_UTIL, STRIX_ENABLE / STRIX_DISABLE.
No NPU needed: the device path and the vLLM config are faked."""

import os
import resource
import types

import pytest

from strix_vllm import npu, qwen3_5_npu, qwen4_exp_npu


@pytest.fixture
def device(tmp_path, monkeypatch):
    """A readable, writable stand-in for /dev/accel/accel0 and the shim."""
    dev = tmp_path / "accel0"
    dev.write_bytes(b"")
    lib = tmp_path / "libstrix_npu.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(npu, "DEVICE", str(dev))
    monkeypatch.setenv("STRIX_NPU_LIB", str(lib))
    monkeypatch.setattr(resource, "getrlimit", lambda _: (resource.RLIM_INFINITY,) * 2)
    monkeypatch.delenv("STRIX_ENABLE", raising=False)
    monkeypatch.delenv("STRIX_DISABLE", raising=False)
    return dev


def _util(monkeypatch, util):
    import vllm.config

    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(gpu_memory_utilization=util))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)


@pytest.mark.parametrize("mod", [qwen4_exp_npu, qwen3_5_npu])
def test_on_when_the_device_is_there(device, mod):
    assert npu.unavailable() is None
    assert mod.enabled()


@pytest.mark.parametrize("mod", [qwen4_exp_npu, qwen3_5_npu])
def test_off_without_the_device_unless_forced(device, monkeypatch, mod):
    monkeypatch.setattr(npu, "DEVICE", str(device) + ".missing")
    assert "no " in npu.unavailable()
    assert not mod.enabled()
    monkeypatch.setenv("STRIX_ENABLE", "npu")
    assert mod.enabled()


@pytest.mark.parametrize("mod", [qwen4_exp_npu, qwen3_5_npu])
def test_disable_wins(device, monkeypatch, mod):
    monkeypatch.setenv("STRIX_ENABLE", "npu")
    monkeypatch.setenv("STRIX_DISABLE", "npu")
    assert not mod.enabled()
    monkeypatch.setenv("STRIX_DISABLE", "all")
    assert not mod.enabled()


def test_needs_the_shim(device, monkeypatch):
    def missing():
        raise FileNotFoundError("no shim")

    monkeypatch.setattr(npu, "_lib_path", missing)
    assert npu.unavailable() == "no shim"


def test_needs_unlimited_memlock(device, monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _: (8 << 20, 8 << 20))
    assert "memlock" in npu.unavailable()


def test_needs_read_write_access(device):
    os.chmod(device, 0o400)
    try:
        if os.access(device, os.W_OK):  # root ignores the mode
            pytest.skip("running as root")
        assert "readable and writable" in npu.unavailable()
    finally:
        os.chmod(device, 0o600)


@pytest.mark.parametrize(("util", "blocked"), [(0.80, False), (0.82, False), (0.85, True)])
def test_gpu_util_limit(device, monkeypatch, util, blocked):
    _util(monkeypatch, util)
    why = npu.util_blocks()
    assert (why is not None) == blocked
    if blocked:
        assert "0.85" in why and "STRIX_ENABLE=npu" in why


def test_forced_skips_the_util_limit(device, monkeypatch):
    _util(monkeypatch, 0.9)
    monkeypatch.setenv("STRIX_ENABLE", "npu")
    assert npu.util_blocks() is None


def test_no_config_no_limit(device, monkeypatch):
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    assert npu.util_blocks() is None


def test_hooks_the_class_the_registry_loads(device):
    """The NPU forward must wrap the Qwen3.5 class vLLM actually runs: the
    registry's entry, not a module the upstream code moved away from."""
    import importlib

    from vllm.model_executor.models.registry import ModelRegistry

    qwen3_5_npu.register()
    entry = ModelRegistry.models["Qwen3_5ForConditionalGeneration"]
    module, name = entry.module_name, entry.class_name
    cls = getattr(importlib.import_module(module), name)
    assert getattr(cls, "_strix_npu", False), f"{module}.{name} is not hooked"


def test_gemma4_is_opt_in(device, monkeypatch):
    """The experimental gemma-4 path stays off under the NPU's auto-detection."""
    from strix_vllm import gemma4_npu

    assert not gemma4_npu.enabled()
    monkeypatch.setenv("STRIX_ENABLE", "npu")
    assert not gemma4_npu.enabled()
    monkeypatch.setenv("STRIX_ENABLE", "npu_gemma")
    assert gemma4_npu.enabled()


def test_gemma4_hooks_the_classes_the_registry_loads(device, monkeypatch):
    """gemma-4's NPU forward wraps the class vLLM runs for Gemma4ForConditionalGeneration."""
    import importlib

    from vllm.model_executor.models.registry import ModelRegistry

    from strix_vllm import gemma4_npu

    monkeypatch.setenv("STRIX_ENABLE", "npu_gemma")
    gemma4_npu.register()
    entry = ModelRegistry.models["Gemma4ForConditionalGeneration"]
    module = entry.module_name if "." in entry.module_name else f"vllm.model_executor.models.{entry.module_name}"
    cls = getattr(importlib.import_module(module), entry.class_name)
    assert getattr(cls, "_strix_npu", False), f"{module}.{entry.class_name} is not hooked"
