# SPDX-License-Identifier: MIT
"""attn.register() defaults vLLM's extensible KV cache off: the vendored
Triton kernel's KV pieces would point at pages it maps only after warmup."""

import pytest

config = pytest.importorskip("vllm.config")
if not hasattr(config.VllmConfig, "_resolve_extensible_kv_cache"):
    pytest.skip("this vLLM has no extensible KV cache", allow_module_level=True)


@pytest.fixture
def patched(monkeypatch):
    # monkeypatch restores whatever resolver was installed before the test.
    monkeypatch.setattr(
        config.VllmConfig,
        "_resolve_extensible_kv_cache",
        config.VllmConfig._resolve_extensible_kv_cache,
    )
    from strix_vllm import attn

    attn._extensible_kv_cache_off()


def _resolve(requested):
    vc = config.VllmConfig()
    vc.cache_config._extensible_kv_cache_resolved = False
    vc.cache_config.enable_extensible_kv_cache = requested
    vc._resolve_extensible_kv_cache()
    return vc.cache_config.enable_extensible_kv_cache


def test_default_off(patched):
    assert _resolve(None) is False


def test_explicit_setting_stands(patched):
    assert _resolve(False) is False
    # An explicit True reaches vLLM's own checks unchanged (it may still turn
    # it off where unsupported, but not because of the plugin).
    from vllm.platforms import current_platform

    if current_platform.is_cuda_alike():
        assert _resolve(True) is True
