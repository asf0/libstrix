# SPDX-License-Identifier: MIT
"""Which attention kernel each recipe's batches run. No GPU or vLLM needed.

python/strix_vllm/attn.py imports only torch and strix_vllm.disabled at module
scope. It reads the kernel table through `import strix`, which loads the
compiled _C.so; the table itself (python/strix/registry.py) does not need it,
so the tests load that file on its own and stand it in for `strix`.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from strix_vllm import attn

RECIPES = Path(__file__).parents[1] / "docker" / "recipes"


def _load_registry():
    path = Path(attn.__file__).parents[1] / "strix" / "registry.py"
    spec = importlib.util.spec_from_file_location("_route_registry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks its module up there
    spec.loader.exec_module(module)
    return module


registry = _load_registry()


@pytest.fixture(autouse=True)
def table(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "strix", SimpleNamespace(geometries=registry.geometries)
    )
    monkeypatch.delenv("STRIX_DISABLE", raising=False)


# layer -> (query heads, KV heads, head dim, window_left, sinks)
LAYERS = {
    "gemma-26b-full": (16, 2, 512, -1, False),
    "gemma-26b-sliding": (16, 8, 256, 1023, False),
    "gemma-31b-full": (32, 4, 512, -1, False),
    "gemma-31b-sliding": (32, 16, 256, 1023, False),
    "muse-full": (32, 2, 128, -1, False),
    "muse-sliding": (32, 2, 128, 2047, False),
    "flint-full": (32, 4, 128, -1, True),
    "flint-sliding": (32, 4, 128, 512, True),
    "qwen3.6-35b": (16, 2, 256, -1, False),
    "qwen3.8-27b": (24, 4, 256, -1, False),
    "lfm2.5": (32, 8, 64, -1, False),
}

# The drafts' own attention (window 2047); their queries are one block
# (draft tokens + 1), never a prompt. run() routes them causal (Muse, gemma
# window layers); test_draft_attention covers the non-causal ones (Qwen / Flint
# DFlash 2, gemma's full layer, LFM's DSpark).
DRAFTS = {
    "dflash-draft": (32, 8, 128, 2047, False),
    "gemma-31b-draft": (64, 8, 128, 2047, False),
}

# recipe -> its attention layers
MODELS = {
    ("gemma-26b-full", "gemma-26b-sliding"): (
        "gemma-4-26b-a4b",
        "gemma-4-26b-a4b-awq-int4",
        "gemma-4-26b-a4b-fp8",
    ),
    ("gemma-31b-full", "gemma-31b-sliding"): (
        "gemma-4-31b-mxfp4",
        "gemma-4-31b-qat-int4",
    ),
    ("muse-full", "muse-sliding"): ("muse-glimmer-30b-fp8-dflash2",),
    ("flint-full", "flint-sliding"): ("flint-fp8",),
    ("qwen3.6-35b",): (
        "qwen3.6-35b-a3b",
        "qwen3.6-35b-a3b-w8a8",
        "qwen3.6-35b-a3b-fp8-dflash2",
        "sage-a1.5-fp8",
        "minicpm-v-4.7-35b-a3b-fp8",
    ),
    ("qwen3.8-27b",): (
        "qwen3.8-27b",
        "qwen3.8-27b-int8",
        "qwen3.8-27b-awq-int4",
        "qwen3.8-27b-awq-mxfp4",
        "qwen3.8-27b-fp8-dflash2",
        "qwen3.8-27b-uncensored-fp8-dflash2",
    ),
    ("lfm2.5",): ("lfm2.5-24b-a2b", "lfm2.5-24b-a2b-fp8-dspark"),
}
RECIPE_LAYERS = {r: layers for layers, rs in MODELS.items() for r in rs}

# Recipes whose attention does not go through attn.py: fp8 KV (no strix kernel
# takes it), and Inkling 2's own attention.
OTHER_RECIPES = {
    "qwen3.8-flash-next-dflash2",
    "inkling-2-mini",
    "bench-inkling-2-mini",
}

BATCHES = (1, 2, 4, 8)

# layer -> the kernel of a one-query batch at b1/b2/b4/b8 (no draft model), of
# a verify batch at the recipe's draft length, and of a full prefill chunk.
D, V, P, T = "decode", "verify", "prefill", "triton"
ROUTES = {
    "gemma-26b-full": ((D, D, D, D), V, P),
    "gemma-26b-sliding": ((D, D, D, D), V, P),
    "gemma-31b-full": ((D, D, D, D), V, P),
    "gemma-31b-sliding": ((D, D, D, D), V, P),
    "muse-full": ((D, D, D, D), V, P),
    "muse-sliding": ((D, D, D, D), V, P),
    "flint-full": ((D, D, D, D), V, P),
    "flint-sliding": ((D, D, D, D), V, P),
    "qwen3.6-35b": ((D, D, D, D), V, P),
    "qwen3.8-27b": ((D, D, D, D), V, P),
    "lfm2.5": ((D, D, D, D), V, P),
}

# Kernels (kernel, head dim, GQA, KV dtype) that no recipe's steady state runs: full
# prefill chunks, and decode steps at the recipe's draft length. A batch with a
# single query (a prompt's last chunk, a request's last token) still reaches
# the int8 decode rows.
BF16_KV = "every recipe serves int8 KV"
DRAFTED = "every recipe on this geometry drafts, so no step has one query"
NO_CALLER = "attn.py never calls it"
NO_PROMPT = "a draft has no prompt pass and no single-query step"
UNREACHED = {
    (P, 128, 4, "bf16"): BF16_KV,
    (P, 128, 4, "int8"): NO_PROMPT,
    (V, 128, 4, "bf16"): BF16_KV,
    (P, 64, 4, "bf16"): BF16_KV,
    (P, 128, 8, "bf16"): BF16_KV,
    (P, 128, 16, "bf16"): BF16_KV,
    (P, 256, 2, "bf16"): BF16_KV,
    (P, 256, 6, "bf16"): BF16_KV,
    (P, 256, 8, "bf16"): BF16_KV,
    (P, 512, 8, "bf16"): BF16_KV,
    (V, 128, 8, "bf16"): BF16_KV,
    (V, 128, 16, "bf16"): BF16_KV,
    (V, 256, 2, "bf16"): BF16_KV,
    (V, 256, 6, "bf16"): BF16_KV,
    (V, 256, 8, "bf16"): BF16_KV,
    (V, 512, 8, "bf16"): BF16_KV,
    (D, 64, 4, "bf16"): BF16_KV,
    (D, 128, 8, "bf16"): BF16_KV,
    (D, 128, 16, "bf16"): BF16_KV,
    (D, 256, 2, "bf16"): BF16_KV,
    (D, 256, 6, "bf16"): BF16_KV,
    (D, 256, 8, "bf16"): BF16_KV,
    (D, 512, 8, "bf16"): BF16_KV,
    (D, 128, 8, "int8"): DRAFTED,
    (D, 128, 16, "int8"): DRAFTED,
    (D, 256, 2, "int8"): DRAFTED,
    (D, 256, 6, "int8"): DRAFTED,
    (D, 256, 8, "int8"): DRAFTED,
    (D, 512, 8, "int8"): DRAFTED,
    ("prefill32", 512, 8, "bf16"): NO_CALLER,
}


def recipe(name):
    """The settings of a recipe that routing depends on."""
    text = (RECIPES / f"{name}.yaml").read_text()

    def setting(pattern, default=None):
        found = re.search(pattern, text)
        assert found or default is not None, f"{name}: no {pattern}"
        return found.group(1) if found else default

    kv = setting(r"--kv-cache-dtype=(\w+)")
    assert kv == "int8_per_token_head", f"{name}: {kv}"
    # A recipe that drafts must say how much, or it would pass as draft-free.
    speculates = "--speculative-config" in text
    return SimpleNamespace(
        kv="int8",
        queries=1
        + int(setting(r"SPEC_TOKENS:-(\d+)", default=None if speculates else "0")),
        chunk=int(setting(r"MAX_BATCHED_TOKENS:-(\d+)")),
        max_seq_len=int(setting(r"MAX_MODEL_LEN:-(\d+)")),
    )


def kernels_of(layer):
    hq, hkv, head, window_left, sinks = {**LAYERS, **DRAFTS}[layer]
    return attn.layer_kernels(head, hq // hkv, sinks, window_left >= 0)


def run(layer, queries, batch, max_seq_len=131072, kv="int8"):
    """route() for `batch` sequences of `queries` query tokens each."""
    _, hkv, _, window_left, _ = {**LAYERS, **DRAFTS}[layer]
    return attn.route(
        kernels_of(layer), queries, batch, kv, max_seq_len, window_left, hkv
    )


def test_every_recipe_is_covered():
    names = {p.stem for p in RECIPES.glob("*.yaml")} - {"common"}
    assert names == set(RECIPE_LAYERS) | OTHER_RECIPES
    assert not set(RECIPE_LAYERS) & OTHER_RECIPES
    assert set(ROUTES) == set(LAYERS) == {x for ls in MODELS for x in ls}


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize(
    "name,layer", [(r, x) for r, layers in RECIPE_LAYERS.items() for x in layers]
)
def test_recipe_routes(name, layer, batch):
    r = recipe(name)
    one_query, drafted, prefill = ROUTES[layer]
    one_query = one_query[BATCHES.index(batch)]

    assert run(layer, 1, batch, r.max_seq_len)[0] == one_query
    step = run(layer, r.queries, batch, r.max_seq_len)[0]
    assert step == (one_query if r.queries == 1 else drafted)
    assert run(layer, r.chunk, batch, r.max_seq_len)[0] == prefill


def table_kernels():
    """(kernel, head_dim, gqa, KV dtype) of every attention row."""
    where = {
        P: ("attn_prefill", {}),
        V: ("attn_prefill", {"split_kv": True}),
        D: ("attn_decode", {}),
        "prefill32": ("attn_prefill32", {}),
    }
    return {
        (kernel, *point, kv)
        for kernel, (op, extra) in where.items()
        for kv in ("bf16", "int8")
        for point in registry.geometries(op, "head_dim", "gqa", kv_dtype=kv, **extra)
    }


def test_unreached_kernels():
    reached = set()
    for name, layers in RECIPE_LAYERS.items():
        r = recipe(name)
        for layer in layers:
            hq, hkv, head, *_ = LAYERS[layer]
            for batch in BATCHES:
                for queries in (r.queries, r.chunk):
                    kernel, _ = run(layer, queries, batch, r.max_seq_len, r.kv)
                    reached.add((kernel, head, hq // hkv, r.kv))
    for draft, (hq, hkv, head, *_) in DRAFTS.items():
        for batch in BATCHES:
            for queries in (4, 8, 16):
                kernel, _ = run(draft, queries, batch)
                reached.add((kernel, head, hq // hkv, "int8"))
    reached = {k for k in reached if k[0] != T}
    assert reached <= table_kernels()
    assert table_kernels() - reached == set(UNREACHED)

    # The reasons hold: without a draft the int8 decode rows are reached, and
    # on bf16 KV every row but prefill32 is.
    for (kernel, head, gqa, kv), why in UNREACHED.items():
        if why in (NO_CALLER, NO_PROMPT):
            continue
        layer = next(
            x
            for x, (hq, hkv, d, *_) in {**LAYERS, **DRAFTS}.items()
            if (d, hq // hkv) == (head, gqa)
        )
        queries = {P: 8192, V: 8, D: 1}[kernel]
        assert run(layer, queries, 1, kv=kv)[0] == kernel, (kernel, head, gqa, kv)


def test_query_length_bounds():
    # 2..VERIFY_MAX_QUERY verify, then Triton up to PREFILL_MIN_QUERY.
    got = {q: run("qwen3.8-27b", q, 1)[0] for q in (0, 1, 2, 16, 17, 63, 64)}
    assert got == {0: T, 1: D, 2: V, 16: V, 17: T, 63: T, 64: P}


def test_draft_attention():
    # Causal window layers of the DFlash drafts: the verify kernel, one block
    # of queries per sequence.
    assert run("dflash-draft", 16, 1) == (V, 7)
    assert run("dflash-draft", 8, 1) == (V, 7)
    assert run("dflash-draft", 16, 8)[1] == 3
    assert run("dflash-draft", 8, 2)[1] == 5
    assert run("gemma-31b-draft", 16, 1)[0] == V
    # A non-causal full layer (gemma draft's last, LFM's DSpark) too, and only
    # without a window; its prompt-sized batches stay on Triton.
    full = attn.layer_kernels(128, 4, sinks=False, window=False)
    assert attn.route(full, 16, 1, "int8", 8192, -1, 8, causal=False)[0] == V
    assert attn.route(full, 1, 1, "int8", 8192, -1, 8, causal=False)[0] == T
    assert attn.route(full, 64, 1, "int8", 8192, -1, 8, causal=False)[0] == T
    # A window is fine too (Qwen / Flint DFlash 2 drafts).
    window = attn.layer_kernels(128, 4, sinks=False, window=True)
    assert attn.route(window, 16, 1, "int8", 8192, 2047, 8, causal=False) == (V, 7)
    assert attn.route(window, 1, 1, "int8", 8192, 2047, 8, causal=False)[0] == T
    assert attn.non_causal_window(-1) == -2
    assert attn.non_causal_window(2047) == -2050


def test_non_causal_rows():
    # route() sends non-causal draft batches to any split-KV row, so every
    # split-KV row must take non-causal batches.
    for row in registry.KERNELS:
        if row.op == "attn_prefill" and row.constraints.get("split_kv") is True:
            causal = row.constraints.get("causal")
            assert causal == (True, False), (row.name, row.constraints)


def test_disable_attn_decode(monkeypatch):
    monkeypatch.setenv("STRIX_DISABLE", "attn_decode")
    for layer, (_, _, prefill) in ROUTES.items():
        assert run(layer, 1, 1)[0] == T
        assert run(layer, 8192, 1)[0] == prefill


def test_sinks_and_window_come_from_the_table():
    # Only Flint's rows take sinks.
    both = {"bf16", "int8"}
    for window in (False, True):
        flint = attn.layer_kernels(128, 8, sinks=True, window=window)
        assert flint.prefill and flint.verify == flint.decode == both
    for head, gqa in ((64, 4), (128, 16), (256, 2), (256, 6), (256, 8), (512, 8)):
        got = attn.layer_kernels(head, gqa, sinks=True, window=False)
        assert not (got.prefill or got.verify or got.decode)
    # Decode at head 256 GQA 6 and 8 takes no window; the others do.
    for gqa in (6, 8):
        assert attn.layer_kernels(256, gqa, sinks=False, window=False).decode == both
        assert not attn.layer_kernels(256, gqa, sinks=False, window=True).decode
    for head, gqa in ((128, 8), (128, 16), (256, 2), (512, 8)):
        for window in (False, True):
            got = attn.layer_kernels(head, gqa, sinks=False, window=window)
            assert got.prefill and got.verify == got.decode == both
    # LFM: bf16 verify batches stay on Triton; decode takes no window.
    lfm = attn.layer_kernels(64, 4, sinks=False, window=False)
    assert lfm.prefill and lfm.verify == {"int8"} and lfm.decode == {"bf16", "int8"}
    assert not attn.layer_kernels(64, 4, sinks=False, window=True).decode
    assert run("lfm2.5", 5, 1, kv="bf16") == (T, 0)
    # A geometry without rows.
    got = attn.layer_kernels(128, 2, sinks=False, window=False)
    assert not (got.prefill or got.verify or got.decode)


def splits(layer, queries, kv="int8", batches=(1, 2, 4, 8, 16), **kw):
    return [run(layer, queries, b, kv=kv, **kw)[1] for b in batches]


def test_decode_split_counts():
    # The grid the splits were measured on: 2, 2, 4 and 8 KV heads.
    assert splits("qwen3.6-35b", 1) == [20, 10, 5, 7, 7]
    assert splits("qwen3.6-35b", 1, "bf16") == [20, 10, 5, 2, 1]
    assert splits("muse-full", 1) == [30, 20, 10, 5, 10]
    assert splits("muse-full", 1, "bf16") == [30, 25, 12, 6, 3]
    assert splits("flint-full", 1) == [20, 10, 5, 10, 10]
    assert splits("qwen3.8-27b", 1, "bf16") == [20, 10, 5, 2, 1]
    assert splits("lfm2.5", 1) == [10, 10, 5, 2, 1]
    assert splits("lfm2.5", 1, "bf16") == [10, 5, 2, 1, 1]
    assert splits("gemma-26b-sliding", 1) == [10, 5, 2, 1, 1]
    # A window splits its visible keys, not the sequence.
    assert splits("flint-sliding", 1, batches=(1,)) == [6]
    assert splits("gemma-26b-sliding", 1, batches=(1,), max_seq_len=300) == [4]
    assert splits("gemma-26b-full", 1, batches=(1,), max_seq_len=10) == [1]


def test_verify_split_counts():
    assert splits("muse-full", 16) == [15, 5, 5, 5, 5]
    assert splits("muse-sliding", 16) == [15, 5, 5, 2, 2]
    assert splits("gemma-26b-full", 16) == [10, 10, 5, 5, 3]
    assert splits("gemma-31b-full", 16) == [10, 5, 5, 3, 3]
    assert splits("gemma-26b-sliding", 16) == [5, 3, 2, 2, 2]
    assert splits("flint-full", 4) == [20, 15, 15, 10, 10]
    assert splits("flint-sliding", 4) == [7, 7, 7, 5, 5]
    assert splits("qwen3.6-35b", 8) == [30, 15, 15, 7, 7]
    assert splits("lfm2.5", 5) == [15, 20, 20, 15, 5]
    # The 27B's splits follow max_seq_len alone.
    assert splits("qwen3.8-27b", 8) == [32] * 5
    assert splits("qwen3.8-27b", 8, batches=(1,), max_seq_len=4096) == [4]
    assert splits("muse-full", 16, batches=(1,), max_seq_len=100) == [2]
    assert run("qwen3.8-27b", 8192, 1) == (P, 1)


def test_split_bounds():
    # The op takes 1..64 splits.
    for layer in LAYERS:
        for queries in (1, 2, 16):
            for kv in ("bf16", "int8"):
                for n in (0, 1, 64, 4096, 262144):
                    for batch in (0, 1, 3, 100, 1024):
                        kernel, s = run(layer, queries, batch, n, kv)
                        assert s == 0 if kernel == T else 1 <= s <= 64


def test_visible_keys():
    assert attn.visible_keys(5000, -1) == 5000
    assert attn.visible_keys(5000, 1023) == 1024
    assert attn.visible_keys(500, 1023) == 500
    # A verify tile sees the window of its first query up to its last.
    assert attn.visible_keys(5000, 1023, queries=16) == 1039
    assert attn.visible_keys(1030, 1023, queries=16) == 1030
