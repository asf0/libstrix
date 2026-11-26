# SPDX-License-Identifier: MIT
"""strix split-KV attention vs the served Triton launch, verify and decode shapes.

    bench/container/run.sh python bench/kernels/attn_verify.py \\
        --cases 35b,27b --ctx 2048,32768 --batch 1,4 --json /work/profiles/x.json

Triton runs the vendored `unified_attention` as vLLM calls it: 16 segments,
seq_threshold_3D = the batch, vLLM's own carving of the packed cache, the served
page size. Its launcher picks split-KV (3D) or 2D by its own rule (2D when a
sequence has more than 8 queries or queries x GQA > 64), so the baseline of a
16-query row is the 2D launch. The header of every point prints what was
launched. Strix runs attn_prefill(num_splits=N) for q > 1 and attn_decode for
q = 1, over every --splits value and the one strix_vllm.attn routes.

Serving fixes one split count per batch size when it captures its graphs, for
every context, so the best split of a point is not a speed a server reaches. The
summary gives, per (case, KV dtype, queries, batch), the speedup at each context
of one split: the one with the best worst case, the one with the best geometric
mean, and the routed one.

The cache is a pool in the served layout (LBNHC: [block][slot][head][K|V]), as
large as a served per-layer view, and blocks are placed at random over all of
it. Triton splits a view past 2 GiB into up to four pieces of 32-bit addressing;
a larger view (gemma-4-26B, served at ~9 GiB) runs whole with 64-bit addressing.
Rows with q > 1 (and prefill) use the page the recipe gets with its draft, q = 1
rows the page without one.

Each timed run goes through `layers` disjoint sets of blocks, at least 128 MiB
of KV where the pool has room, so a small context is not read from the caches.
Verify and decode run as captured CUDA graphs, as vLLM serves full decode
batches; prefill rows (--prefill-len) run eagerly, as vLLM runs them. Arms are
interleaved, each rep in a new order.

Outputs are checked against Triton's: RMS of the difference over RMS of Triton's
output at most --tol, and no element off by more than 10 x --tol x that RMS. They
are reset to NaN before the checked run, so it is the graph replay that is
checked. The exit status is 1 after a MISMATCH, 3 when an arm or a point was
skipped (a geometry the build lacks, a pool past --max-pool-gib) without
--allow-skips.

The pool is host RAM: at most --max-pool-gib, freed between cases.
"""

import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import strix
import strix_vllm
import torch
from strix_vllm import attn
from strix_vllm import triton_unified_attention as tua
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.core import kv_cache_planning
from vllm.v1.kv_cache_interface import KVQuantMode

GIB = 1 << 30
# vLLM's split-KV segment count off CUDA SM120 (NUM_PAR_SOFTMAX_SEGMENTS).
SEGMENTS = 16
# Queries per sequence vLLM sizes the split-KV scratch for (MAX_3D_QUERY_LEN).
SCRATCH_QUERIES = 8
# vLLM's default page.
DEFAULT_PAGE = 64
# The context graphs are captured with (max_model_len of most recipes).
CAPTURE_LEN = 131072
# Whether vLLM gives a draft model a KV pool of its own, as the v0.34 fork does.
# Without it all layers share one pool, and the target's pages are sized and
# padded for the draft's.
DRAFT_POOL = hasattr(kv_cache_planning, "_is_draft_attention_spec")
COLD_BYTES = 128 << 20
MAX_LAYERS = 64
KV_DTYPES = {"int8": torch.int8, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class Case:
    """One attention layer geometry and how its recipe serves it.

    view_gib: the served KV cache over the layers of a group, at GPU_UTIL 0.85.
    sibling, draft: (KV heads, head_dim) of the model's larger layers and of its
    draft model's. state, draft_state: bytes of one GDN state page of a hybrid,
    without and with the draft (ReplaySSM keeps more per draft token).
    """

    hq: int
    hkv: int
    hd: int
    queries: tuple[int, ...]
    view_gib: float
    window: int = -1
    sibling: tuple[int, int] | None = None
    draft: tuple[int, int] | None = None
    state: int = 0
    draft_state: int = 0
    sinks: bool = False


# State bytes: from the page padding vLLM logs with 7 drafts (35b, 27b) and 2
# (35b), linear in the drafts for 4; without a draft any value that gives the
# logged pages (2112 and 1600 on int8) does.
CASES = {
    "gemma26-full": Case(16, 2, 512, (16,), 8.1, sibling=(8, 256), draft=(8, 128)),
    "gemma31-full": Case(32, 4, 512, (16,), 7.6, sibling=(16, 256), draft=(8, 128)),
    "gemma26-sliding": Case(16, 8, 256, (16,), 8.1, window=1023, draft=(8, 128)),
    # Qwen3.6-35B-A3B with DFlash 2 (7 drafts), and plain decode.
    # DFlash drafts: 32 query / 8 KV heads, 2048-key window (Qwen, Muse, Flint).
    "dflash-draft": Case(32, 8, 128, (4, 8, 16), 1.0, window=2047),
    "dflash-draft-q4": Case(32, 8, 128, (4,), 1.0, window=2047),
    "35b": Case(
        16, 2, 256, (8, 1), 5.4, draft=(8, 128), state=2_170_000, draft_state=2_458_541
    ),
    # MiniCPM-V-4.7 with DSpark (4 drafts).
    "35b-dspark": Case(16, 2, 256, (5,), 5.4, draft=(2, 256), draft_state=2_335_000),
    "27b": Case(
        24, 4, 256, (8, 1), 3.9, draft=(8, 128), state=3_250_000, draft_state=3_614_120
    ),
    "flint-sliding": Case(
        32, 4, 128, (4,), 3.9, window=512, draft=(8, 128), sinks=True
    ),
    "flint-full": Case(32, 4, 128, (4,), 3.9, draft=(8, 128), sinks=True),
    "lfm": Case(32, 8, 64, (5, 1), 7.4, draft=(8, 64)),
    "muse-sliding": Case(32, 2, 128, (16, 8, 1), 4.3, window=2047, draft=(8, 128)),
    "muse-full": Case(32, 2, 128, (16, 8, 1), 4.3, draft=(8, 128)),
}


def slot_bytes(hkv: int, hd: int, kv: str) -> int:
    """Cache bytes of one token of one layer (int8 carries two fp32 scales)."""
    return 2 * hkv * (hd + 4 if kv == "int8" else 2 * hd)


def served_page(
    case: Case, kv: str, drafted: bool, draft_pool: bool = DRAFT_POOL
) -> tuple[int, int]:
    """(page tokens, block stride in bytes), by the rules of vLLM's
    _align_hybrid_block_size and unify_kv_cache_spec_page_size."""
    own = slot_bytes(case.hkv, case.hd, kv)
    sibling = slot_bytes(*case.sibling, kv) if case.sibling else 0
    draft = 0
    if drafted and case.draft and not draft_pool:
        draft = slot_bytes(*case.draft, kv)
    # A hybrid's block is as many tokens as its state page needs.
    basis = own
    if draft > own:
        common = math.lcm(own, draft)
        basis = common if common <= 2 * draft else draft
    state = case.draft_state if drafted else case.state
    block = DEFAULT_PAGE * max(1, -(-state // (DEFAULT_PAGE * basis)))
    # Smaller pages grow to the largest, then are padded to it.
    largest = max(own, sibling, draft)
    grow = largest // own
    if draft_pool and largest % own:
        grow = 1
    return block * grow, block * largest


def routed_splits(case: Case, kv: str, q: int, batch: int) -> dict[str, int]:
    """Splits strix_vllm.attn picks for this batch at capture, by kernel; none
    where it keeps the batch on Triton."""
    layer = attn.layer_kernels(
        case.hd, case.hq // case.hkv, case.sinks, case.window >= 0
    )
    kernel, splits = attn.route(
        layer, q, batch, kv, CAPTURE_LEN, case.window, case.hkv
    )
    return {kernel: splits} if splits else {}


def build_info() -> dict:
    """What this run measures: the kernels, the plugin and vLLM."""
    so = Path(strix.__file__).with_name("_C.so")
    return {
        "_C.so": str(so),
        "_C.so md5": hashlib.md5(so.read_bytes()).hexdigest(),
        "strix": strix.__file__,
        "strix_vllm": strix_vllm.__file__,
        "triton_unified_attention md5": hashlib.md5(
            Path(tua.__file__).read_bytes()
        ).hexdigest(),
        "vllm": importlib.metadata.version("vllm"),
        **{f.name: f.read_text().strip() for f in Path("/opt/strix").glob("*.ref")},
        "draft pool": DRAFT_POOL,
    }


def visible_keys(case: Case, seq_len: int, q: int) -> int:
    """Keys the q queries of one sequence read between them."""
    return min(seq_len, case.window + q) if case.window >= 0 else seq_len


def live_blocks(case: Case, seq_len: int, q: int, page: int) -> range:
    """Table entries a layer reads; a sliding layer's older ones are freed."""
    return range(
        (seq_len - visible_keys(case, seq_len, q)) // page, -(-seq_len // page)
    )


def make_pool(case: Case, kv: str, page: int, stride: int, blocks: int) -> torch.Tensor:
    """Random cache in vLLM's logical [B, H, N, C] over physical [B, N, H, C]."""
    dtype = KV_DTYPES[kv]
    raw = torch.empty(blocks * stride, dtype=torch.int8, device="cuda").view(dtype)
    if kv == "int8":
        raw.random_(-127, 128)
    else:
        raw.normal_()
    content = slot_bytes(1, case.hd, kv) // raw.element_size()
    return raw.as_strided(
        (blocks, case.hkv, page, content),
        (stride // raw.element_size(), content, case.hkv * content, 1),
    )


def carve(pool: torch.Tensor, kv: str):
    """K, V and scale views, as TritonAttentionImpl.forward makes them."""
    if kv == "bf16":
        k, v = pool.transpose(1, 2).split(pool.size(-1) // 2, dim=-1)
        return k, v, None, None
    impl = SimpleNamespace(
        _k_scale_cache=None,
        _v_scale_cache=None,
        _set_kv_ptr_range=False,
        _kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    impl._ensure_scale_caches = functools.partial(
        TritonAttentionImpl._ensure_scale_caches, impl
    )
    k, v = TritonAttentionImpl._pth_key_value_caches(impl, pool)
    return k, v, impl._k_scale_cache, impl._v_scale_cache


class LaunchSpy:
    """Passes launches on to a Triton kernel and keeps the last launch's shape."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.last: dict = {}

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.last = {
                "grid": tuple(int(g) for g in grid),
                "mode": "3D" if kwargs["IS_3D"] else "2D",
                "pieces": kwargs["KV_PIECES"],
                "BLOCK_M": kwargs["BLOCK_M"],
                "BLOCK_Q": kwargs["BLOCK_Q"],
                "TILE": kwargs["TILE_SIZE"],
                "segments": kwargs["NUM_SEGMENTS_PER_SEQ"],
                "num_warps": kwargs.get("num_warps", "default"),
            }
            return self.kernel[grid](**kwargs)

        return launch


class Point:
    """Inputs of one (case, KV dtype, queries, context, batch)."""

    def __init__(self, case, kv, page, pool, q, ctx, batch, layers, seed, sinks):
        self.case, self.kv, self.q, self.batch, self.layers = case, kv, q, batch, layers
        self.seq_len = ctx + q
        self.k, self.v, self.ks, self.vs = carve(pool, kv)
        if self.ks is not None:
            # Random bytes times this have unit variance.
            self.ks.fill_(1 / 73)
            self.vs.fill_(1 / 73)
        live = live_blocks(case, self.seq_len, q, page)
        gen = torch.Generator("cuda").manual_seed(seed)
        perm = torch.randperm(pool.size(0) - 1, generator=gen, device="cuda")
        perm = (perm[: layers * batch * len(live)] + 1).int()
        # Block 0 is vLLM's null block, where freed table entries point.
        self.tables = torch.zeros(
            layers, batch, live.stop, dtype=torch.int32, device="cuda"
        )
        self.tables[:, :, live.start :] = perm.view(layers, batch, len(live))
        self.queries = torch.randn(
            batch * q, case.hq, case.hd, generator=gen, device="cuda"
        ).bfloat16()
        self.cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * q
        self.lens = torch.full((batch,), self.seq_len, dtype=torch.int32, device="cuda")
        self.scale = case.hd**-0.5
        self.sinks = None
        if case.sinks and sinks:
            # About as much softmax weight as all the keys together.
            self.sinks = math.log(visible_keys(case, self.seq_len, q)) + torch.randn(
                case.hq, generator=gen, device="cuda"
            )
        self.kv_bytes = (
            batch
            * visible_keys(case, self.seq_len, q)
            * slot_bytes(case.hkv, case.hd, kv)
        )

    def outputs(self) -> list[torch.Tensor]:
        return [torch.zeros_like(self.queries) for _ in range(self.layers)]

    def triton(self):
        """(run, outputs) of the launch TritonAttentionImpl.forward makes."""
        c = self.case
        rows = self.batch * min(self.q, SCRATCH_QUERIES)
        shape = (rows, c.hq, SEGMENTS)
        segm_out = torch.empty((*shape, c.hd), dtype=torch.float32, device="cuda")
        segm_max = torch.empty(shape, dtype=torch.float32, device="cuda")
        segm_sum = torch.empty(shape, dtype=torch.float32, device="cuda")
        descale = None
        mode = KVQuantMode.INT8_PER_TOKEN_HEAD
        if self.kv == "bf16":
            mode = KVQuantMode.NONE
            descale = torch.ones((), device="cuda").expand(self.batch, c.hkv)
        outs = self.outputs()

        def run():
            for out, table in zip(outs, self.tables):
                tua.unified_attention(
                    q=self.queries,
                    k=self.k,
                    v=self.v,
                    out=out,
                    cu_seqlens_q=self.cu,
                    max_seqlen_q=self.q,
                    seqused_k=self.lens,
                    max_seqlen_k=self.seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=(c.window, 0) if c.window >= 0 else (-1, -1),
                    block_table=table,
                    softcap=0,
                    q_descale=None,
                    k_descale=descale,
                    v_descale=descale,
                    seq_threshold_3D=self.batch,
                    num_par_softmax_segments=SEGMENTS,
                    softmax_segm_output=segm_out,
                    softmax_segm_max=segm_max,
                    softmax_segm_expsum=segm_sum,
                    sinks=self.sinks,
                    kv_quant_mode=mode,
                    k_scale_cache=self.ks,
                    v_scale_cache=self.vs,
                )

        return run, outs

    def strix(self, splits: int):
        """(run, outputs) of the call strix_vllm.attn makes, with these splits."""
        outs = self.outputs()

        def run():
            for out, table in zip(outs, self.tables):
                if self.q == 1:
                    strix.attn_decode(
                        out, self.queries, self.k, self.v, self.lens, table,
                        self.scale, splits, self.case.window, self.ks, self.vs,
                        self.sinks,
                    )  # fmt: skip
                else:
                    strix.attn_prefill(
                        out, self.queries, self.k, self.v, self.cu, self.lens, table,
                        self.scale, splits, self.case.window, self.ks, self.vs,
                        self.sinks,
                    )  # fmt: skip

        return run, outs


def capture(run, graph_pool):
    """A replay of `run` as one CUDA graph (its temporaries share graph_pool)."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=graph_pool):
        run()
    return graph.replay


def sample_us(run, iters: int, layers: int) -> float:
    """Time of one layer's attention, after one unmeasured run."""
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    run()
    a.record()
    for _ in range(iters):
        run()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters / layers * 1e3


def disagreement(ref: list[torch.Tensor], got: list[torch.Tensor]):
    """(RMS error, largest error) over the RMS of ref; inf for NaN or inf."""
    a, b = torch.stack(ref).float(), torch.stack(got).float()
    rms = a.square().mean().sqrt()
    if not (b.isfinite().all() and rms > 0):
        return float("inf"), float("inf")
    err = (a - b).abs()
    return float(err.square().mean().sqrt() / rms), float(err.max() / rms)


def arm_splits(pt: Point, args, use_graph: bool) -> list[int]:
    """--splits and the routed counts; prefill runs unsplit."""
    if not use_graph:
        return [1]
    routed = routed_splits(pt.case, pt.kv, pt.q, pt.batch).values()
    return sorted({*args.splits, *routed})


def refused(e: RuntimeError) -> bool:
    """Whether the op says this build has no kernel for the call."""
    return "not built" in str(e) or "sinks need num_splits" in str(e)


def measure(pt: Point, args, spy: LaunchSpy, use_graph: bool):
    """(rows, Triton launch, refused arms by reason) of one point; Triton is
    the first row."""
    arms = {"triton": pt.triton()}
    for n in arm_splits(pt, args, use_graph):
        arms[f"strix/{n}"] = pt.strix(n)
    graph_pool = torch.cuda.graph_pool_handle() if use_graph else None
    runs, errors, launch = {}, {}, {}
    skipped: dict[str, list[str]] = {}
    for name, (run, outs) in arms.items():
        try:
            # Triton compiles its kernel on the first call.
            for _ in range(2):
                run()
        except RuntimeError as e:
            if name == "triton" or not refused(e):
                raise
            skipped.setdefault(str(e).splitlines()[0], []).append(name)
            continue
        if name == "triton":
            launch = dict(spy.last)
        torch.cuda.synchronize()
        runs[name] = capture(run, graph_pool) if use_graph else run
        # The warm-up runs wrote the outputs too: check this run's alone.
        for out in outs:
            out.fill_(float("nan"))
        runs[name]()
        torch.cuda.synchronize()
        errors[name] = disagreement(arms["triton"][1], outs)
    iters = {}
    for name, run in runs.items():
        one = sample_us(run, 1, pt.layers) * pt.layers
        iters[name] = min(50, max(1, round(args.sample_ms * 1e3 / one)))
    times = {name: [] for name in runs}
    for rep in range(args.reps):
        order = list(runs)
        random.Random(rep).shuffle(order)
        for name in order:
            times[name].append(sample_us(runs[name], iters[name], pt.layers))
    rows = []
    for name, ts in times.items():
        rms, worst = errors[name]
        us = statistics.median(ts)
        rows.append(
            {
                "arm": name,
                "us": us,
                "min_us": min(ts),
                "max_us": max(ts),
                "vs_triton": statistics.median(times["triton"]) / us,
                "gb_s": pt.kv_bytes / us / 1e3,
                "rms_err": rms,
                "max_err": worst,
                "ok": rms <= args.tol and worst <= 10 * args.tol,
                "iters": iters[name],
                "samples_us": ts,
            }
        )
    return rows, launch, skipped


def summarize(points: list[dict]) -> list[dict]:
    """Per (case, KV dtype, queries, batch): what one split count gives at every
    context measured. Arms that are missing or wrong at any of them are out."""
    groups: dict[tuple, dict[int, dict]] = {}
    for pt in points:
        if pt["graph"]:
            key = (pt["case"], pt["kv"], pt["q"], pt["batch"])
            groups.setdefault(key, {})[pt["ctx"]] = {
                int(r["arm"].split("/")[1]): r["vs_triton"]
                for r in pt["rows"]
                if r["arm"] != "triton" and r["ok"]
            }
    out = []
    for (name, kv, q, batch), by_ctx in groups.items():
        ctxs = sorted(by_ctx)
        usable = set.intersection(*(set(by_ctx[c]) for c in ctxs))

        def pick(splits, ctxs=ctxs, by_ctx=by_ctx, usable=usable):
            if splits not in usable:
                return {"splits": splits, "speedup": None}
            speedup = [by_ctx[c][splits] for c in ctxs]
            return {
                "splits": splits,
                "speedup": speedup,
                "worst": min(speedup),
                "geomean": statistics.geometric_mean(speedup),
            }

        picks = {}
        if usable:
            every = [pick(n) for n in sorted(usable)]
            picks["best worst case"] = max(every, key=lambda r: r["worst"])
            picks["best geomean"] = max(every, key=lambda r: r["geomean"])
        for rule, n in routed_splits(CASES[name], kv, q, batch).items():
            picks[rule] = pick(n)
        out.append(
            {"case": name, "kv": kv, "q": q, "batch": batch, "ctx": ctxs, **picks}
        )
    return out


def print_summary(run: dict) -> None:
    print("\n== summary: speedup over Triton of one split count at every context")
    for s in run["summary"]:
        print(
            f"{s['case']} {s['kv']} q{s['q']} b{s['batch']}  ctx "
            f"{' / '.join(map(str, s['ctx']))}"
        )
        for rule, pick in s.items():
            if not isinstance(pick, dict):
                continue
            line = "not measured (refused, wrong or left out)"
            if pick["speedup"]:
                each = " / ".join(f"{x:.2f}x" for x in pick["speedup"])
                line = f"{each}  worst {pick['worst']:.2f}x geo {pick['geomean']:.2f}x"
            print(f"   {rule:>15} splits {pick['splits']:>2}: {line}")
    if run["skipped"]:
        print("\n== skipped")
        for s in run["skipped"]:
            print(f"{s['point']}: {s['what']}: {s['why']}")


def report(point: str, kv_bytes, layers, blocks, page, stride, launch, rows, skipped):
    view = blocks * stride
    note = ""
    if launch["pieces"] == 1 and view > 2**31 - 1:
        note = " (whole view, 64-bit)"
    print(
        f"\n== {point}: page {page}, block stride {stride} B, view "
        f"{view / GIB:.2f} GiB, {blocks} blocks, {layers} layers, KV "
        f"{kv_bytes / 1e6:.1f} MB\n   triton {launch['mode']} pieces "
        f"{launch['pieces']}{note} BLOCK_M {launch['BLOCK_M']} BLOCK_Q "
        f"{launch['BLOCK_Q']} TILE {launch['TILE']} segments {launch['segments']} "
        f"warps {launch['num_warps']} grid {launch['grid']}"
    )
    for why, names in skipped.items():
        print(f"   SKIPPED {', '.join(names)}: {why}")
    print(f"   {'arm':>9} {'us':>9} {'min..max':>19} {'speedup':>8} {'GB/s':>6} err")
    for r in rows:
        bad = ""
        if not r["ok"]:
            bad = f"  MISMATCH rms {r['rms_err']:.4f} max {r['max_err']:.4f}"
        spread = f"{r['min_us']:.0f}..{r['max_us']:.0f}"
        print(
            f"   {r['arm']:>9} {r['us']:9.1f} {spread:>19} {r['vs_triton']:7.2f}x "
            f"{r['gb_s']:6.0f} {r['rms_err']:.4f}{bad}",
            flush=True,
        )


def run_case(name: str, kv: str, args, spy: LaunchSpy, run: dict) -> None:
    """Adds every point of one case and KV dtype to `run`."""
    case = CASES[name]
    queries = [q for q in case.queries if not args.queries or q in args.queries]
    points = [(q, c, b, True) for q in queries for c in args.ctx for b in args.batch]
    if args.prefill_len:
        points += [(args.prefill_len, c, 1, False) for c in args.ctx]
    # One pool per page layout: drafted rows and plain decode differ.
    for drafted in (True, False):
        mine = [p for p in points if (p[0] > 1) == drafted]
        if mine:
            run_points(name, kv, drafted, mine, args, spy, run)
            gc.collect()
            torch.cuda.empty_cache()


def run_points(name, kv, drafted, points, args, spy, run) -> None:
    case = CASES[name]
    page, stride = served_page(case, kv, drafted)
    if args.page:
        page, stride = args.page, args.page * slot_bytes(case.hkv, case.hd, kv)
    cap = int(args.max_pool_gib * GIB) // stride
    view = min(cap, int((args.view_gib or case.view_gib) * GIB) // stride)

    plan = []
    for q, ctx, batch, use_graph in points:
        point = f"{name} {kv} q{q} ctx {ctx} b{batch}"
        per_layer = batch * len(live_blocks(case, ctx + q, q, page))
        if 1 + per_layer > cap:
            need = (1 + per_layer) * stride / GIB
            why = f"needs {need:.2f} GiB of KV (--max-pool-gib)"
            print(f"\n== {point}: SKIPPED, {why}")
            run["skipped"].append({"point": point, "what": "point", "why": why})
            continue
        # Enough KV per run to miss the caches, where the pool has it.
        bytes_ = batch * visible_keys(case, ctx + q, q)
        bytes_ *= slot_bytes(case.hkv, case.hd, kv)
        layers = min(MAX_LAYERS, -(-COLD_BYTES // bytes_)) if use_graph else 1
        layers = min(layers, (max(view, 1 + per_layer) - 1) // per_layer)
        blocks = max(view, 1 + layers * per_layer)
        plan.append((point, q, ctx, batch, use_graph, layers, blocks))
    if not plan:
        return

    pool = make_pool(case, kv, page, stride, max(p[-1] for p in plan))
    for i, (point, q, ctx, batch, use_graph, layers, blocks) in enumerate(plan):
        pt = Point(
            case, kv, page, pool[:blocks], q, ctx, batch, layers, i, not args.no_sinks
        )
        rows, launch, skipped = measure(pt, args, spy, use_graph)
        sinks = " with sinks" if pt.sinks is not None else ""
        report(
            point + sinks, pt.kv_bytes, layers, blocks, page, stride, launch, rows,
            skipped,
        )  # fmt: skip
        for why, names in skipped.items():
            run["skipped"].append({"point": point, "what": " ".join(names), "why": why})
        run["points"].append(
            {
                "case": name,
                "kv": kv,
                "q": q,
                "ctx": ctx,
                "batch": batch,
                "page": page,
                "block_stride": stride,
                "sinks": pt.sinks is not None,
                "view_bytes": blocks * stride,
                "kv_bytes": pt.kv_bytes,
                "layers": layers,
                "graph": use_graph,
                "triton": launch,
                "rows": rows,
            }
        )
        run["summary"] = summarize(run["points"])
        if args.json:
            Path(args.json).write_text(json.dumps(run, indent=1))
        # The graphs hold their temporaries: drop them before the next point.
        del pt, rows
        gc.collect()
        torch.cuda.empty_cache()


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=",".join(CASES), help="comma list")
    p.add_argument("--ctx", type=ints, default=[2048, 32768, 131072])
    p.add_argument("--batch", type=ints, default=[1, 4, 8])
    p.add_argument("--kv", default="int8", help="int8, bf16 or both, comma list")
    p.add_argument(
        "--queries", type=ints, help="only these of each case's queries per sequence"
    )
    p.add_argument(
        "--splits", type=ints, default=[1, 4, 7, 12, 15, 20, 28, 32, 40, 50, 60]
    )
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--sample-ms", type=float, default=20, help="time per sample")
    p.add_argument(
        "--prefill-len",
        type=int,
        default=0,
        help="also price a prefill chunk of this many queries, batch 1, unsplit",
    )
    p.add_argument(
        "--no-sinks",
        action="store_true",
        help="leave Flint's sinks out on both sides (builds whose split reduce "
        "refuses them)",
    )
    p.add_argument(
        "--allow-skips",
        action="store_true",
        help="exit 0 although an arm or a point was skipped",
    )
    p.add_argument(
        "--page", type=int, help="page size instead of the served one, not padded"
    )
    p.add_argument("--view-gib", type=float, help="KV view size instead of the case's")
    p.add_argument("--max-pool-gib", type=float, default=8.2)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--json", help="write the build, every row and the summary here")
    args = p.parse_args()
    unknown = set(args.cases.split(",")) - set(CASES)
    if unknown or set(args.kv.split(",")) - set(KV_DTYPES):
        p.error(f"cases: {', '.join(CASES)}; kv: int8, bf16")
    if not strix.is_supported_device():
        p.error("needs gfx1151")
    if not tua._ON_GFX1151:
        p.error("the Triton arm is not on its gfx1151 settings")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)

    run: dict = {
        "build": build_info(),
        "args": vars(args),
        "points": [],
        "skipped": [],
        "summary": [],
    }
    for k, v in run["build"].items():
        print(f"{k}: {v}")
    # Twice the pool cap leaves room for split partials and graph memory.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 2 * args.max_pool_gib * GIB / total)
    )
    spy = tua.kernel_unified_attention = LaunchSpy(tua.kernel_unified_attention)
    for name in args.cases.split(","):
        for kv in args.kv.split(","):
            run_case(name, kv, args, spy, run)
    if args.json:
        Path(args.json).write_text(json.dumps(run, indent=1))
    print_summary(run)
    if not all(r["ok"] for pt in run["points"] for r in pt["rows"]):
        print("\nMISMATCH: at least one strix arm disagrees with Triton")
        sys.exit(1)
    if run["skipped"] and not args.allow_skips:
        print("\nSKIPPED arms or points (--allow-skips accepts them)")
        sys.exit(3)


if __name__ == "__main__":
    main()
