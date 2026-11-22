"""bmm_nvfp4 (vllm/models/inkling2/amd/ops/triton_bmm.py) against a torch
reference that decodes the same NVFP4 bytes. Run inside the image:

    python -m pytest -q bench/container/test_inkling2_nvfp4_bmm.py
"""

import pytest
import torch
from strix_vllm.inkling2_moe import gemm_mxfp8, gemm_nvfp4, pretile_nvfp4
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan, bmm, bmm_nvfp4

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def decode(weight, scale, amax, denominator):
    """[E, N, K // 2] uint8, [E, N, K // 16] e4m3 -> fp32 [E, N, K]."""
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    values = E2M1[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    blocks = scale.float().repeat_interleave(16, dim=-1)
    return values * blocks * (amax / denominator)[:, None, None]


@pytest.mark.parametrize("bf16_act", [False, True])
@pytest.mark.parametrize(
    "tile,rows",
    [
        (16, 5),
        (16, 16),
        (32, 23),
        (64, 40),
        (128, 50),
        (128, 100),
        (256, 150),
        (256, 200),
    ],
)
def test_bmm_nvfp4_matches_reference(tile, rows, bf16_act):
    torch.manual_seed(0)
    num_experts, n, k = 3, 256, 384
    weight = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8)
    scale = (torch.rand(num_experts, n, k // 16) * 2 + 0.1).to(torch.float8_e4m3fn)
    amax = torch.rand(num_experts) + 0.5
    denominator = torch.tensor([6.0 * 448.0])
    # Two tiles: expert 2 then expert 0, each with `rows` routed rows.
    experts = [2, 0]
    max_ctas = len(experts)
    plan = ExpertPlan(tile=tile, max_ctas=max_ctas)
    tiles = torch.zeros(2 * max_ctas + 1, dtype=torch.int32)
    for t, e in enumerate(experts):
        tiles[t] = e
        tiles[max_ctas + t] = t * tile + rows
    tiles[2 * max_ctas] = max_ctas
    act = (torch.randn(plan.rows, k) * 2).to(torch.float8_e4m3fn)
    act_scale = torch.full((plan.rows * k // 32,), 127, dtype=torch.uint8)

    dev = torch.device("cuda")
    gpu = [t.to(dev) for t in (weight, scale, amax, denominator, tiles)]
    if bf16_act:  # libstrix's gfx1151 kernels
        gpu[0], gpu[1] = pretile_nvfp4(gpu[0], gpu[1])
        out = gemm_nvfp4(act.to(torch.bfloat16).to(dev), *gpu, plan)
    else:  # the port's portable path
        out = bmm_nvfp4(act.to(dev), act_scale.to(dev), *gpu, plan)
    out = out.float().cpu()

    w = decode(weight, scale, amax, denominator)
    for t, e in enumerate(experts):
        sl = slice(t * tile, t * tile + rows)
        ref = act[sl].float() @ w[e].T
        torch.testing.assert_close(out[sl], ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@pytest.mark.parametrize("scale_lo", [120, 100])  # 100: outside mxfp8_moe_wmma's range
@pytest.mark.parametrize("bf16_act", [False, True])
@pytest.mark.parametrize(
    "tile,rows", [(16, 1), (16, 7), (64, 40), (128, 50), (128, 100), (256, 150)]
)
def test_bmm_mxfp8_decode_matches_reference(tile, rows, bf16_act, scale_lo):
    torch.manual_seed(0)
    num_experts, n, k = 3, 256, 384
    weight = (torch.randn(num_experts, n, k) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.randint(scale_lo, 130, (num_experts, n, k // 32), dtype=torch.uint8)
    experts = [2, 0]
    max_ctas = len(experts)
    plan = ExpertPlan(tile=tile, max_ctas=max_ctas)
    tiles = torch.zeros(2 * max_ctas + 1, dtype=torch.int32)
    for t, e in enumerate(experts):
        tiles[t] = e
        tiles[max_ctas + t] = t * tile + rows
    tiles[2 * max_ctas] = max_ctas
    act = (torch.randn(plan.rows, k) * 2).to(torch.float8_e4m3fn)
    act_scale = torch.full((plan.rows * k // 32,), 127, dtype=torch.uint8)
    dev = torch.device("cuda")
    gpu = [t.to(dev) for t in (weight, scale, tiles)]
    if bf16_act:
        out = gemm_mxfp8(act.to(torch.bfloat16).to(dev), *gpu, plan)
    else:
        out = bmm(act.to(dev), act_scale.to(dev), *gpu, plan)
    out = out.float().cpu()
    w = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
    for t, e in enumerate(experts):
        sl = slice(t * tile, t * tile + rows)
        ref = act[sl].float() @ w[e].T
        torch.testing.assert_close(out[sl], ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@pytest.mark.parametrize("tokens", [1, 4, 8])
@pytest.mark.parametrize("streams", [1, 3])
def test_loop_controllers_match_bmm(tokens, streams):
    from strix_vllm.inkling2_loop import controllers

    torch.manual_seed(0)
    dev = torch.device("cuda")
    k, rank = 4608, 64
    pair = torch.randn(streams, tokens, k, device=dev).to(torch.bfloat16)
    down = (torch.randn(streams, rank, k, device=dev) * 0.02).to(torch.bfloat16)
    up = (torch.randn(streams, k, rank, device=dev) * 0.05).to(torch.bfloat16)
    hidden = torch.bmm(pair, down.transpose(1, 2))
    ref = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
    got = controllers(pair, down, up)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@pytest.mark.parametrize("tokens", [17, 300])
@pytest.mark.parametrize("streams,broadcast", [(1, False), (3, True), (3, False)])
def test_blend_up_matches_blend_rms_norm(tokens, streams, broadcast):
    from strix_vllm.inkling2_loop import blend_up
    from vllm.models.inkling2.amd.ops.loop import blend_rms_norm

    torch.manual_seed(0)
    dev = torch.device("cuda")
    d, rank = 2304, 64
    if broadcast:  # local layers: one dest and norm weight for all streams
        dest = torch.randn(1, tokens, d, device=dev).expand(streams, -1, -1)
        norm_weight = (torch.rand(1, d, device=dev) + 0.5).expand(streams, -1)
    else:
        dest = torch.randn(streams, tokens, d, device=dev)
        norm_weight = torch.rand(streams, d, device=dev) + 0.5
    message = torch.randn(streams, tokens, d, device=dev)
    hidden = torch.randn(streams, tokens, rank, device=dev).to(torch.bfloat16)
    up = (torch.randn(streams, 2 * d, rank, device=dev) * 0.1).to(torch.bfloat16)
    gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
    ref = blend_rms_norm(dest, message, gain_forget, norm_weight, 1e-6)
    got = blend_up(dest, message, hidden, up, norm_weight, 1e-6)
    torch.testing.assert_close(got.float(), ref.float(), rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("rows", [1, 7, 16])
def test_nvfp4_tile_gemv_matches_reference(rows):
    torch.manual_seed(0)
    num_experts, n, k, tile = 3, 256, 384, 16
    weight = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8)
    scale = (torch.rand(num_experts, n, k // 16) * 2 + 0.1).to(torch.float8_e4m3fn)
    amax = torch.rand(num_experts) + 0.5
    denominator = torch.tensor([6.0 * 448.0])
    experts = [2, 0, 1]
    max_ctas = len(experts)
    plan = ExpertPlan(tile=tile, max_ctas=max_ctas)
    tiles = torch.zeros(2 * max_ctas + 1, dtype=torch.int32)
    for t, e in enumerate(experts):
        tiles[t] = e
        tiles[max_ctas + t] = t * tile + max(rows - t, 1)
    tiles[2 * max_ctas] = max_ctas
    act = torch.randn(plan.rows, k).to(torch.bfloat16)
    dev = torch.device("cuda")
    gpu = [t.to(dev) for t in (weight, scale, amax, denominator, tiles)]
    gpu[0], gpu[1] = pretile_nvfp4(gpu[0], gpu[1])
    out = gemm_nvfp4(act.to(dev), *gpu, plan).float().cpu()
    w = decode(weight, scale, amax, denominator)
    for t, e in enumerate(experts):
        sl = slice(t * tile, t * tile + max(rows - t, 1))
        ref = act[sl].float() @ w[e].T
        torch.testing.assert_close(out[sl], ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("tile,rows", [(64, 40), (128, 100), (256, 150)])
def test_wmma_swiglu_matches_gemm_then_swiglu(tile, rows, gather):
    from types import SimpleNamespace

    from strix_vllm.inkling2_moe import gemm_nvfp4_swiglu, swiglu_bf16

    torch.manual_seed(0)
    dev = torch.device("cuda")
    num_experts, n, k = 3, 512, 384
    weight = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8)
    scale = (torch.rand(num_experts, n, k // 16) * 2 + 0.1).to(torch.float8_e4m3fn)
    weight, scale = pretile_nvfp4(weight.to(dev), scale.to(dev))
    experts = SimpleNamespace(
        nvfp4=True,
        w13_weight=weight,
        w13_weight_scale=scale,
        w13_weight_amax=(torch.rand(num_experts) * 4 + 1).to(dev),
        w13_weight_outer_scale_denominator=torch.tensor([6.0 * 448.0], device=dev),
        gate_cap=4.0,  # Inkling 2 Mini's softcaps
        up_cap=25.0,
    )
    experts_order = [2, 0]
    max_ctas = len(experts_order)
    plan = ExpertPlan(tile=tile, max_ctas=max_ctas)
    tiles = torch.zeros(2 * max_ctas + 3, dtype=torch.int32)
    routed = []
    for t, e in enumerate(experts_order):
        tiles[t] = e
        tiles[max_ctas + t] = t * tile + rows
        routed += range(t * tile, t * tile + rows)
    tiles[2 * max_ctas] = max_ctas
    tiles = tiles.to(dev)
    act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
    gammas = torch.rand(plan.rows, device=dev) + 0.1
    routed = torch.tensor(routed, dtype=torch.int32, device=dev)
    num_routed = torch.tensor([routed.numel()], dtype=torch.int32, device=dev)

    gate_up = gemm_nvfp4(
        act,
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w13_weight_amax,
        experts.w13_weight_outer_scale_denominator,
        tiles,
        plan,
    )
    ref = swiglu_bf16(gate_up, gammas, 4.0, 25.0, (routed, num_routed), tile)
    if gather:  # layout rows read token rows of a smaller x
        tokens = torch.randn(37, k, device=dev).to(torch.bfloat16)
        row_token = torch.randint(0, 37, (plan.rows,), dtype=torch.int32, device=dev)
        act.copy_(tokens[row_token.long()])
        gate_up = gemm_nvfp4(
            act,
            experts.w13_weight,
            experts.w13_weight_scale,
            experts.w13_weight_amax,
            experts.w13_weight_outer_scale_denominator,
            tiles,
            plan,
        )
        ref = swiglu_bf16(gate_up, gammas, 4.0, 25.0, (routed, num_routed), tile)
        got = gemm_nvfp4_swiglu(tokens, experts, tiles, plan, gammas, row_token)
    else:
        got = gemm_nvfp4_swiglu(act, experts, tiles, plan, gammas)
    ref, got = ref[routed.long()].float(), got[routed.long()].float()
    # Same bf16 GEMM outputs in; the activation may differ by a bf16 ulp.
    torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=1e-3 * ref.abs().max())
