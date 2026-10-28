# SPDX-License-Identifier: MIT
"""attn_prefill32: D=512 prefill with 32 query columns per K/V stage.

Same reference and paged layout as test_attn_prefill; the prototype only
supports bf16 KV, causal, no window, single KV segment.
"""

import pytest
import strix
import torch
from test_attn_prefill import build, reference

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([32], [0]), ([16], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])],
)
@pytest.mark.parametrize("block_size", [16, 272])
def test_matches_reference(q_lens, ctx_lens, block_size):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 4, block_size, hd=512)
    scale = 512**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill32(out, q, kc, vc, cu, seq_lens, table, scale)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
