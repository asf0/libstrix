# SPDX-License-Identifier: MIT
"""Does strix's prefill attention error grow with context length?

The KLD pass shows the plugin's Gemma attention drifting 0.020 nats at ~400
tokens and 0.199 KLD at 2k, on a model with a deterministic reference. The
window logic is per-row correct, so the remaining suspect is accumulation:
more key tiles means more online-softmax rescales of a bf16-cast partial.

This measures the kernel's own error against a float32 torch reference at
several context lengths, for the two Gemma geometries:
  (16 q / 8 kv, D=256)  sliding layers, GQA 2
  (32 q / 4 kv, D=512)  global layers,  GQA 8
If max|err| / RMS grows with ctx, the accumulator is the cause; if it stays
flat at bf16 rounding level (~2e-3 relative), the drift is inherent to bf16
attention and the KLD rows should be read that way.

    python tools/attn_precision.py [--iters 3]
"""

import sys
from pathlib import Path

import strix
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
from test_attn_prefill import build, reference


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.9)

    cases = [
        (16, 8, 256, (512, 1024, 2048, 4096)),
        (32, 4, 512, (512, 1024, 2048)),
    ]
    print(
        f"{'hq/kv':>7} {'D':>4} {'ctx':>6} {'out RMS':>8} {'max|err|':>9} "
        f"{'RMS err':>8} {'max/RMS':>8}"
    )
    for hq, hkv, hd, ctxs in cases:
        for ctx in ctxs:
            q, kc, vc, cu, seq_lens, table = build([ctx], [0], hq, hkv, 16, hd=hd)
            scale = hd**-0.5
            out = torch.empty_like(q)
            strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale)
            ref = reference(q, kc, vc, cu, seq_lens, table, scale)
            d = (out.float() - ref).abs()
            rms = ref.pow(2).mean().sqrt().item()
            print(
                f"{hq:3d}/{hkv:<3d} {hd:4d} {ctx:6d} {rms:8.4f} "
                f"{d.max().item():9.5f} {d.pow(2).mean().sqrt().item():8.5f} "
                f"{d.max().item() / rms:8.5f}",
                flush=True,
            )
            del q, kc, vc, out, ref, d
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
