# SPDX-License-Identifier: MIT
# gfx1151 shares host RAM with the GPU; an unbounded test can OOM the machine.
import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.1)
