"""GPU time per kernel for Inkling 2 Mini decode (offline, eager).

    IMAGE=strix-vllm:inkling MODELS=/mnt/8tb-usb/models bench/container/run.sh \
        python bench/model/inkling_profile.py
"""

import time

import torch
from torch.profiler import ProfilerActivity, profile

from vllm import LLM, SamplingParams

llm = LLM(
    "/models/Inkling-2-Mini",
    max_model_len=4096,
    max_num_seqs=4,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)
params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
prompt = [{"role": "user", "content": "Write a story about a lighthouse keeper."}]
llm.chat(prompt, params)  # warm up and compile

t0 = time.time()
out = llm.chat(prompt, params)
decode_s = time.time() - t0
print(f"32 tokens in {decode_s:.2f} s ({32 / decode_s:.1f} tok/s, eager)")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    llm.chat(prompt, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
print(
    prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=35, max_name_column_width=90
    )
)
