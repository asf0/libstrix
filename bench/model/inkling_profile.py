"""GPU time per kernel for Inkling 2 Mini decode (offline, eager).

    IMAGE=strix-vllm:inkling MODELS=/mnt/8tb-usb/models bench/container/run.sh \
        python bench/model/inkling_profile.py
"""

import os
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
    disable_log_stats=False,
    # STRIX_DSPARK=n: DSpark drafts (the checkpoint's block of 7), n verified per step.
    speculative_config=(
        {
            "method": "dspark",
            "model": "/models/Inkling-2-Mini",
            "num_speculative_tokens": 7,
            "num_speculative_tokens_per_batch_size": [[1, 4, int(os.environ["STRIX_DSPARK"])]],
        }
        if os.environ.get("STRIX_DSPARK")
        else None
    ),
)
params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
prompt = [{"role": "user", "content": "Write a story about a lighthouse keeper."}]
llm.chat(prompt, params)  # warm up and compile

t0 = time.time()
out = llm.chat(prompt, params)
decode_s = time.time() - t0
print(f"32 tokens in {decode_s:.2f} s ({32 / decode_s:.1f} tok/s, eager)")

if os.environ.get("PROFILE_PREFILL"):
    words = " ".join(["amber basil cedar delta ember fable garnet harbor"] * 400)
    target = [{"role": "user", "content": "Summarize: " + words}]
    params = SamplingParams(temperature=0, max_tokens=1)
    llm.chat(target, params)  # compile the prefill shapes
    target = [{"role": "user", "content": "Summarize briefly: " + words}]
else:
    target = prompt
    params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    llm.chat(target, params)
print(
    prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=35, max_name_column_width=90
    )
)
rows = [e for e in prof.key_averages(group_by_input_shape=True) if e.key in ("aten::mm", "aten::addmm", "aten::linear", "aten::bmm", "aten::matmul")]
rows.sort(key=lambda e: -e.device_time_total)
for e in rows[:14]:
    print(f"SHAPE {e.key:12s} calls {e.count:5d} dev {e.device_time_total/1000:8.1f} ms  {e.input_shapes}")

for m in llm.get_metrics():
    if any(w in m.name for w in ("spec", "draft", "accept")):
        print("METRIC", m.name, getattr(m, "value", None) or getattr(m, "values", None))
