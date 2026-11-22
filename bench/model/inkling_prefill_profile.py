"""GPU time per kernel for one Inkling 2 Mini prefill (offline, eager, no
prefix caching).

    IMAGE=strix-vllm:inkling MODELS=... VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        bench/container/run.sh python bench/model/inkling_prefill_profile.py
"""

import random
import time

from torch.profiler import ProfilerActivity, profile

from vllm import LLM, SamplingParams

llm = LLM(
    "/models/Inkling-2-Mini",
    max_model_len=16384,
    max_num_seqs=4,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    enable_prefix_caching=False,
)
words = "amber basil cedar delta ember fable garnet harbor indigo juniper".split()


def prompt(seed: int) -> list[dict]:
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(3000))
    return [{"role": "user", "content": "Summarize: " + text}]


params = SamplingParams(temperature=0, max_tokens=1)
llm.chat(prompt(1), params)  # compile
t0 = time.time()
out = llm.chat(prompt(2), params)
dt = time.time() - t0
n = len(out[0].prompt_token_ids)
print(f"prefill {n} tokens in {dt:.2f} s = {n / dt:.0f} tok/s (eager)")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    llm.chat(prompt(3), params)
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30,
                                max_name_column_width=80))
