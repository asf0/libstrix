"""Start one target, keep the full log on failure: python start_check.py target out.log"""

import os
import sys
from pathlib import Path

HARNESS = Path("/home/ubuntu/local-llm/worktrees/libstrix-vllm-main")
os.environ.setdefault("CACHE", "/home/ubuntu/local-llm/engines/libstrix/docker/recipes/cache")
sys.path.insert(0, str(HARNESS / "bench" / "container"))
from common import Server, parse_target  # noqa: E402


class LoggingServer(Server):
    def __exit__(self, *exc):
        Path(sys.argv[2]).write_text(self.logs())
        super().__exit__(*exc)


recipe, env = parse_target(sys.argv[1])
env.setdefault("GPU_UTIL", "0.80")
try:
    with LoggingServer(recipe, env, timeout=1500) as s:
        print("healthy", s.startup_numbers())
        print(s.chat("What is 17*23? Answer briefly.", max_tokens=64))
except RuntimeError as e:
    print("FAILED", str(e)[:300])
