# SPDX-License-Identifier: MIT
"""Run a docker/recipes recipe the way users do, and talk to it over HTTP.

The container tests (smoke, spec_equiv, kv_pressure, gsm8k) start a recipe
with `docker compose` in their own compose project (`strixtest`), on their
own port, so they never touch containers someone else started. The image is
$IMAGE (default strix-vllm:test, see docker/build-test-image.sh); models come
from $MODELS, compile caches from $CACHE (default docker/recipes/cache).

A target is `recipe.yaml` optionally followed by env overrides:
`qwen3.6-35b-a3b-fp8.yaml:MTP=0,MAX_MODEL_LEN=16384`; a `+` layers an
overlay over its base: `qwen3.8-27b.yaml+npu/qwen3.8-27b.yaml`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "docker" / "recipes"
PROJECT = "strixtest"
PORT = int(os.environ.get("STRIX_TEST_PORT", "18200"))


def parse_target(target: str) -> tuple[str, dict[str, str]]:
    recipe, _, env = target.partition(":")
    over = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    return recipe, over


def recipe_command(recipe: str) -> list[str]:
    """The recipe's command list, as written (for flag checks); a layered
    target's command comes from its base recipe."""
    import yaml

    path = RECIPES / recipe.split("+")[0]
    svc = next(iter(yaml.safe_load(open(path))["services"].values()))
    return [str(c) for c in svc.get("command", [])]


def gpu_busy() -> list[str]:
    """Running containers that hold the GPU (other than our own project)."""
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Label \"com.docker.compose.project\"}}"],
        capture_output=True, text=True, check=True,
    ).stdout
    busy = []
    for line in out.splitlines():
        name, _, project = line.partition("\t")
        if project == PROJECT:
            continue
        devs = subprocess.run(
            ["docker", "inspect", "--format", "{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}", name],
            capture_output=True, text=True,
        ).stdout
        if "/dev/kfd" in devs:
            busy.append(name)
    return busy


class Server:
    """`with Server("recipe.yaml", {"MTP": "0"}) as s:` starts it, waits for
    /health and stops it on exit. Raises RuntimeError with the log tail if the
    container exits first."""

    def __init__(self, recipe: str, env: dict[str, str] | None = None, timeout: int = 2400):
        self.recipe = recipe
        self.timeout = timeout
        self.env = dict(os.environ)
        self.env.setdefault("IMAGE", "strix-vllm:test")
        self.env.setdefault("CACHE", str(RECIPES / "cache"))
        if "MODELS" not in self.env:
            sys.exit("set MODELS to the host directory holding the model folders")
        self.env["PORT"] = str(PORT)
        self.env.update(env or {})
        self.base = f"http://127.0.0.1:{PORT}"
        self.model = ""

    def _compose(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["docker", "compose", "-p", PROJECT]
        for part in self.recipe.split("+"):
            cmd += ["-f", str(RECIPES / part)]
        return subprocess.run(
            [*cmd, *args], env=self.env, capture_output=True, text=True, check=check,
        )

    def __enter__(self) -> Server:
        busy = gpu_busy()
        if busy and not os.environ.get("STRIX_TEST_FORCE"):
            sys.exit(f"GPU in use by {', '.join(busy)}; stop it or set STRIX_TEST_FORCE=1")
        self._compose("down", "--remove-orphans", check=False)
        self._compose("up", "-d")
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            try:
                urllib.request.urlopen(self.base + "/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                pass
            if not self._compose("ps", "-q").stdout.strip():
                lines = self.logs().splitlines()
                # The root cause sits above the engine-startup traceback that
                # ends the log: show the first error lines, then the tail.
                errors = [ln for ln in lines if re.search(r"\b\w*(Error|Exception)\b:", ln)][:6]
                tail = "\n".join(errors + ["..."] + lines[-12:])
                self.__exit__(None, None, None)
                raise RuntimeError(f"{self.recipe} exited during startup:\n{tail}")
            time.sleep(5)
        else:
            self.__exit__(None, None, None)
            raise RuntimeError(f"{self.recipe} not healthy after {self.timeout} s")
        self.model = self.get("/v1/models")["data"][0]["id"]
        return self

    def __exit__(self, *exc) -> None:
        self._compose("down", "-t", "60", check=False)

    def logs(self) -> str:
        return self._compose("logs", "--no-color", check=False).stdout

    def startup_numbers(self) -> dict[str, str | None]:
        text = self.logs()
        out = {}
        for key, pat in (
            ("weights_gib", r"Model loading took ([\d.]+) GiB"),
            ("kv_gib", r"Available KV cache memory: ([\d.]+) GiB"),
            ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens"),
            ("max_requests", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
        ):
            m = re.findall(pat, text)
            out[key] = m[-1] if m else None
        return out

    def kernels(self) -> list[str]:
        """The kernel choices the startup log reports (plugin and vLLM)."""
        seen = []
        for line in self.logs().splitlines():
            m = re.search(r"(Selected \S+ for \S+|libstrix: .*)", line)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
        return seen

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.load(r)

    def post(self, path: str, body: dict, timeout: int = 1800) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return {"error": {"code": e.code, "message": e.read().decode()[:500]}}

    # Plain greedy decoding: overrides every sampling default a recipe sets
    # (--override-generation-config), penalties included, since those change
    # the argmax too. Measurements compare runs token for token.
    GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
              "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}

    def chat(self, content, max_tokens: int = 512, thinking: bool = False,
             recipe_sampling: bool = False, **extra) -> dict:
        """recipe_sampling=True sends no sampling fields, so the request runs
        with the recipe's defaults, as a chat client's would."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            **({} if recipe_sampling else self.GREEDY),
            "chat_template_kwargs": {"enable_thinking": thinking},
            **extra,
        }
        return self.post("/v1/chat/completions", body)

    def metric(self, name: str) -> float:
        """Sum of a Prometheus counter/gauge over its label sets."""
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as r:
            text = r.read().decode()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        )


def tokens_and_margins(resp: dict) -> tuple[list[str], list[float]]:
    """Generated tokens and, per position, top-1 minus top-2 logprob (needs
    logprobs=True, top_logprobs>=2 in the request)."""
    content = resp["choices"][0]["logprobs"]["content"]
    toks, margins = [], []
    for c in content:
        toks.append(c["token"])
        tops = sorted((t["logprob"] for t in c.get("top_logprobs", [])), reverse=True)
        margins.append(tops[0] - tops[1] if len(tops) >= 2 else float("inf"))
    return toks, margins


def first_divergence(a: list[str], b: list[str]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
