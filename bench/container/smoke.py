# SPDX-License-Identifier: MIT
"""Start each recipe in the image and check it answers.

Per recipe: one text request; one image request (a red square) unless the
recipe is `--language-model-only` or the model takes no images; a tool call
when the recipe sets `--tool-call-parser`, and parsed reasoning when it sets
`--reasoning-parser`. Prints the startup memory numbers that go into
docs/docker.md. Run before pushing an image.

    python bench/container/smoke.py                       # every recipe
    python bench/container/smoke.py qwen3.8-27b-fp8.yaml muse-glimmer-30b-fp8.yaml
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib

from common import RECIPES, Server, parse_target, recipe_command

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def red_square_png(n: int = 512) -> str:
    row = lambda y: b"\x00" + b"".join(
        b"\xdc\x14\x14" if n // 5 <= x < 4 * n // 5 and n // 5 <= y < 4 * n // 5 else b"\xff\xff\xff"
        for x in range(n))
    raw = b"".join(row(y) for y in range(n))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def text_of(r: dict) -> str:
    return (r["choices"][0]["message"].get("content") or "") if "choices" in r else ""


def check(target: str) -> dict:
    recipe, env = parse_target(target)
    cmd = " ".join(recipe_command(recipe))
    res: dict = {"recipe": target}
    try:
        with Server(recipe, env) as s:
            res.update(s.startup_numbers())
            res["kernels"] = s.kernels()
            # 1024 tokens for every request: a checkpoint that always reasons first
            # (Muse-Glimmer has no switch to turn it off) needs room before its
            # answer, and Qwen3.6-35B-A3B FP8 thinks for ~600 tokens before a
            # tool call.
            t = text_of(s.chat("Reply with one word: hello.", max_tokens=1024))
            res["text"] = "ok" if "hello" in t.lower() else f"FAIL: {t[:80]!r}"
            if "--language-model-only" in cmd:
                res["image"] = "text only"
            else:
                r = s.chat([{"type": "image_url", "image_url": {"url": red_square_png()}},
                            {"type": "text", "text": "What color is the square? Answer in one word."}],
                           max_tokens=1024)
                if "error" in r and r["error"]["code"] == 400:
                    res["image"] = "text model"
                else:
                    t = text_of(r)
                    res["image"] = "ok" if "red" in t.lower() else f"FAIL: {t[:80]!r}"
            if "--tool-call-parser" in cmd:
                # The recipe's sampling defaults, as a chat client gets them:
                # Sage-A1.5 greedy reasons in circles and never calls the tool.
                # Sampled, a thinking model sometimes still runs out of tokens
                # first (Sage: 1 in 8), so up to 3 tries.
                for _ in range(3):
                    r = s.chat("What is the weather in Paris right now? Use the tool.",
                               max_tokens=1024, thinking=True, recipe_sampling=True,
                               tools=TOOLS, tool_choice="auto")
                    calls = r["choices"][0]["message"].get("tool_calls") if "choices" in r else None
                    good = bool(calls) and calls[0]["function"]["name"] == "get_weather" \
                        and "paris" in calls[0]["function"]["arguments"].lower()
                    if good:
                        break
                res["tool_call"] = "ok" if good else f"FAIL: {json.dumps(r)[:120]}"
            if "--reasoning-parser" in cmd:
                r = s.chat("What is 17 * 23?", max_tokens=1024, thinking=True)
                m = r["choices"][0]["message"] if "choices" in r else {}
                reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
                res["reasoning"] = "ok" if reasoning and "391" in (m.get("content") or "") \
                    else f"FAIL: {json.dumps(m)[:120]}"
    except RuntimeError as e:
        res["start"] = f"FAIL: {str(e).splitlines()[0]}"
        res["log_tail"] = str(e).splitlines()[-5:]
    res["pass"] = not any(str(v).startswith("FAIL") for v in res.values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="recipe.yaml[:ENV=VAL,...] (default: all)")
    ap.add_argument("--out")
    args = ap.parse_args()
    targets = args.targets or sorted(p.name for p in RECIPES.glob("*.yaml") if p.name != "common.yaml")
    results = []
    for t in targets:
        r = check(t)
        results.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    failed = [r["recipe"] for r in results if not r["pass"]]
    print(f"{len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
