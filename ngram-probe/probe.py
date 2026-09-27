#!/usr/bin/env python3
"""Throwaway probe: does an n-gram speculator draft on a given workload?

Sends one completion to a llama.cpp router and prints the response timings.
PROBE_HOST selects the router (default 127.0.0.1:8100); PROBE_PROXY is passed to
curl when the router is reached through a proxy, for example a sandbox SOCKS5
relay. The timings carry the aggregate draft numbers; the per-implementation
breakdown is printed at TRACE by server_slot::print_timings() ->
common_speculative_print_stats() at the end of every request, so read the server
log right after.

Usage:
    python3 ngram-probe/probe.py --model Qwen3.8-27B --kind source
    PROBE_HOST=<router-host>:8100 PROBE_PROXY=socks5h://<proxy>:1080 \
        python3 ngram-probe/probe.py --kind control
"""

import argparse
import json
import os
import subprocess
import sys

HOST = os.environ.get("PROBE_HOST", "127.0.0.1:8100")
PROXY = os.environ.get("PROBE_PROXY", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# files whose concatenation stands in for the "source tree" prompt
SOURCE_FILES = [
    "common/speculative.cpp",
    "common/ngram-map.cpp",
    "tools/server/server-context.cpp",
]

REVIEW_INSTRUCTION = (
    "\n\nReview the code above. List the concrete bugs and design problems, "
    "most important first. Be specific about file and function."
)


def read_source(budget_chars):
    parts = []
    used = 0
    for rel in SOURCE_FILES:
        path = os.path.join(REPO, rel)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if used + len(text) > budget_chars:
            text = text[: budget_chars - used]
        parts.append(text)
        used += len(text)
        if used >= budget_chars:
            break
    return "\n".join(parts)


def build_prompt(kind, budget_chars):
    if kind == "source":
        return read_source(budget_chars) + REVIEW_INSTRUCTION
    if kind == "control":
        # forced verbatim repetition: the generation tail must re-walk the block
        block = read_source(budget_chars)
        return (
            "Repeat the following text back to me exactly, character for "
            "character, with no commentary, no reformatting and no omissions. "
            "Start at the first character and continue to the end.\n\n"
            "===== BEGIN =====\n" + block + "\n===== END =====\n"
        )
    raise SystemExit(f"unknown kind: {kind}")


def ask(model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    cmd = ["curl", "-sS", "--max-time", str(timeout)]
    if PROXY:
        cmd += ["--proxy", PROXY]
    cmd += [
        "-H", "Content-Type: application/json",
        "-X", "POST",
        f"http://{HOST}/v1/chat/completions",
        "--data-binary", "@-",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, input=body)
    if cp.returncode != 0:
        raise SystemExit(f"curl failed ({cp.returncode}): {cp.stderr.strip()}")
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"not json: {cp.stdout[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--kind", choices=["source", "control"], default="source")
    ap.add_argument("--budget", type=int, default=8000,
                    help="prompt budget in characters (source excerpt)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dump", default=None, help="write the raw response here")
    args = ap.parse_args()

    prompt = build_prompt(args.kind, args.budget)
    print(f"kind={args.kind} prompt_chars={len(prompt)}", flush=True)

    res = ask(args.model, prompt, args.max_tokens, args.timeout)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"raw response -> {args.dump}")

    if "error" in res:
        raise SystemExit(f"server error: {json.dumps(res['error'])[:400]}")

    timings = res.get("timings", {})
    draft_n = timings.get("draft_n", 0)
    draft_ok = timings.get("draft_n_accepted", 0)
    ratio = (draft_ok / draft_n) if draft_n else 0.0

    print(json.dumps(timings, indent=2, sort_keys=True))
    print(f"draft_n={draft_n} draft_n_accepted={draft_ok} ratio={ratio:.4f}")
    preview = (res.get("choices", [{}])[0].get("message", {}).get("content")
               or res.get("content") or "")
    print(f"first 200 chars: {preview[:200]!r}")


if __name__ == "__main__":
    sys.exit(main())
