#!/usr/bin/env python3
"""Compare upstream adaptive KV streaming against this fork's MTP path.

For every context size and every server configuration, start a fresh server,
prefill a software-development corpus truncated to the requested context,
decode a fixed token count, and record prefill/decode throughput. Results are
written to CSV/JSON and a comparison graph.

Each configuration uses its own maximum phase-arena size that still loads on
the target GPU: upstream 3072 MiB, this fork 3264 MiB (the preset value).

Reuses the server/HTTP helpers from benchmark_kv_stream.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import benchmark_kv_stream as bks


ROOT = Path(__file__).resolve().parents[1]

CONFIGS = ("upstream", "fork")

SOURCE_SUFFIXES = (
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
    ".cu", ".cuh", ".py", ".sh", ".cmake", ".md", ".txt",
)

DEFAULT_INSTRUCTION = (
    "\n\nReview the preceding source code. List the main functions and "
    "describe what each one does. Point out any bugs or memory-safety issues "
    "you find.\n"
)

MTP_CAP_RE = re.compile(
    r"MTP KV pin = (\d+) pages \((\d+) tokens\), decode window = (\d+) pages"
)
MTP_EJECT_RE = re.compile(r"draft ejected, decode capacity = (\d+) pages/layer")
MTP_REENABLE_RE = re.compile(r"draft re-enabled")


def parse_context_list(value: str) -> list[int]:
    parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    contexts = [int(part) for part in parts]
    if not contexts or any(context <= 0 for context in contexts):
        raise argparse.ArgumentTypeError("context sizes must be positive")
    return contexts


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_corpus(root: Path, max_chars: int) -> tuple[str, str, int]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed in {root}: {result.stderr.strip()}")
    files = sorted(
        name for name in result.stdout.splitlines()
        if name.lower().endswith(SOURCE_SUFFIXES)
    )
    parts: list[str] = []
    total = 0
    used = 0
    for name in files:
        try:
            text = (root / name).read_text(errors="replace")
        except OSError:
            continue
        parts.append(text)
        total += len(text)
        used += 1
        if total >= max_chars:
            break
    blob = "\n".join(parts)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return blob, digest, used


def arena_for(config: str, args: argparse.Namespace) -> int:
    return args.upstream_arena_mib if config == "upstream" else args.fork_arena_mib


def server_argv(config: str, args: argparse.Namespace, context: int) -> list[str]:
    if config == "upstream":
        server = args.upstream_server
        spec_type = "none"
    else:
        server = args.fork_server
        spec_type = "draft-mtp"
    argv = [
        str(server),
        "-m", str(args.model),
        "--alias", "bench",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", str(args.n_ctx),
        "-fa", "on",
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-ngl", "all",
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--spec-type", spec_type,
        "-np", "1",
        "--no-mmproj",
        "--no-warmup",
        "--fit", "off",
        "--reasoning-format", "none",
        "--kv-stream-arena-mib", str(arena_for(config, args)),
    ]
    if config == "fork":
        argv += [
            "--spec-draft-n-max", str(args.spec_draft_n_max),
            "--model-draft", str(args.mtp_model),
            "--spec-draft-ngl", "all",
            "--kv-stream-spec-dynamic",
            "--kv-stream-spec-reenable-pages", "8",
            "--kv-stream-spec-stable-decodes", "4",
        ]
    return argv


class ServerHandle:
    def __init__(self, argv: list[str], args: argparse.Namespace, log_path: Path) -> None:
        self.port = args.port
        self.log_path = log_path
        self.log_file = log_path.open("wb")
        self.process: subprocess.Popen | None = None
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=Path(argv[0]).parent,
                env=bks.clean_server_env(args.cuda_visible_devices),
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self.log_file.close()
            raise
        deadline = time.monotonic() + args.startup_timeout
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.log_file.flush()
                tail = self.log_tail()
                self.stop()
                raise RuntimeError(f"server exited with status {status}: {tail}")
            try:
                health = bks.http_json(self.url("/health"), None, 2)
                if health.get("status") == "ok":
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{last_error}: {self.log_tail()}")

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def log_tail(self, lines: int = 30) -> str:
        try:
            return "\n".join(
                self.log_path.read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            except ProcessLookupError:
                pass
        if not self.log_file.closed:
            self.log_file.close()


def tokenize(server: ServerHandle, content: str, timeout: int) -> list[int]:
    response = bks.http_json(
        server.url("/tokenize"),
        {"content": content, "add_special": False},
        timeout,
    )
    tokens = response.get("tokens")
    if not tokens or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"unexpected tokenize response: {response}")
    return tokens


def load_or_tokenize(
    server: ServerHandle,
    content: str,
    cache_path: Path,
    timeout: int,
) -> list[int]:
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("sha256") == hashlib.sha256(content.encode()).hexdigest():
            return cached["tokens"]
    tokens = tokenize(server, content, timeout)
    cache_path.write_text(json.dumps({
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "tokens": tokens,
    }))
    return tokens


def run_point(
    server: ServerHandle,
    config: str,
    args: argparse.Namespace,
    context: int,
    corpus_tokens: list[int],
    suffix_tokens: list[int],
    log_path: Path,
) -> dict:
    need = context - args.decode_tokens
    prefix_count = need - len(suffix_tokens)
    if prefix_count < 0:
        raise RuntimeError("context is smaller than the instruction suffix")
    if prefix_count > len(corpus_tokens):
        raise RuntimeError(
            f"corpus has {len(corpus_tokens)} tokens, need {prefix_count}; "
            "increase --corpus-max-chars"
        )
    prompt = corpus_tokens[:prefix_count] + suffix_tokens
    response = bks.http_json(
        server.url("/completion"),
        {
            "prompt": prompt,
            "n_predict": args.decode_tokens,
            "ignore_eos": True,
            "cache_prompt": False,
            "temperature": 0,
            "seed": 1,
            "reasoning_format": "none",
            "response_fields": ["timings"],
        },
        args.request_timeout,
    )
    timings = response.get("timings") or {}
    if timings.get("predicted_n") != args.decode_tokens:
        raise RuntimeError(
            f"incomplete decode: expected {args.decode_tokens}, "
            f"received {timings.get('predicted_n')}"
        )
    log_text = log_path.read_text(errors="replace")
    cap = MTP_CAP_RE.search(log_text)
    return {
        "type": "measurement",
        "status": "ok",
        "config": config,
        "server": str(server_argv(config, args, context)[0]),
        "context": context,
        "n_ctx": args.n_ctx,
        "prompt_tokens": len(prompt),
        "decode_tokens": args.decode_tokens,
        "arena_mib": arena_for(config, args),
        "spec_type": "draft-mtp" if config == "fork" else "none",
        "prefill_tps": timings.get("prompt_per_second"),
        "decode_tps": timings.get("predicted_per_second"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "mtp_pin_pages": int(cap.group(1)) if cap else None,
        "mtp_window_pages": int(cap.group(3)) if cap else None,
        "mtp_ejected": bool(MTP_EJECT_RE.search(log_text)),
        "mtp_reenabled": bool(MTP_REENABLE_RE.search(log_text)),
    }


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def load_results(path: Path) -> dict[tuple[str, int], dict]:
    results: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return results
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "measurement" and row.get("status") == "ok":
            results[(row["config"], row["context"])] = row
    return results


def write_csv(path: Path, results: dict[tuple[str, int], dict]) -> None:
    fields = [
        "config", "context", "n_ctx", "prompt_tokens", "decode_tokens", "arena_mib",
        "prefill_tps", "decode_tps", "prompt_ms", "predicted_ms",
        "mtp_pin_pages", "mtp_window_pages", "mtp_ejected", "mtp_reenabled",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(results, key=lambda item: (item[1], item[0])):
            writer.writerow({field: results[key].get(field) for field in fields})


def series(
    results: dict[tuple[str, int], dict],
    config: str,
) -> tuple[list[float], list[float], list[float]]:
    contexts = sorted(context for (name, context) in results if name == config)
    return (
        [context / 1000 for context in contexts],
        [results[(config, context)]["decode_tps"] for context in contexts],
        [results[(config, context)]["prefill_tps"] for context in contexts],
    )


def plot_results(output_dir: Path, results: dict[tuple[str, int], dict], plt) -> None:
    if not results:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"upstream": "#1f77b4", "fork": "#d62728"}
    labels = {"upstream": "upstream (Raymond)", "fork": "fork + MTP (dynamic)"}
    markers = {"upstream": "o", "fork": "s"}

    for config in CONFIGS:
        if not any(name == config for (name, _) in results):
            continue
        x, decode, prefill = series(results, config)
        for ax, values, title in (
            (axes[0][0], decode, "Token generation (decode) throughput"),
            (axes[0][1], prefill, "Prompt processing (prefill) throughput"),
        ):
            ax.plot(
                x, values,
                color=colors[config],
                marker=markers[config],
                linewidth=2.2,
                label=labels[config],
            )
            ax.set_title(title)
            ax.set_xlabel("prompt size (thousands of tokens)")
            ax.set_ylabel("tokens/s")
            ax.grid(True, alpha=0.3)
            ax.legend()
        contexts = sorted(context for (name, context) in results if name == config)
        for context in contexts:
            row = results[(config, context)]
            if row.get("mtp_ejected"):
                axes[0][0].annotate(
                    "MTP ejected",
                    (context / 1000, row["decode_tps"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=colors[config],
                )

    contexts = sorted({context for (_, context) in results})
    ratio_contexts = []
    decode_ratio = []
    prefill_ratio = []
    for context in contexts:
        upstream = results.get(("upstream", context))
        fork = results.get(("fork", context))
        if upstream and fork:
            ratio_contexts.append(context / 1000)
            decode_ratio.append(fork["decode_tps"] / upstream["decode_tps"])
            prefill_ratio.append(fork["prefill_tps"] / upstream["prefill_tps"])
    ratios = (
        (axes[1][0], decode_ratio, "^", "#2ca02c", "Decode speedup (fork / upstream)"),
        (axes[1][1], prefill_ratio, "v", "#9467bd", "Prefill ratio (fork / upstream)"),
    )
    for ax, values, marker, color, title in ratios:
        if values:
            ax.plot(ratio_contexts, values, color=color, marker=marker, linewidth=2.2)
            ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("prompt size (thousands of tokens)")
        ax.set_ylabel("x")
        ax.grid(True, alpha=0.3)

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"upstream-vs-mtp.{suffix}", dpi=150)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("model", "mtp_model", "upstream_server", "fork_server"):
        if not getattr(args, name).is_file():
            raise SystemExit(f"--{name.replace('_', '-')} not found: {getattr(args, name)}")
    if args.batch_size <= 0 or args.ubatch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.ubatch_size > args.batch_size:
        raise SystemExit("--ubatch-size must not exceed --batch-size")
    if args.decode_tokens <= 0:
        raise SystemExit("--decode-tokens must be positive")
    if args.n_ctx <= 0:
        raise SystemExit("--n-ctx must be positive")
    for context in args.contexts:
        if context > args.n_ctx:
            raise SystemExit(f"context {context} exceeds --n-ctx {args.n_ctx}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=Path("/home/troed/llm-models/Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf"))
    parser.add_argument("--mtp-model", type=Path, default=Path("/home/troed/llm-models/Qwen3.8-27B-ASCII-Condensed-MTP.gguf"))
    parser.add_argument(
        "--upstream-server",
        type=Path,
        default=Path("/tmp/opencode/up-phase/build/bin/llama-server"),
    )
    parser.add_argument(
        "--fork-server",
        type=Path,
        default=ROOT / "build/bin" / "llama-server",
    )
    parser.add_argument("--contexts", type=parse_context_list, default=parse_context_list("20000,40000,80000,120000,160000"))
    parser.add_argument("--n-ctx", type=int, default=160000, help="configured context size; held fixed like the preset")
    parser.add_argument("--configs", default="upstream,fork")
    parser.add_argument("--upstream-arena-mib", type=int, default=3072)
    parser.add_argument("--fork-arena-mib", type=int, default=3264)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--spec-draft-n-max", type=int, default=3)
    parser.add_argument("--corpus-root", type=Path, default=ROOT)
    parser.add_argument("--corpus-max-chars", type=int, default=8_000_000)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--release-timeout", type=int, default=90)
    parser.add_argument("--release-slack-mib", type=int, default=64)
    args = parser.parse_args(argv)
    args.model = args.model.resolve()
    args.mtp_model = args.mtp_model.resolve()
    args.upstream_server = args.upstream_server.resolve()
    args.fork_server = args.fork_server.resolve()
    args.corpus_root = args.corpus_root.resolve()
    args.selected_configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    for name in args.selected_configs:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config: {name}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    plt = bks.require_matplotlib()
    plt.switch_backend("Agg")

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = ROOT / "benchmarks/results" / f"upstream-vs-mtp-{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    corpus, digest, file_count = build_corpus(args.corpus_root, args.corpus_max_chars)
    results = load_results(results_path)
    print(
        f"corpus: {len(corpus)} chars from {file_count} files, sha256 {digest[:12]}",
        flush=True,
    )

    suffix_tokens: list[int] | None = None
    baseline = bks.query_gpu_memory(args.nvidia_smi, args.gpu_index).used_mib
    try:
        for context in args.contexts:
            for config in args.selected_configs:
                if (config, context) in results:
                    print(f"[{config}] context {context}: cached", flush=True)
                    continue
                print(f"[{config}] context {context}: starting", flush=True)
                argv = server_argv(config, args, context)
                log_path = logs_dir / f"{config}-context-{context}.log"
                server = ServerHandle(argv, args, log_path)
                try:
                    if suffix_tokens is None:
                        suffix_tokens = tokenize(server, args.instruction, 60)
                    corpus_tokens = load_or_tokenize(
                        server,
                        corpus,
                        args.output_dir / "corpus-tokens.json",
                        args.request_timeout,
                    )
                    row = run_point(
                        server, config, args, context,
                        corpus_tokens, suffix_tokens, log_path,
                    )
                finally:
                    server.stop()
                append_jsonl(results_path, row)
                results[(config, context)] = row
                write_csv(args.output_dir / "results.csv", results)
                plot_results(args.output_dir, results, plt)
                bks.wait_for_release(args, baseline)
    finally:
        write_csv(args.output_dir / "results.csv", results)
        plot_results(args.output_dir, results, plt)

    print(f"results: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
