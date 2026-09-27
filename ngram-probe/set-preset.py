#!/usr/bin/env python3
"""Rewrite the [Qwen3.8-27B] block of the models preset for one probe leg.

Always rewrites from ngram-probe/models-preset.ini.orig, so earlier legs and
any appended scratch blocks are dropped each time. The other blocks are left
untouched. A log-file line is injected after spec-type so the child server
writes its full log to a path the sandbox can read; the file is truncated by
the child on every restart.

usage:
    set-preset.py ngram-simple --n-max 48 --draft off
    set-preset.py draft-dflash,ngram-simple --n-max 5 --draft on
    set-preset.py draft-dflash,ngram-simple --n-max 5 --draft on --keep-pages 1
    set-preset.py --restore
"""

import argparse
import os
import pathlib
import shutil
import sys

# the llama-server model directory: PROBE_MODELS_DIR, else ~/llm-models
MODELS_DIR = pathlib.Path(os.environ.get("PROBE_MODELS_DIR", "~/llm-models")).expanduser()
ORIG = pathlib.Path(__file__).with_name("models-preset.ini.orig")
TARGET = MODELS_DIR / "models-preset.ini"
SECTION = "[Qwen3.8-27B]"
DRAFT_KEYS = ("md", "device-draft", "n-gpu-layers-draft")
LOG_FILE = str(MODELS_DIR / "probe-server.log")


def key_of(line):
    bare = line.strip().lstrip("#").strip()
    if "=" not in bare:
        return None, False
    return bare.split("=", 1)[0].strip(), line.strip().startswith("#")


def transform(text, spec_type, n_max, draft, keep_pages, arena_mib, log_file, md_path):
    out = []
    inside = False
    for line in text.splitlines():
        if line.strip().startswith("["):
            inside = line.strip() == SECTION
            out.append(line)
            continue
        if not inside:
            out.append(line)
            continue

        key, commented = key_of(line)
        if key == "spec-type":
            if not commented:
                out.append(f"spec-type = {spec_type}")
                if log_file is not None:
                    out.append(f"log-file = {log_file}")
            continue
        if key == "spec-draft-n-max":
            if not commented:
                out.append(f"spec-draft-n-max = {n_max}")
            continue
        if key == "kv-stream-spec-keep-pages" and keep_pages is not None:
            if not commented:
                out.append(f"kv-stream-spec-keep-pages = {keep_pages}")
            continue
        if key == "kv-stream-arena-mib" and arena_mib is not None:
            if not commented:
                out.append(f"kv-stream-arena-mib = {arena_mib}")
            continue
        if key == "md" and md_path is not None and not commented:
            out.append(f"md = {md_path}")
            continue
        if key in DRAFT_KEYS:
            if draft or commented:
                out.append(line)
            else:
                out.append("#" + line.strip())
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_type", nargs="?")
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--draft", choices=["on", "off"], default="off")
    ap.add_argument("--keep-pages", type=int, default=None)
    ap.add_argument("--arena-mib", type=int, default=None)
    ap.add_argument("--log-file", default=LOG_FILE,
                    help="child server log path, or 'none' to omit")
    ap.add_argument("--md", default=None,
                    help="draft model path; overrides the preset's active md line")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_file = None if args.log_file == "none" else args.log_file

    orig = ORIG.read_text(encoding="utf-8")
    if args.restore:
        text = orig
        action = "restore"
    else:
        if not args.spec_type:
            raise SystemExit("spec_type required (or --restore)")
        text = transform(orig, args.spec_type, args.n_max, args.draft == "on",
                         args.keep_pages, args.arena_mib, log_file, args.md)
        action = (f"spec-type={args.spec_type} n-max={args.n_max} draft={args.draft} "
                  f"keep-pages={args.keep_pages} arena-mib={args.arena_mib} "
                  f"log-file={log_file} md={args.md}")

    if args.dry_run:
        block, inside = [], False
        for line in text.splitlines():
            if line.strip().startswith("["):
                inside = line.strip() == SECTION
            if inside:
                block.append(line)
        print("\n".join(block))
        return

    shutil.copyfile(TARGET, str(TARGET) + ".prev")
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}: {action}")


if __name__ == "__main__":
    sys.exit(main())
