#!/usr/bin/env python3
"""Verify the built wheel contains the package code and the bundled defaults, and no stray data.

Run after `uv build`. Prints the wheel's file list and exits 1 if a required file is missing or
if a forbidden path (results, raw completions, model weights, local lock files) is present.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"

REQUIRED_SUFFIXES = (
    "quantgate/__init__.py",
    "quantgate/cli.py",
    "quantgate/schema.py",
    "quantgate/backend.py",
    "quantgate/envinfo.py",
    "quantgate/resources.py",
    "quantgate/evals/refusal.py",
    "quantgate/_data/gate.yaml",
    "quantgate/_data/variants.yaml",
    "quantgate/_data/subsets/arc_easy_test.json",
    "quantgate/_data/subsets/gsm8k_test.json",
    "quantgate/_data/subsets/wikitext2_test_passages.json",
)
FORBIDDEN_FRAGMENTS = (
    "results/",
    "/raw/",
    ".gguf",
    ".npy",
    "variants.lock.json",
    "__pycache__",
)


def main() -> int:
    wheels = sorted(DIST.glob("*.whl"))
    if not wheels:
        print("check_wheel: no wheel in dist/; run `uv build` first", file=sys.stderr)
        return 2
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    print(f"{wheel.name}: {len(names)} entries")
    for name in names:
        print(f"  {name}")
    status = 0
    for suffix in REQUIRED_SUFFIXES:
        if not any(n.endswith(suffix) for n in names):
            print(f"check_wheel: missing {suffix}", file=sys.stderr)
            status = 1
    for name in names:
        for bad in FORBIDDEN_FRAGMENTS:
            if bad in name:
                print(f"check_wheel: forbidden entry {name}", file=sys.stderr)
                status = 1
    if status == 0:
        print("wheel contents OK")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
