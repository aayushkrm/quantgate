#!/usr/bin/env python3
"""Print environment information as JSON, for provenance of every result (DECISION.md §6.2).

Collected: git SHA and dirty flag, UTC time, CPU model and AVX2/AVX-512 flags, logical and
physical core counts, RAM, OS, Python and package versions (including llama-cpp-python when
installed). The output never includes a hostname or a username. This file is copied into each
repo, with no cross-repo dependency.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

# Distribution names to report. Missing packages are reported as null.
PACKAGES = [
    "llama-cpp-python",
    "pydantic",
    "typer",
    "PyYAML",
    "numpy",
    "matplotlib",
    "jinja2",
    "huggingface_hub",
    "datasets",
]


def _run(cmd: list[str]) -> str | None:
    """Stdout of a command, stripped, or None if the command fails or is missing."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def _read(path: str) -> str | None:
    """File contents, or None if the file cannot be read."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def cpu_flags(tokens: Iterable[str]) -> dict[str, bool]:
    """AVX2 and AVX-512 presence from a list of CPU feature tokens (any case)."""
    upper = {t.upper() for t in tokens}
    return {
        "avx2": "AVX2" in upper,
        "avx512": any(t.startswith("AVX512") for t in upper),
    }


def _cpu_tokens() -> tuple[str | None, list[str]]:
    """CPU brand string and feature tokens, from sysctl on macOS and /proc/cpuinfo on Linux."""
    if sys.platform == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        tokens: list[str] = []
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            feats = _run(["sysctl", "-n", key])
            if feats:
                tokens.extend(feats.split())
        return brand, tokens
    text = _read("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    flags = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
    brand = model.group(1).strip() if model else None
    tokens = flags.group(1).split() if flags else []
    return brand, tokens


def _ram_bytes() -> int | None:
    """Total physical memory in bytes."""
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        return int(raw) if raw and raw.isdigit() else None
    text = _read("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def _physical_cores() -> int | None:
    """Physical core count on macOS. None on Linux, where it is not read here."""
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(raw) if raw and raw.isdigit() else None
    return None


def _package_versions() -> dict[str, str | None]:
    """Installed version for each name in PACKAGES, or None if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_info(repo: Path) -> dict[str, Any]:
    """Commit SHA and whether the working tree has uncommitted changes."""
    sha = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo), "status", "--porcelain"])
    return {
        "git_sha": sha or None,
        "git_dirty": None if status is None else status != "",
    }


def collect(repo: Path) -> dict[str, Any]:
    """Build the environment record for the repo at ``repo``."""
    brand, tokens = _cpu_tokens()
    mac_version = platform.mac_ver()[0] if sys.platform == "darwin" else ""
    info: dict[str, Any] = {
        **_git_info(repo),
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": brand,
        "cpu_flags": cpu_flags(tokens),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": _physical_cores(),
        "ram_bytes": _ram_bytes(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "macos_version": mac_version or None,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": _package_versions(),
    }
    return info


if __name__ == "__main__":
    print(json.dumps(collect(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
