"""Where quantgate keeps local state: raw refusal completions, reference log-probs, audit samples.

The default root is ~/.cache/quantgate. The environment variable QUANTGATE_CACHE_DIR overrides it.
It is read at call time, not at import, so tests and scripts can point it elsewhere. The test suite
sets it to a temporary directory for every test, so no test writes to the real cache (see
tests/conftest.py and docs/deviations.md, entry 25).
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV: str = "QUANTGATE_CACHE_DIR"


def cache_root() -> Path:
    """The cache root: QUANTGATE_CACHE_DIR when set and non-empty, else ~/.cache/quantgate."""
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "quantgate"


def raw_root() -> Path:
    """Raw refusal completions: <root>/raw. Local only, never committed."""
    return cache_root() / "raw"


def ref_root() -> Path:
    """Reference log-probs for the KL cache: <root>/ref. Local only."""
    return cache_root() / "ref"


def audit_dir() -> Path:
    """Audit samples written by `quantgate audit export` by default: <root>/audit."""
    return cache_root() / "audit"
