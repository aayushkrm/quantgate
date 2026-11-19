"""Test isolation. Every test runs with a temporary HOME and a temporary quantgate cache.

Before this fixture existed, the end-to-end tests wrote FakeBackend output into the real
~/.cache/quantgate/raw (incident in docs/deviations.md, entry 25). The real cache root is recorded
here, before any test can change the environment, so that tests/test_cache_isolation.py can check
against it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quantgate.cache import CACHE_ENV


def pytest_configure(config: pytest.Config) -> None:
    """Record the real cache root, as the environment has it before any test runs."""
    override = os.environ.get(CACHE_ENV, "").strip()
    real = Path(override).expanduser() if override else Path.home() / ".cache" / "quantgate"
    config.real_cache_root = real  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME and QUANTGATE_CACHE_DIR at temporary directories for this test."""
    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(CACHE_ENV, str(cache))
    return cache
