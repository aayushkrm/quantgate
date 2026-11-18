"""Locate the default configs and committed subsets.

In a wheel built with `uv build`, the defaults are bundled under ``quantgate/_data/`` (see the
force-include table in pyproject.toml). In a repository checkout they are read from
``configs/`` and ``data/subsets/``. The bundled copy is used when present, so an installed
``quantgate demo`` or ``quantgate eval -c <registry>`` works from any directory.
"""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
BUNDLED_DIR: Path = PACKAGE_DIR / "_data"
REPO_ROOT: Path = PACKAGE_DIR.parents[1]


def is_bundled(path: Path) -> bool:
    """True if path lies inside the bundled data directory of an installed package."""
    try:
        Path(path).resolve().relative_to(BUNDLED_DIR.resolve())
    except ValueError:
        return False
    return True


def config_path(name: str) -> Path:
    """configs/<name>: the bundled copy if the package carries one, else the repo file."""
    bundled = BUNDLED_DIR / name
    if bundled.is_file():
        return bundled
    return REPO_ROOT / "configs" / name


def subsets_dir() -> Path:
    """data/subsets: the bundled directory if present, else the repo directory."""
    bundled = BUNDLED_DIR / "subsets"
    if bundled.is_dir():
        return bundled
    return REPO_ROOT / "data" / "subsets"
