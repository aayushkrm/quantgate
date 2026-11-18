"""Variant registry and model fetching.

configs/variants.yaml is parsed into a Registry. ``fetch`` downloads each GGUF with
huggingface_hub into the models directory (default ~/models, never inside the repo), records
sha256 and byte size, and verifies them on later runs.
"""

from pathlib import Path

from pydantic import BaseModel

from quantgate.schema import Variant


class Registry(BaseModel):
    """Parsed configs/variants.yaml: a model family, the reference variant name and the ladder."""

    family: str
    reference: str
    variants: list[Variant]


def load_registry(path: Path) -> Registry:
    """Parse and validate the variant registry YAML.

    Raises ValueError if the reference name is not one of the variants.
    """
    raise NotImplementedError


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks so that multi-GB GGUFs do not fill memory."""
    raise NotImplementedError


def verify_variant(variant: Variant, path: Path) -> None:
    """Check size and sha256 against the registry. Raises ValueError on mismatch.

    Skips checks whose expected value is None, and reports that it did so.
    """
    raise NotImplementedError


def fetch_variant(variant: Variant, models_dir: Path) -> Path:
    """Download one GGUF via huggingface_hub into models_dir/<repo_id>/ and return its path."""
    raise NotImplementedError


def fetch_registry(registry_path: Path, models_dir: Path) -> list[tuple[Variant, Path]]:
    """Fetch and verify every variant in the registry. Returns (variant, local path) pairs."""
    raise NotImplementedError
