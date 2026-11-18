"""Variant registry and model fetching.

configs/variants.yaml is parsed into a Registry. ``fetch_registry`` downloads each GGUF with
huggingface_hub into ``<models_dir>/<repo_id>/`` (default ~/models, never inside the repo),
records sha256 and byte size, and verifies them on later runs. ``ensure_ladder`` does the
same and, for a rung the Hub does not have, quantizes it from the fp16 reference with
llama.cpp and labels it "self-quantized, no imatrix".
"""

import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from pydantic import BaseModel, model_validator

from quantgate.schema import Variant

logger = logging.getLogger(__name__)

SELF_QUANT_LABEL = "self-quantized, no imatrix"
HUB_LABEL = "hub"

# Variant name -> attribute of the llama_cpp module holding the llama_ftype constant.
_FTYPE_ATTR: dict[str, str] = {
    "q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0",
    "q6_k": "LLAMA_FTYPE_MOSTLY_Q6_K",
    "q5_k_m": "LLAMA_FTYPE_MOSTLY_Q5_K_M",
    "q5_0": "LLAMA_FTYPE_MOSTLY_Q5_0",
    "q4_k_m": "LLAMA_FTYPE_MOSTLY_Q4_K_M",
    "q4_0": "LLAMA_FTYPE_MOSTLY_Q4_0",
    "q3_k_m": "LLAMA_FTYPE_MOSTLY_Q3_K_M",
    "q2_k": "LLAMA_FTYPE_MOSTLY_Q2_K",
}


class Registry(BaseModel):
    """Parsed configs/variants.yaml: a model family, the reference variant name and the ladder."""

    family: str
    reference: str
    variants: list[Variant]

    @model_validator(mode="after")
    def _check_reference(self) -> "Registry":
        """Reject a reference that is not in the ladder, duplicate names, or inconsistent flags."""
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variant names in registry")
        if self.reference not in names:
            raise ValueError(f"reference {self.reference!r} is not one of the variants")
        for v in self.variants:
            if v.is_reference != (v.name == self.reference):
                raise ValueError(f"is_reference flag on {v.name!r} disagrees with reference")
        return self

    def variant(self, name: str) -> Variant:
        """Return the variant with this name. Raises KeyError if absent."""
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(name)


class LadderEntry(BaseModel):
    """One rung of the ladder after ensure_ladder: the filled variant, its file and provenance.

    provenance is HUB_LABEL for files downloaded from the Hub, SELF_QUANT_LABEL for files
    quantized locally from the fp16 reference, and the same label for the reference.
    """

    variant: Variant
    path: Path
    provenance: str


def load_registry(path: Path) -> Registry:
    """Parse and validate the variant registry YAML.

    Raises ValueError if the reference name is not one of the variants, names repeat, or
    is_reference flags disagree with the reference. pydantic's ValidationError is a ValueError.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"registry {path} must be a YAML mapping")
    return Registry.model_validate(raw)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks so that multi-GB GGUFs do not fill memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(variant: Variant, models_dir: Path) -> Path:
    """Where a variant's GGUF lives: ``<models_dir>/<repo_id>/<filename>`` with ~ expanded.

    Raises ValueError if the filename would escape its repo directory.
    """
    if Path(variant.filename).name != variant.filename or variant.filename in {".", ".."}:
        raise ValueError(f"filename must be a plain file name: {variant.filename!r}")
    return Path(models_dir).expanduser() / variant.repo_id / variant.filename


def verify_variant(variant: Variant, path: Path) -> list[str]:
    """Check size and sha256 against the registry. Raises ValueError on mismatch.

    Checks whose expected value is None are skipped. The skipped check names are returned
    and logged at INFO, so the caller can record that nothing was verified.
    FileNotFoundError is raised if the file is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    skipped: list[str] = []
    if variant.file_bytes is None:
        skipped.append("file_bytes")
    else:
        size = path.stat().st_size
        if size != variant.file_bytes:
            raise ValueError(
                f"{variant.name}: size {size} != expected {variant.file_bytes} bytes ({path})"
            )
    if variant.sha256 is None:
        skipped.append("sha256")
    else:
        digest = sha256_file(path)
        if digest != variant.sha256.lower():
            raise ValueError(
                f"{variant.name}: sha256 {digest} != expected {variant.sha256} ({path})"
            )
    for check in skipped:
        logger.info("verify %s: no expected %s in registry, check skipped", variant.name, check)
    return skipped


def _hub_file_info(repo_id: str, filename: str) -> tuple[str | None, int | None]:
    """Return (sha256, size) of one file from the Hub tree API, or (None, None) if absent.

    Uses ``lfs.sha256`` and ``lfs.size`` when the file is LFS-tracked (GGUFs are).
    Network errors are not caught: the caller decides whether to continue.
    """
    infos = HfApi().get_paths_info(repo_id, [filename], expand=True)
    for info in infos:
        if getattr(info, "path", None) != filename:
            continue
        lfs = getattr(info, "lfs", None)
        if lfs is None:
            return None, getattr(info, "size", None)
        return getattr(lfs, "sha256", None), getattr(lfs, "size", None)
    return None, None


def fetch_variant(variant: Variant, models_dir: Path) -> tuple[Variant, Path]:
    """Download one GGUF into ``models_dir/<repo_id>/`` if needed and verify it.

    Steps:
    1. If the file exists and the registry pins sha256 or size, verify against the pins.
    2. Otherwise, if the file is missing, download it with hf_hub_download.
    3. Compare the local file with the Hub's lfs sha256 and size, when the Hub reports them.
    4. Return a copy of the variant with sha256 and file_bytes filled from the local file.

    Raises ValueError on any mismatch. The file is not deleted, so the caller can inspect it.
    Raises huggingface_hub EntryNotFoundError if the Hub lacks the file.
    """
    path = local_path(variant, models_dir)
    if path.is_file() and (variant.sha256 is not None or variant.file_bytes is not None):
        verify_variant(variant, path)
    elif not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=variant.repo_id,
                filename=variant.filename,
                local_dir=path.parent,
            )
        )
        if downloaded.resolve() != path.resolve():
            raise ValueError(f"download landed at {downloaded}, expected {path}")

    hub_sha, hub_size = _hub_file_info(variant.repo_id, variant.filename)
    size = path.stat().st_size
    digest = sha256_file(path)
    if hub_size is not None and hub_size != size:
        raise ValueError(f"{variant.name}: local size {size} != Hub size {hub_size} ({path})")
    if hub_sha is not None and hub_sha.lower() != digest:
        raise ValueError(f"{variant.name}: local sha256 {digest} != Hub {hub_sha} ({path})")
    if variant.sha256 is not None and variant.sha256.lower() != digest:
        raise ValueError(f"{variant.name}: sha256 {digest} != pinned {variant.sha256} ({path})")
    filled = variant.model_copy(update={"sha256": digest, "file_bytes": size})
    return filled, path


def fetch_registry(registry_path: Path, models_dir: Path) -> list[tuple[Variant, Path]]:
    """Fetch and verify every variant in the registry. Returns (filled variant, path) pairs.

    The registry YAML is not rewritten. Callers that want the recorded sha256 and bytes
    saved into configs/variants.yaml must do that themselves.
    """
    registry = load_registry(registry_path)
    return [fetch_variant(v, models_dir) for v in registry.variants]


def self_quantize(source: Path, variant: Variant, out_path: Path, nthread: int = 4) -> None:
    """Quantize ``source`` (fp16 GGUF) to the variant's ftype with llama.cpp, no imatrix.

    Writes to a ``.partial`` file and renames it on success, so an interrupted run never
    leaves a file that looks complete. Requires the ``llama`` extra.

    Raises RuntimeError if llama-cpp-python is missing or the quantizer returns non-zero, and
    ValueError if the variant name has no known ftype.
    """
    attr = _FTYPE_ATTR.get(variant.name)
    if attr is None:
        raise ValueError(f"no llama_ftype known for variant {variant.name!r}")
    try:
        import llama_cpp
    except ImportError as exc:
        raise RuntimeError("self-quantize needs llama-cpp-python (the 'llama' extra)") from exc

    params = llama_cpp.llama_model_quantize_default_params()
    params.nthread = nthread
    params.ftype = getattr(llama_cpp, attr)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    rc = llama_cpp.llama_model_quantize(
        str(Path(source)).encode("utf-8"),
        str(partial).encode("utf-8"),
        ctypes.byref(params),
    )
    if rc != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"llama_model_quantize failed for {variant.name} (rc={rc})")
    os.replace(partial, out_path)


def _provenance_path(path: Path) -> Path:
    """Sidecar JSON that records where a local GGUF came from."""
    return path.with_name(path.name + ".provenance.json")


def _write_provenance(path: Path, provenance: str, variant: Variant) -> None:
    """Record provenance, sha256 and size next to the file."""
    record = {
        "provenance": provenance,
        "sha256": variant.sha256,
        "file_bytes": variant.file_bytes,
    }
    _provenance_path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read_provenance(path: Path) -> dict[str, Any] | None:
    """Return the sidecar record, or None if it is absent."""
    sidecar = _provenance_path(path)
    if not sidecar.is_file():
        return None
    data: Any = json.loads(sidecar.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def ensure_ladder(
    registry_path: Path,
    models_dir: Path,
    *,
    self_quantize_missing: bool = True,
    nthread: int = 4,
) -> list[LadderEntry]:
    """Make every rung of the ladder available locally and verified.

    The reference (fp16) must come from the Hub. For each other rung, the Hub file is fetched
    and verified. If the Hub has no such file and ``self_quantize_missing`` is True, the rung
    is quantized from the local fp16 file with llama.cpp and labelled SELF_QUANT_LABEL. A
    sidecar ``.provenance.json`` makes reruns reuse a self-quantized file instead of redoing
    it. Raises EntryNotFoundError if a rung is missing and self-quantizing is disabled.
    """
    registry = load_registry(registry_path)
    ref_variant, ref_path = fetch_variant(registry.variant(registry.reference), models_dir)
    entries: list[LadderEntry] = [
        LadderEntry(variant=ref_variant, path=ref_path, provenance=HUB_LABEL)
    ]
    for variant in registry.variants:
        if variant.name == registry.reference:
            continue
        path = local_path(variant, models_dir)
        record = _read_provenance(path) if path.is_file() else None
        if record is not None and record.get("provenance") == SELF_QUANT_LABEL:
            filled = variant.model_copy(
                update={"sha256": record.get("sha256"), "file_bytes": record.get("file_bytes")}
            )
            verify_variant(filled, path)
            entries.append(LadderEntry(variant=filled, path=path, provenance=SELF_QUANT_LABEL))
            continue
        try:
            filled, path = fetch_variant(variant, models_dir)
        except EntryNotFoundError:
            if not self_quantize_missing:
                raise
            logger.warning("%s not on the Hub; quantizing from %s", variant.name, ref_path)
            self_quantize(ref_path, variant, path, nthread=nthread)
            size = path.stat().st_size
            filled = variant.model_copy(update={"sha256": sha256_file(path), "file_bytes": size})
            _write_provenance(path, SELF_QUANT_LABEL, filled)
            entries.append(LadderEntry(variant=filled, path=path, provenance=SELF_QUANT_LABEL))
            continue
        entries.append(LadderEntry(variant=filled, path=path, provenance=HUB_LABEL))
    return entries
