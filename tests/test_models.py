"""Registry, verification, fetch and ladder tests. No network: the Hub and llama.cpp are mocked."""

import ctypes
import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import EntryNotFoundError

from quantgate import models
from quantgate.models import (
    HUB_LABEL,
    SELF_QUANT_LABEL,
    ensure_ladder,
    fetch_registry,
    fetch_variant,
    load_registry,
    local_path,
    self_quantize,
    sha256_file,
    verify_variant,
)
from quantgate.schema import Variant

REPO = "Fake/Tiny-GGUF"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _variant(name: str, filename: str, *, ref: bool = False, **kw: Any) -> Variant:
    return Variant(name=name, repo_id=REPO, filename=filename, is_reference=ref, **kw)


REGISTRY_YAML = """\
family: tiny
reference: fp16
variants:
  - name: fp16
    repo_id: Fake/Tiny-GGUF
    filename: tiny-fp16.gguf
    is_reference: true
  - name: q8_0
    repo_id: Fake/Tiny-GGUF
    filename: tiny-q8_0.gguf
  - name: q4_0
    repo_id: Fake/Tiny-GGUF
    filename: tiny-q4_0.gguf
"""


def _write_registry(tmp_path: Path, text: str = REGISTRY_YAML) -> Path:
    path = tmp_path / "variants.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- fake Hub and fake llama_cpp --------------------------------------------------------------


class FakeHub:
    """Stands in for huggingface_hub: serves files from a dict and counts downloads."""

    def __init__(self, files: dict[str, bytes], hub_sha: bool = True) -> None:
        self.files = files
        self.hub_sha = hub_sha
        self.downloads: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = self

        class _Info:
            def __init__(self, path: str, data: bytes) -> None:
                self.path = path
                self.size = len(data)
                self.lfs = types.SimpleNamespace(sha256=_sha(data), size=len(data))

        class _Api:
            def get_paths_info(self, repo_id: str, paths: list[str], *, expand: bool) -> list:
                assert repo_id == REPO and expand
                return [_Info(p, hub.files[p]) for p in paths if p in hub.files and hub.hub_sha]

        def _download(repo_id: str, filename: str, *, local_dir: Path) -> str:
            assert repo_id == REPO
            if filename not in hub.files:
                raise EntryNotFoundError(f"{filename} not in repo")
            hub.downloads.append(filename)
            target = Path(local_dir) / filename
            target.write_bytes(hub.files[filename])
            return str(target)

        monkeypatch.setattr(models, "HfApi", _Api)
        monkeypatch.setattr(models, "hf_hub_download", _download)


class FakeLlamaCpp:
    """Stands in for the llama_cpp module. Records quantize calls and writes a small file."""

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple[bytes, bytes, int]] = []

    def module(self) -> types.ModuleType:
        fake = self

        class _Params(ctypes.Structure):
            _fields_ = [("nthread", ctypes.c_int), ("ftype", ctypes.c_int)]

        mod = types.ModuleType("llama_cpp")
        mod.llama_model_quantize_default_params = _Params  # type: ignore[attr-defined]

        def _quantize(src: bytes, dst: bytes, params: Any) -> int:
            # ctypes.byref returns a CArgObject; the struct is on its _obj attribute.
            struct = getattr(params, "_obj", params)
            fake.calls.append((src, dst, struct.ftype))
            if fake.rc == 0:
                Path(dst.decode()).write_bytes(b"QUANTIZED:" + Path(src.decode()).read_bytes())
            return fake.rc

        mod.llama_model_quantize = _quantize  # type: ignore[attr-defined]
        for i, attr in enumerate(sorted(models._FTYPE_ATTR.values()), start=1):
            setattr(mod, attr, i)
        return mod


# --- registry ----------------------------------------------------------------------------------


def test_load_registry_parses_ladder(tmp_path: Path) -> None:
    reg = load_registry(_write_registry(tmp_path))
    assert reg.family == "tiny"
    assert reg.reference == "fp16"
    assert [v.name for v in reg.variants] == ["fp16", "q8_0", "q4_0"]
    assert reg.variant("fp16").is_reference
    assert reg.variant("q8_0").sha256 is None
    with pytest.raises(KeyError):
        reg.variant("nope")


def test_registry_rejects_unknown_reference(tmp_path: Path) -> None:
    bad = REGISTRY_YAML.replace("reference: fp16", "reference: q2_k")
    with pytest.raises(ValueError, match="reference"):
        load_registry(_write_registry(tmp_path, bad))


def test_registry_rejects_inconsistent_reference_flag(tmp_path: Path) -> None:
    bad = REGISTRY_YAML.replace("is_reference: true", "is_reference: false")
    with pytest.raises(ValueError, match="is_reference"):
        load_registry(_write_registry(tmp_path, bad))


def test_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    dup = REGISTRY_YAML + "  - name: q8_0\n    repo_id: Fake/Tiny-GGUF\n    filename: x.gguf\n"
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(_write_registry(tmp_path, dup))


def test_registry_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_registry(_write_registry(tmp_path, "- just\n- a list\n"))


def test_committed_registry_loads() -> None:
    reg = load_registry(Path(__file__).resolve().parents[1] / "configs" / "variants.yaml")
    assert reg.reference == "fp16"
    assert len(reg.variants) == 9


# --- paths and hashing -------------------------------------------------------------------------


def test_sha256_file_matches_hashlib_on_multi_chunk_file(tmp_path: Path) -> None:
    data = b"x" * (3 * (1 << 20) + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert sha256_file(path) == _sha(data)


def test_local_path_expands_home_and_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    v = _variant("q8_0", "tiny-q8_0.gguf")
    assert local_path(v, Path("~/models")) == tmp_path / "models" / REPO / "tiny-q8_0.gguf"
    with pytest.raises(ValueError):
        local_path(_variant("evil", "../escape.gguf"), tmp_path)


# --- verification ------------------------------------------------------------------------------


def test_verify_passes_with_matching_pins_and_reports_no_skips(tmp_path: Path) -> None:
    data = b"gguf-bytes"
    path = tmp_path / "f.gguf"
    path.write_bytes(data)
    v = _variant("q8_0", "f.gguf", sha256=_sha(data), file_bytes=len(data))
    assert verify_variant(v, path) == []


def test_verify_detects_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "f.gguf"
    path.write_bytes(b"abc")
    v = _variant("q8_0", "f.gguf", sha256="0" * 64, file_bytes=3)
    with pytest.raises(ValueError, match="sha256"):
        verify_variant(v, path)


def test_verify_detects_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "f.gguf"
    path.write_bytes(b"abcd")
    v = _variant("q8_0", "f.gguf", file_bytes=3)
    with pytest.raises(ValueError, match="size"):
        verify_variant(v, path)


def test_verify_skips_unpinned_checks_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "f.gguf"
    path.write_bytes(b"abc")
    skipped = verify_variant(_variant("q8_0", "f.gguf"), path)
    assert skipped == ["file_bytes", "sha256"]


def test_verify_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_variant(_variant("q8_0", "nope.gguf"), tmp_path / "nope.gguf")


# --- fetch -------------------------------------------------------------------------------------


def test_fetch_downloads_into_repo_subdir_and_records_sha_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"fake fp16 weights"
    hub = FakeHub({"tiny-fp16.gguf": data})
    hub.install(monkeypatch)
    models_dir = tmp_path / "models"
    filled, path = fetch_variant(_variant("fp16", "tiny-fp16.gguf", ref=True), models_dir)
    assert path == models_dir / REPO / "tiny-fp16.gguf"
    assert path.read_bytes() == data
    assert filled.sha256 == _sha(data)
    assert filled.file_bytes == len(data)
    assert hub.downloads == ["tiny-fp16.gguf"]


def test_fetch_skips_download_when_pinned_local_file_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"already here"
    hub = FakeHub({"tiny-q8_0.gguf": data})
    hub.install(monkeypatch)
    models_dir = tmp_path / "models"
    path = models_dir / REPO / "tiny-q8_0.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    v = _variant("q8_0", "tiny-q8_0.gguf", sha256=_sha(data), file_bytes=len(data))
    filled, _ = fetch_variant(v, models_dir)
    assert hub.downloads == []
    assert filled.sha256 == _sha(data)


def test_fetch_refuses_pinned_mismatch_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub({"tiny-q8_0.gguf": b"new"})
    hub.install(monkeypatch)
    models_dir = tmp_path / "models"
    path = models_dir / REPO / "tiny-q8_0.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted")
    v = _variant("q8_0", "tiny-q8_0.gguf", sha256=_sha(b"new"))
    with pytest.raises(ValueError, match="sha256"):
        fetch_variant(v, models_dir)
    assert hub.downloads == []


def test_fetch_detects_local_file_differing_from_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub({"tiny-q8_0.gguf": b"hub copy"})
    hub.install(monkeypatch)
    models_dir = tmp_path / "models"
    path = models_dir / REPO / "tiny-q8_0.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"local copy!")  # same idea, different bytes and size
    with pytest.raises(ValueError, match="Hub"):
        fetch_variant(_variant("q8_0", "tiny-q8_0.gguf"), models_dir)


def test_fetch_registry_returns_filled_pairs_in_registry_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"tiny-fp16.gguf": b"a", "tiny-q8_0.gguf": b"bb", "tiny-q4_0.gguf": b"ccc"}
    FakeHub(files).install(monkeypatch)
    pairs = fetch_registry(_write_registry(tmp_path), tmp_path / "models")
    assert [v.name for v, _ in pairs] == ["fp16", "q8_0", "q4_0"]
    assert [v.file_bytes for v, _ in pairs] == [1, 2, 3]
    assert all(p.is_file() for _, p in pairs)


# --- ladder and self-quantization --------------------------------------------------------------


def test_self_quantize_writes_output_and_uses_ftype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLlamaCpp()
    monkeypatch.setitem(sys.modules, "llama_cpp", fake.module())
    src = tmp_path / "fp16.gguf"
    src.write_bytes(b"F16DATA")
    out = tmp_path / "q4_0.gguf"
    self_quantize(src, _variant("q4_0", "q4_0.gguf"), out)
    assert out.read_bytes() == b"QUANTIZED:F16DATA"
    assert not out.with_name(out.name + ".partial").exists()
    assert fake.calls[0][2] == sys.modules["llama_cpp"].LLAMA_FTYPE_MOSTLY_Q4_0  # type: ignore[attr-defined]


def test_self_quantize_failure_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", FakeLlamaCpp(rc=1).module())
    src = tmp_path / "fp16.gguf"
    src.write_bytes(b"F16DATA")
    out = tmp_path / "q4_0.gguf"
    with pytest.raises(RuntimeError, match="rc=1"):
        self_quantize(src, _variant("q4_0", "q4_0.gguf"), out)
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_self_quantize_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ftype"):
        self_quantize(tmp_path / "x", _variant("weird", "w.gguf"), tmp_path / "y")


def test_ensure_ladder_hub_rungs_and_self_quantized_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # q4_0 is absent from the Hub, so it must be quantized locally from fp16.
    files = {"tiny-fp16.gguf": b"FP16", "tiny-q8_0.gguf": b"Q8"}
    hub = FakeHub(files)
    hub.install(monkeypatch)
    fake = FakeLlamaCpp()
    monkeypatch.setitem(sys.modules, "llama_cpp", fake.module())
    models_dir = tmp_path / "models"

    entries = ensure_ladder(_write_registry(tmp_path), models_dir)

    by_name = {e.variant.name: e for e in entries}
    assert by_name["fp16"].provenance == HUB_LABEL
    assert by_name["q8_0"].provenance == HUB_LABEL
    assert by_name["q4_0"].provenance == SELF_QUANT_LABEL
    assert by_name["q4_0"].path.read_bytes() == b"QUANTIZED:FP16"
    assert by_name["q4_0"].variant.sha256 == _sha(b"QUANTIZED:FP16")
    assert len(fake.calls) == 1

    # Rerun: the self-quantized rung is reused from its sidecar, not quantized again.
    again = ensure_ladder(_write_registry(tmp_path), models_dir)
    assert len(fake.calls) == 1
    assert {e.variant.name: e.provenance for e in again}["q4_0"] == SELF_QUANT_LABEL


def test_ensure_ladder_raises_when_self_quantizing_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeHub({"tiny-fp16.gguf": b"FP16"}).install(monkeypatch)
    monkeypatch.setitem(sys.modules, "llama_cpp", FakeLlamaCpp().module())
    with pytest.raises(EntryNotFoundError):
        ensure_ladder(_write_registry(tmp_path), tmp_path / "models", self_quantize_missing=False)


def test_ensure_ladder_needs_the_reference_from_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeHub({}).install(monkeypatch)
    with pytest.raises(EntryNotFoundError):
        ensure_ladder(_write_registry(tmp_path), tmp_path / "models")


# --- sidecar lock file (configs/variants.lock.json) ---------------------------------------------


def _pinned(name: str, filename: str, data: bytes, tmp_path: Path) -> tuple[Variant, Path]:
    path = tmp_path / "models" / "repo" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    variant = Variant(
        name=name,
        repo_id="org/repo",
        filename=filename,
        sha256=sha256_file(path),
        file_bytes=path.stat().st_size,
    )
    return variant, path


def test_lock_round_trip_and_apply_fills_null_pins(tmp_path: Path) -> None:
    from quantgate.models import apply_lock, read_lock, write_lock

    filled, path = _pinned("q4_k_m", "q4.gguf", b"abc", tmp_path)
    lock_file = write_lock(tmp_path / "variants.lock.json", [(filled, path)])
    text = lock_file.read_text(encoding="utf-8")
    assert str(tmp_path) not in text, "the lock file must not contain local paths"

    entries = read_lock(lock_file)
    assert entries["q4_k_m"]["sha256"] == filled.sha256
    blank = Variant(name="q4_k_m", repo_id="org/repo", filename="q4.gguf")
    applied = apply_lock(blank, entries)
    assert applied.sha256 == filled.sha256
    assert applied.file_bytes == filled.file_bytes


def test_apply_lock_never_overwrites_registry_pins(tmp_path: Path) -> None:
    from quantgate.models import apply_lock

    pinned = Variant(name="x", repo_id="org/repo", filename="x.gguf", sha256="a" * 64)
    entries = {
        "x": {"repo_id": "org/repo", "filename": "x.gguf", "sha256": "b" * 64, "file_bytes": 3}
    }
    applied = apply_lock(pinned, entries)
    assert applied.sha256 == "a" * 64
    assert applied.file_bytes == 3


def test_apply_lock_rejects_entry_for_another_file() -> None:
    from quantgate.models import apply_lock

    blank = Variant(name="x", repo_id="org/repo", filename="x.gguf")
    entries = {
        "x": {"repo_id": "org/other", "filename": "x.gguf", "sha256": "c" * 64, "file_bytes": 3}
    }
    with pytest.raises(ValueError, match="another file"):
        apply_lock(blank, entries)


def test_read_lock_missing_file_is_empty_and_bad_schema_is_refused(tmp_path: Path) -> None:
    from quantgate.models import read_lock

    assert read_lock(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": "9", "variants": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="lock file"):
        read_lock(bad)


def test_default_lock_path_sits_next_to_registry() -> None:
    from quantgate.models import default_lock_path

    assert default_lock_path(Path("configs/variants.yaml")) == Path("configs/variants.lock.json")
