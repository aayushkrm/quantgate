"""scripts/envinfo.py prints a JSON record with the provenance keys and no host or user identity."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "envinfo.py"
REQUIRED_KEYS = {
    "git_sha",
    "git_dirty",
    "utc_time",
    "cpu_model",
    "cpu_flags",
    "cpu_cores_logical",
    "cpu_cores_physical",
    "ram_bytes",
    "os",
    "python",
    "packages",
}


@pytest.fixture(scope="module")
def envinfo_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("envinfo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_flags_detects_avx2_and_avx512(envinfo_module: ModuleType) -> None:
    flags = envinfo_module.cpu_flags(["sse4_2", "avx2", "avx512f", "avx512vl"])
    assert flags == {"avx2": True, "avx512": True}


def test_cpu_flags_on_haswell_style_list(envinfo_module: ModuleType) -> None:
    flags = envinfo_module.cpu_flags(["FMA", "AVX", "AVX2", "BMI2"])
    assert flags == {"avx2": True, "avx512": False}


def test_cpu_flags_empty(envinfo_module: ModuleType) -> None:
    assert envinfo_module.cpu_flags([]) == {"avx2": False, "avx512": False}


def test_script_prints_required_keys_as_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    info = json.loads(proc.stdout)
    assert REQUIRED_KEYS <= set(info)
    assert info["utc_time"].endswith("+00:00")
    assert info["python"]["version"].startswith("3.")
    assert "llama-cpp-python" in info["packages"]


def test_script_output_has_no_hostname_or_username_keys() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    info = json.loads(proc.stdout)

    def keys(obj: object) -> set[str]:
        if isinstance(obj, dict):
            found = set(obj)
            for value in obj.values():
                found |= keys(value)
            return found
        return set()

    banned = {"hostname", "node", "user", "username", "login"}
    assert not (keys(info) & banned)
