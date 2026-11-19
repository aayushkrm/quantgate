"""Command-line interface: `quantgate fetch | eval | gate | report | demo`.

Every command accepts --json. Exit codes: 0 = pass, 1 = check failed (gate rule or hidden
danger), 2 = usage or runtime error. Errors print one clean line (or a JSON error object with
--json), never a traceback. Demo output is labelled SYNTHETIC.
"""

import hashlib
import json
import math
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
import yaml

from quantgate import __version__ as quantgate_version
from quantgate.backend import FakeBackend, LlamaCppBackend
from quantgate.cache import raw_root as raw_root_path
from quantgate.data import load_arc_easy, load_gsm8k, load_wikitext_passages, select_subset
from quantgate.envinfo import collect as collect_envinfo
from quantgate.evals.gsm8k import run_gsm8k, score_gsm8k
from quantgate.evals.kl import run_kl
from quantgate.evals.mcq import McqItem, run_mcq, score_mcq
from quantgate.evals.refusal import (
    RAW_FILENAME,
    PromptSet,
    default_raw_path,
    load_benign_prompts,
    load_harmful_prompts,
    run_refusal,
)
from quantgate.gate import evaluate, gate_label, gate_report, load_policy, recommend
from quantgate.models import (
    LOCK_FILENAME as LOCK_NAME,
)
from quantgate.models import (
    Registry,
    apply_lock,
    default_lock_path,
    fetch_registry,
    load_registry,
    read_lock,
    sha256_file,
    verify_variant,
    write_lock,
)
from quantgate.perf import (
    PERF_N_CTX,
    measure_in_process,
    power_state,
    run_perf_subprocess,
    summarize_power_state,
)
from quantgate.report import (
    build_report,
    load_report,
    render_markdown,
    render_readme_block,
    update_readme,
    write_report,
    write_site,
)
from quantgate.resources import config_path, is_bundled, subsets_dir
from quantgate.schema import (
    ArcConfig,
    Backend,
    GatePolicy,
    GsmConfig,
    KLConfig,
    MetricValue,
    PerfConfig,
    QuantReport,
    RefusalConfig,
    SuiteConfig,
    Variant,
    VariantResult,
)

app = typer.Typer(
    help="CI gate for quantized GGUF models: KL drift, accuracy, refusal and CPU speed.",
    no_args_is_help=True,
    add_completion=False,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]

SUITE_NAMES: tuple[str, ...] = ("kl", "arc", "gsm8k", "refusal", "perf")
# Defaults resolve to the bundled copy in an installed wheel, or the repo files in a checkout.
DEFAULT_POLICY_NAME = "gate.yaml"
DEFAULT_REGISTRY_NAME = "variants.yaml"
MIB = 2**20
_T = TypeVar("_T")
CTX_BACKEND = 512  # quality suites: KL passages are 256 tokens; logits buffer is n_ctx x vocab
KL_CTX_MARGIN = 128  # tokens of headroom required above kl.ctx in the quality backend
PERF_TIMEOUT_S = 600.0


class CliError(Exception):
    """A usage or runtime error with a message fit for the user. Exit code 2."""


def _portable(path: Path) -> str:
    """A path for output: relative to the working directory, or a placeholder when outside it.

    Keeps user home directories and other absolute prefixes out of committed JSON.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"<outside-repo>/{path.name}"


def _emit(json_out: bool, payload: Any, text: str) -> None:
    """Print the JSON payload or the human text, to stdout."""
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(text)


def _describe(exc: BaseException) -> str:
    """A one-line description of an exception for the user."""
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    return text.splitlines()[0] if isinstance(exc, CliError) else f"{type(exc).__name__}: {text}"


@contextmanager
def _guard(json_out: bool) -> Iterator[None]:
    """Turn any failure into a clean error and exit code 2.

    typer.Exit is re-raised first: it subclasses RuntimeError, so catching it as an error would
    turn every deliberate exit code into 2.
    """
    try:
        yield
    except typer.Exit:
        raise
    except Exception as exc:
        message = _describe(exc)
        if json_out:
            typer.echo(json.dumps({"ok": False, "error": message, "exit_code": 2}, indent=2))
        else:
            typer.echo(f"quantgate: error: {message}", err=True)
        raise typer.Exit(code=2) from None


def _collect_env() -> dict[str, Any]:
    """Environment record from quantgate.envinfo (git SHA of the working directory, CPU, RAM,
    package versions). No host names or user names."""
    try:
        data = collect_envinfo(Path.cwd())
    except Exception as exc:
        raise CliError(f"environment record failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError("environment record is not a JSON object")
    return data


RAW_META_FILENAME = "refusal.meta.json"


def _check_raw_overwrite(raw_dir: Path, fingerprint: str, force: bool) -> None:
    """Refuse to replace a non-empty raw completions file made under a different configuration.

    The raw file is local and cannot be regenerated without a model run, so a rerun must not
    silently replace it (incident: docs/deviations.md, entry 25). A file with no record of its
    fingerprint is treated as a different configuration. --force replaces it.
    """
    raw = raw_dir / RAW_FILENAME
    if not raw.is_file() or raw.stat().st_size == 0:
        return
    meta = raw_dir / RAW_META_FILENAME
    if meta.is_file():
        stored = json.loads(meta.read_text(encoding="utf-8")).get("fingerprint")
        if stored == fingerprint:
            return
        reason = "a different configuration"
    else:
        reason = "no record of the configuration that wrote it"
    if force:
        return
    raise CliError(
        f"{_portable(raw)} is a non-empty raw file from {reason}. Rerunning the refusal suite "
        "would replace it. Pass --force to replace it, or set QUANTGATE_CACHE_DIR to another "
        "directory."
    )


def _write_raw_meta(raw_dir: Path, fingerprint: str, variant: str) -> None:
    """Record which configuration is about to write the raw file. Written before the run starts,
    so a crash mid-run leaves a record that matches the file being written."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": variant,
        "fingerprint": fingerprint,
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (raw_dir / RAW_META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _raw_dir(variant: str) -> Path:
    """Local-only raw refusal completions for one variant, outside the repository (review Q8).

    The directory is ~/.cache/quantgate/raw/<variant>/, as set by refusal.default_raw_path.
    """
    return default_raw_path(variant).parent


def _registry_arg(path: Path | None) -> Path:
    """The variant registry: the path given, else the bundled or repo default."""
    return config_path(DEFAULT_REGISTRY_NAME) if path is None else path


def _policy_arg(path: Path | None) -> Path:
    """The gate policy: the path given, else the bundled or repo default."""
    return config_path(DEFAULT_POLICY_NAME) if path is None else path


def _lock_default(registry: Path) -> Path:
    """Where the lock file goes by default: next to the registry. For the bundled registry of an
    installed package the package directory is read-only in spirit, so the lock goes to the
    working directory's configs/ instead."""
    if is_bundled(registry):
        return Path.cwd() / "configs" / LOCK_NAME
    return default_lock_path(registry)


def load_suite_config(path: Path) -> SuiteConfig:
    """Read the `suite:` section of the variant registry YAML into a SuiteConfig.

    Raises CliError if the section is missing. Validation errors come from pydantic (ValueError).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "suite" not in data:
        raise CliError(f"{path} has no 'suite' section (see configs/variants.yaml)")
    return SuiteConfig.model_validate(data["suite"])


def _scale_count(n: int, factor: float) -> int:
    """Scaled sample count: at least 1, rounded up. Unchanged when factor is 1."""
    if factor >= 1.0:
        return n
    return max(1, math.ceil(n * factor))


def scale_suite(cfg: SuiteConfig, factor: float) -> SuiteConfig:
    """Copy of cfg with every sample count multiplied by factor. Perf settings are not scaled."""
    return SuiteConfig(
        kl=KLConfig(n_passages=_scale_count(cfg.kl.n_passages, factor), ctx=cfg.kl.ctx),
        arc=ArcConfig(n=_scale_count(cfg.arc.n, factor)),
        gsm8k=GsmConfig(n=_scale_count(cfg.gsm8k.n, factor), max_tokens=cfg.gsm8k.max_tokens),
        refusal=RefusalConfig(
            n_harmful=_scale_count(cfg.refusal.n_harmful, factor),
            n_benign=_scale_count(cfg.refusal.n_benign, factor),
            max_tokens=cfg.refusal.max_tokens,
        ),
        perf=cfg.perf,
        seed=cfg.seed,
    )


def _parse_suites(text: str) -> list[str]:
    """Validate --suite and return the suites in canonical order, without duplicates."""
    names = [s.strip() for s in text.split(",") if s.strip()]
    if not names:
        raise CliError("--suite is empty")
    unknown = sorted(set(names) - set(SUITE_NAMES))
    if unknown:
        raise CliError(
            f"unknown suite(s): {', '.join(unknown)}; choose from {', '.join(SUITE_NAMES)}"
        )
    return [s for s in SUITE_NAMES if s in names]


def _select_variants(registry: Registry, only: str | None) -> list[Variant]:
    """Reference first, then candidates in registry order. The reference is always included.

    With --only, candidates are restricted to the named variants.
    """
    by_name = {v.name: v for v in registry.variants}
    ref = by_name.get(registry.reference)
    if ref is None:
        raise CliError(f"reference {registry.reference!r} is not in the registry")
    if only is None:
        chosen = [v for v in registry.variants if v.name != ref.name]
    else:
        names = {s.strip() for s in only.split(",") if s.strip()}
        unknown = sorted(names - set(by_name))
        if unknown:
            raise CliError(f"unknown variant(s) in --only: {', '.join(unknown)}")
        chosen = [v for v in registry.variants if v.name in names and v.name != ref.name]
    return [ref, *chosen]


def _model_path(models_dir: Path, variant: Variant) -> Path:
    """Where fetch stores a variant: models_dir/<repo_id>/<filename>."""
    return models_dir / variant.repo_id / variant.filename


METHOD_VERSION = "3"  # bump whenever the way a suite is computed changes (docs/deviations.md)
GEN_N_CTX = 1024  # generation suites (gsm8k, refusal): no logits buffer, room for 384 + prompt
GEN_PROMPT_ROOM = 256  # tokens reserved for the chat prompt inside a generation context
LOGIT_SUITES: tuple[str, ...] = ("kl", "arc")  # need logits: KL and ARC scoring
GEN_SUITES: tuple[str, ...] = ("gsm8k", "refusal")  # generation only (deviation 15)


def _subset_digest() -> str:
    """SHA-256 over the committed subset files, so a changed subset changes every fingerprint."""
    digest = hashlib.sha256()
    base = subsets_dir()
    for path in sorted(base.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _suite_ctx(suite: str) -> int:
    """The context size a suite runs with, which is part of its fingerprint."""
    if suite in GEN_SUITES:
        return GEN_N_CTX
    if suite == "perf":
        return PERF_N_CTX
    return CTX_BACKEND


def _fingerprint(suite: str, cfg: SuiteConfig, variant: Variant, n_scale: float) -> str:
    """Identity of one suite result: method, code version, settings, model file and subsets.

    Two runs share a result file only when this matches (review Q3). The variant should carry
    its file size, from the registry, the lock or the file on disk.
    """
    payload: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "quantgate_version": quantgate_version,
        "suite": suite,
        "suite_cfg": cfg.model_dump(mode="json")[suite],
        "seed": cfg.seed,
        "n_scale": n_scale,
        "ctx": _suite_ctx(suite),
        "variant": {
            "name": variant.name,
            "repo_id": variant.repo_id,
            "filename": variant.filename,
            "sha256": variant.sha256,
            "file_bytes": variant.file_bytes,
        },
        "subsets_sha256": _subset_digest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _done(out: Path, variant: str, suite: str, fingerprint: str) -> bool:
    """True if a finished result file exists for this fingerprint (resumable runs).

    Raises CliError if the file exists but was made under a different configuration. Such a file
    is never reused silently, and it is never overwritten either: the user chooses a new --out
    directory or deletes the file.
    """
    path = out / variant / f"{suite}.json"
    if not path.is_file():
        return False
    stored = json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
    if stored != fingerprint:
        raise CliError(
            f"{_portable(path)} was produced under a different configuration (fingerprint "
            "differs). Use a new --out directory, or delete that file to recompute it."
        )
    return True


def _write_suite(
    out: Path,
    variant: str,
    suite: str,
    metrics: list[MetricValue],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one variant's metrics for one suite as JSON, plus any provenance in extra."""
    path = out / variant / f"{suite}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "variant": variant,
        "suite": suite,
        "metrics": [m.model_dump(mode="json") for m in metrics],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _backend_options(generation: bool, cfg: SuiteConfig) -> dict[str, Any]:
    """Context and logits settings for one backend kind (deviation 15).

    Logits backend (KL, ARC): logits_all=True, n_ctx=512, which is the passage length plus margin.
    The logits buffer is n_ctx x vocab, so it stays small. Generation backend (GSM8K, refusal):
    logits_all=False, n_ctx=1024, with no per-position logits. Raises CliError if the settings
    cannot hold their inputs.
    """
    if generation:
        if cfg.gsm8k.max_tokens + GEN_PROMPT_ROOM > GEN_N_CTX:
            raise CliError(
                f"gsm8k.max_tokens={cfg.gsm8k.max_tokens} does not fit n_ctx={GEN_N_CTX} "
                f"with {GEN_PROMPT_ROOM} tokens of prompt"
            )
        return {"n_ctx": GEN_N_CTX, "logits_all": False, "n_batch": GEN_N_CTX}
    if cfg.kl.ctx + KL_CTX_MARGIN > CTX_BACKEND:
        raise CliError(
            f"kl.ctx={cfg.kl.ctx} needs n_ctx >= {cfg.kl.ctx + KL_CTX_MARGIN}; "
            f"the logits backend has n_ctx={CTX_BACKEND}"
        )
    return {"n_ctx": CTX_BACKEND, "logits_all": True, "n_batch": CTX_BACKEND}


def _load_backend(path: Path, cfg: SuiteConfig, generation: bool = False) -> Backend:
    """Load one GGUF file for one kind of quality suite: CPU only, perf thread count."""
    return LlamaCppBackend(
        path,
        n_threads=cfg.perf.threads,
        seed=cfg.seed,
        **_backend_options(generation, cfg),
    )


def _split_quality(todo: list[str]) -> tuple[list[str], list[str]]:
    """Split pending quality suites into those that need the logits backend and the rest."""
    logit = [s for s in todo if s in LOGIT_SUITES]
    gen = [s for s in todo if s in GEN_SUITES]
    unknown = [s for s in todo if s not in LOGIT_SUITES and s not in GEN_SUITES]
    if unknown:
        raise CliError(f"no backend kind for suite(s): {', '.join(unknown)}")
    return logit, gen


def _take(items: list[_T], n: int, seed: int) -> list[_T]:
    """A seeded subset of n items, in their original order. All items when n covers them.

    Selection uses data.select_subset over item indices, so a scaled run is reproducible.
    """
    if n >= len(items):
        return list(items)
    chosen = select_subset([str(i) for i in range(len(items))], n, seed)
    return [items[i] for i in sorted(int(c) for c in chosen)]


def _provenance(prompts: PromptSet) -> dict[str, Any]:
    """Where a refusal prompt set came from, and whether a fallback source was used."""
    return {
        "name": prompts.name,
        "source": prompts.source,
        "fallback_reason": prompts.fallback_reason,
        "seed": prompts.seed,
        "total_available": prompts.total_available,
        "n_selected": len(prompts.prompts),
    }


def _run_suite(
    name: str,
    backend: Backend,
    ref_backend: Backend | None,
    cfg: SuiteConfig,
    cache: dict[str, Any],
    raw_dir: Path,
) -> tuple[list[MetricValue], dict[str, Any] | None]:
    """Run one quality suite for one backend.

    Returns the metrics and any provenance to record (refusal prompt sets). Inputs are loaded
    once per run into cache, then reduced to the scaled sample counts.
    """
    if name == "kl":
        if ref_backend is None:
            raise CliError("the kl suite needs the reference backend")
        if "kl" not in cache:
            cache["kl"] = _take(load_wikitext_passages(), cfg.kl.n_passages, cfg.seed)
        kl_metrics = run_kl(ref_backend, backend, cache["kl"], cfg.kl.ctx, seed=cfg.seed)
        return kl_metrics, None
    if name == "arc":
        if "arc" not in cache:
            cache["arc"] = _take(load_arc_easy(), cfg.arc.n, cfg.seed)
        arc_items: list[McqItem] = cache["arc"]
        arc_result = score_mcq(backend, arc_items)
        # Per-item correctness lets the gate run a paired test (review Q2). Items keep their ids.
        arc_extra = {
            "item_ids": {"arc_acc": [item.item_id for item in arc_items]},
            "per_item": {"arc_acc": [int(c) for c in arc_result.correct]},
        }
        return [arc_result.metric], arc_extra
    if name == "gsm8k":
        if "gsm8k" not in cache:
            cache["gsm8k"] = _take(load_gsm8k(), cfg.gsm8k.n, cfg.seed)
        pairs: list[tuple[str, str]] = cache["gsm8k"]
        gsm_result = score_gsm8k(backend, pairs, cfg.gsm8k.max_tokens, cfg.seed)
        # A truncated generation's last number is not an answer, so the count is recorded.
        diagnostics = {
            "gsm8k": {
                "n": len(pairs),
                "n_truncated": gsm_result.n_truncated,
                "max_tokens": cfg.gsm8k.max_tokens,
            }
        }
        # GSM8K items have no ids in the committed subset, so their position in the subset is
        # the id. The subset order is fixed by the seed, and the fingerprint covers it.
        gsm_extra = {
            "item_ids": {"gsm8k_acc": [str(i) for i in range(len(pairs))]},
            "per_item": {"gsm8k_acc": [int(c) for c in gsm_result.correct]},
            "diagnostics": diagnostics,
        }
        return [gsm_result.metric], gsm_extra
    if name == "refusal":
        if "harmful" not in cache:
            cache["harmful"] = load_harmful_prompts(n=cfg.refusal.n_harmful, seed=cfg.seed)
            cache["benign"] = load_benign_prompts(n=cfg.refusal.n_benign, seed=cfg.seed)
        harm_set: PromptSet = cache["harmful"]
        ben_set: PromptSet = cache["benign"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        refusal_metrics = run_refusal(
            backend,
            [p.text for p in harm_set.prompts],
            [p.text for p in ben_set.prompts],
            cfg.refusal.max_tokens,
            cfg.seed,
            raw_dir=raw_dir,
        )
        provenance = {
            "prompt_sets": {"harmful": _provenance(harm_set), "benign": _provenance(ben_set)}
        }
        return refusal_metrics, provenance
    raise CliError(f"unknown quality suite {name!r}")


def _perf_metrics(
    raw: dict[str, float],
    repeats: int,
    file_path: Path | None,
    need_rss: bool,
) -> list[MetricValue]:
    """Perf metrics from a measure_in_process / run_perf_subprocess result.

    decode_tok_s uses the min and max over repeats as its range. file_mb is the size on disk,
    included only when file_path is given. Raises CliError if a required key is missing.
    """

    def need(key: str) -> float:
        if key not in raw:
            raise CliError(f"perf result is missing {key!r}")
        return float(raw[key])

    decode_lo = raw.get("decode_tok_s_min")
    decode_hi = raw.get("decode_tok_s_max")
    metrics = [
        MetricValue(
            name="decode_tok_s",
            value=need("decode_tok_s_p50"),
            ci_low=None if decode_lo is None else float(decode_lo),
            ci_high=None if decode_hi is None else float(decode_hi),
            n=repeats,
        ),
        MetricValue(name="prefill_tok_s", value=need("prefill_tok_s_p50"), n=repeats),
    ]
    rss_mb: float | None = None
    if "peak_rss_bytes" in raw:
        rss_mb = float(raw["peak_rss_bytes"]) / MIB
    elif "peak_rss_mb" in raw:
        rss_mb = float(raw["peak_rss_mb"])
    elif need_rss:
        raise CliError("perf result is missing peak RSS (peak_rss_bytes)")
    if rss_mb is not None:
        metrics.append(MetricValue(name="peak_rss_mb", value=rss_mb, n=repeats))
    if file_path is not None:
        metrics.append(MetricValue(name="file_mb", value=file_path.stat().st_size / MIB, n=1))
    return metrics


def _with_size(variant: Variant, paths: dict[str, Path]) -> Variant:
    """The variant with file_bytes from the file on disk when the registry and lock lack it."""
    if variant.file_bytes is not None or variant.name not in paths:
        return variant
    return variant.model_copy(update={"file_bytes": paths[variant.name].stat().st_size})


def _quality_phase(
    variants: list[Variant],
    registry: Registry,
    paths: dict[str, Path],
    suites: list[str],
    cfg: SuiteConfig,
    out: Path,
    ran: list[str],
    skipped: list[str],
    n_scale: float = 1.0,
    force: bool = False,
) -> None:
    """Run the quality suites variant by variant, reference first, skipping finished files.

    Each variant gets up to two backends, one after the other (deviation 15). The logits backend
    serves KL and ARC. The generation backend serves GSM8K and refusal. The reference's logits
    backend stays loaded for the whole phase, because KL needs it. Candidates are loaded one at a
    time.
    """
    quality = [s for s in suites if s != "perf"]
    if not quality:
        return
    cache: dict[str, Any] = {}
    ref_logits: Backend | None = None

    def fp(v: Variant, suite: str) -> str:
        return _fingerprint(suite, cfg, _with_size(v, paths), n_scale)

    def ensure_ref_logits() -> Backend:
        nonlocal ref_logits
        if ref_logits is None:
            ref_logits = _load_backend(paths[registry.reference], cfg, generation=False)
        return ref_logits

    try:
        for v in variants:
            todo = [s for s in quality if not _done(out, v.name, s, fp(v, s))]
            skipped += [f"{v.name}/{s}" for s in quality if s not in todo]
            if not todo:
                continue
            logit_todo, gen_todo = _split_quality(todo)
            owned = v.name != registry.reference
            if logit_todo:
                backend = (
                    _load_backend(paths[v.name], cfg, generation=False)
                    if owned
                    else ensure_ref_logits()
                )
                try:
                    ref_for_kl = ensure_ref_logits() if "kl" in logit_todo else None
                    for s in logit_todo:
                        metrics, extra = _run_suite(
                            s, backend, ref_for_kl, cfg, cache, _raw_dir(v.name)
                        )
                        _write_suite(
                            out, v.name, s, metrics, {**(extra or {}), "fingerprint": fp(v, s)}
                        )
                        ran.append(f"{v.name}/{s}")
                finally:
                    if owned:
                        backend.close()
            if gen_todo:
                gen_backend = _load_backend(paths[v.name], cfg, generation=True)
                try:
                    for s in gen_todo:
                        if s == "refusal":
                            _check_raw_overwrite(_raw_dir(v.name), fp(v, s), force)
                            _write_raw_meta(_raw_dir(v.name), fp(v, s), v.name)
                        metrics, extra = _run_suite(
                            s, gen_backend, None, cfg, cache, _raw_dir(v.name)
                        )
                        _write_suite(
                            out, v.name, s, metrics, {**(extra or {}), "fingerprint": fp(v, s)}
                        )
                        ran.append(f"{v.name}/{s}")
                finally:
                    gen_backend.close()
    finally:
        if ref_logits is not None:
            ref_logits.close()


def _check_perf_conditions(
    variant: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    """Refuse a perf result that broke the pre-registered conditions (docs/prereg.md §3).

    On macOS the power source must be AC both before and after, and no thermal or performance
    warning may be recorded. Otherwise nothing is written for this variant, so a rerun measures
    it again. Off macOS pmset is unavailable, and the conditions cannot be checked. That is
    recorded in the result, not treated as a failure.
    """
    if not (before.get("pmset_available") or after.get("pmset_available")):
        return
    for label, state in (("before", before), ("after", after)):
        source = state.get("power_source")
        if source != "AC Power":
            raise CliError(
                f"perf for {variant} measured on '{source}' power ({label}); the protocol needs "
                "AC power. Plug in and rerun this variant (docs/prereg.md §3)."
            )
        if state.get("thermal_warning_recorded") or state.get("performance_warning_recorded"):
            raise CliError(
                f"perf for {variant} has a thermal or performance warning ({label}). Let the "
                "machine cool and rerun this variant (docs/prereg.md §3)."
            )


def _perf_phase(
    variants: list[Variant],
    paths: dict[str, Path],
    cfg: SuiteConfig,
    out: Path,
    ran: list[str],
    skipped: list[str],
    n_scale: float = 1.0,
) -> None:
    """Run perf for each variant in its own subprocess, one at a time, after the quality phase."""
    for v in variants:
        fp = _fingerprint("perf", cfg, _with_size(v, paths), n_scale)
        if _done(out, v.name, "perf", fp):
            skipped.append(f"{v.name}/perf")
            continue
        power_before = summarize_power_state(power_state())
        raw = run_perf_subprocess(paths[v.name], cfg.perf, timeout_s=PERF_TIMEOUT_S)
        power_after = summarize_power_state(power_state())
        _check_perf_conditions(v.name, power_before, power_after)
        metrics = _perf_metrics(raw, cfg.perf.repeats, paths[v.name], need_rss=True)
        # Demo perf is in-process, so it has no file; here the file size is always recorded.
        _write_suite(
            out,
            v.name,
            "perf",
            metrics,
            {
                "fingerprint": fp,
                "power_before": power_before,
                "power_after": power_after,
            },
        )
        ran.append(f"{v.name}/perf")


def _assemble(
    registry: Registry,
    paths: dict[str, Path],
    out: Path,
    pinned: dict[str, Variant] | None = None,
) -> list[VariantResult]:
    """Build one VariantResult per variant from every suite file on disk, and write variant.json.

    The merge reads all suites in SUITE_NAMES, whatever suites this invocation ran, so the report
    is the same however the runs were split (review bug: a perf-only run had dropped the quality
    metrics). pinned maps variant names to registry entries with lock values filled in.
    """
    results: list[VariantResult] = []
    for registry_variant in registry.variants:
        v = (pinned or {}).get(registry_variant.name, registry_variant)
        metrics: dict[str, MetricValue] = {}
        item_ids: dict[str, list[str]] = {}
        per_item: dict[str, list[int]] = {}
        found = False
        for s in SUITE_NAMES:
            path = out / v.name / f"{s}.json"
            if not path.is_file():
                continue
            found = True
            data = json.loads(path.read_text(encoding="utf-8"))
            for m in data["metrics"]:
                mv = MetricValue.model_validate(m)
                metrics[mv.name] = mv
            item_ids.update(data.get("item_ids", {}))
            per_item.update(data.get("per_item", {}))
        if not found:
            continue
        variant = v
        if v.name in paths and v.file_bytes is None and paths[v.name].is_file():
            variant = v.model_copy(update={"file_bytes": paths[v.name].stat().st_size})
        raw_dir = _raw_dir(v.name)
        result = VariantResult(
            variant=variant,
            metrics=metrics,
            raw_path=_portable(raw_dir) if raw_dir.is_dir() else None,
            item_ids=item_ids,
            per_item=per_item,
        )
        (out / v.name).mkdir(parents=True, exist_ok=True)
        (out / v.name / "variant.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
    return results


@app.command()
def fetch(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Variant registry YAML.")
    ] = None,
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Where GGUF files are stored (outside the repo).")
    ] = Path("~/models"),
    lock: Annotated[
        Path | None,
        typer.Option("--lock", help="Lock file to write (default: next to the registry)."),
    ] = None,
    json_out: JsonOption = False,
) -> None:
    """Download every variant in the registry and verify sha256 and size.

    The observed sha256 and size of every file are written to a sidecar lock file (by default
    configs/variants.lock.json). `eval` verifies against the lock when the registry pin is null.
    The registry YAML itself is not rewritten, so its comments stay as they are.
    """
    with _guard(json_out):
        config = _registry_arg(config)
        pairs = fetch_registry(config, models_dir.expanduser())
        lock_path = _lock_default(config) if lock is None else lock
        write_lock(lock_path, pairs)
        rows: list[dict[str, Any]] = []
        for variant, path in pairs:
            observed = sha256_file(path) if variant.sha256 is None else None
            rows.append(
                {
                    "variant": variant.name,
                    "path": _portable(path),
                    "file_bytes": path.stat().st_size,
                    "sha256_observed": observed,
                    "registry_sha256_checked": variant.sha256 is not None,
                }
            )
        text = "\n".join(
            f"{r['variant']:10} {r['file_bytes']:>12} bytes  "
            + (
                "verified against registry"
                if r["registry_sha256_checked"]
                else f"sha256 {r['sha256_observed']} (not in registry)"
            )
            for r in rows
        )
        text += f"\nWrote lock file {_portable(lock_path)}."
        _emit(json_out, {"ok": True, "variants": rows, "lock": _portable(lock_path)}, text)


@app.command("eval")
def eval_cmd(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Variant registry YAML.")
    ] = None,
    suite: Annotated[
        str, typer.Option("--suite", help="Comma-separated subset of: kl,arc,gsm8k,refusal,perf.")
    ] = "kl,arc,gsm8k,refusal,perf",
    out: Annotated[Path, typer.Option("--out", help="Results directory.")] = Path("results"),
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Where GGUF files are stored.")
    ] = Path("~/models"),
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated candidate variants. The reference is always run."
        ),
    ] = None,
    n_scale: Annotated[
        float,
        typer.Option("--n-scale", help="Multiply every sample count by this (0 < x <= 1)."),
    ] = 1.0,
    policy: Annotated[Path | None, typer.Option("--policy", "-p", help="Gate policy YAML.")] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace an existing raw completions file made under a different configuration.",
        ),
    ] = False,
    json_out: JsonOption = False,
) -> None:
    """Run the eval suite for every variant and write per-variant JSON under --out.

    Resumable: a suite whose result file exists is skipped. Writes <out>/<variant>/<suite>.json,
    <out>/<variant>/variant.json, <out>/report.json and <out>/env.json. Perf runs last, one
    subprocess per variant. A reduced --n-scale is recorded in env.json and must be logged as a
    deviation.
    """
    with _guard(json_out):
        config = _registry_arg(config)
        policy = _policy_arg(policy)
        suites = _parse_suites(suite)
        if not 0.0 < n_scale <= 1.0:
            raise CliError("--n-scale must be greater than 0 and at most 1")
        registry = load_registry(config)
        base_cfg = load_suite_config(config)
        cfg = scale_suite(base_cfg, n_scale)
        gate_policy: GatePolicy = load_policy(policy)
        variants = _select_variants(registry, only)

        lock_entries = read_lock(_lock_default(config))
        variants = [apply_lock(v, lock_entries) for v in variants]

        models = models_dir.expanduser()
        paths: dict[str, Path] = {}
        for v in variants:
            path = _model_path(models, v)
            if not path.is_file():
                raise CliError(f"missing model file for {v.name}; run `quantgate fetch` first")
            paths[v.name] = path
        unverified: dict[str, list[str]] = {}
        for v in variants:
            skipped_checks = verify_variant(v, paths[v.name])
            if skipped_checks:
                unverified[v.name] = skipped_checks

        out.mkdir(parents=True, exist_ok=True)
        env_info = _collect_env()
        run_now: dict[str, Any] = {
            "suite": cfg.model_dump(mode="json"),
            "n_scale": n_scale,
            "suites": suites,
            "variants": [v.name for v in variants],
            "policy": _portable(policy),
            "unverified": unverified,
        }
        env_info["run"] = _merge_env_run(out / "env.json", run_now)
        env_info["power_at_start"] = summarize_power_state(power_state())
        env_info["run_history"] = _run_history(out / "env.json", env_info, run_now)
        (out / "env.json").write_text(json.dumps(env_info, indent=2, sort_keys=True) + "\n")

        ran: list[str] = []
        skipped: list[str] = []
        _quality_phase(variants, registry, paths, suites, cfg, out, ran, skipped, n_scale, force)
        if "perf" in suites:
            _perf_phase(variants, paths, cfg, out, ran, skipped, n_scale)

        report = _aggregate_report(
            out, registry, gate_policy, env_info, paths, {v.name: v for v in variants}
        )
        report_path = out / "report.json"
        results = report.variants
        recommended = report.recommended
        payload = {
            "ok": True,
            "report": _portable(report_path),
            "variants": [r.variant.name for r in results],
            "ran": ran,
            "skipped": skipped,
            "recommended": recommended,
        }
        text = (
            f"Wrote {_portable(report_path)}. Ran {len(ran)} suite run(s), "
            f"skipped {len(skipped)} finished. Recommended: {recommended or 'none'}."
        )
        _emit(json_out, payload, text)


def _merge_env_run(env_path: Path, run_now: dict[str, Any]) -> dict[str, Any]:
    """The run settings to record. An existing env.json must agree on suite, n-scale and policy.

    Variants and suites accumulate across runs into the same results directory. Settings that
    change the numbers may not change silently (review Q3), so a mismatch is refused.
    """
    if not env_path.is_file():
        return run_now
    prior_run = json.loads(env_path.read_text(encoding="utf-8")).get("run", {})
    for key in ("suite", "n_scale", "policy"):
        if prior_run.get(key) != run_now[key]:
            raise CliError(
                f"{_portable(env_path)} records a different {key}. Use a new --out directory "
                "for different settings."
            )
    merged = dict(run_now)
    merged["variants"] = sorted(set(prior_run.get("variants", [])) | set(run_now["variants"]))
    merged["suites"] = [
        s for s in SUITE_NAMES if s in set(prior_run.get("suites", [])) | set(run_now["suites"])
    ]
    return merged


def _run_history(env_path: Path, env_info: dict[str, Any], run_now: dict[str, Any]) -> list[Any]:
    """Earlier run records from env.json plus this run, so that reruns stay visible."""
    prior: list[Any] = []
    if env_path.is_file():
        prior = json.loads(env_path.read_text(encoding="utf-8")).get("run_history", [])
    entry = {
        "utc_time": env_info.get("utc_time"),
        "git_sha": env_info.get("git_sha"),
        "suites": run_now["suites"],
        "variants": run_now["variants"],
    }
    return [*prior, entry]


def _aggregate_report(
    out: Path,
    registry: Registry,
    gate_policy: GatePolicy,
    env_info: dict[str, Any],
    paths: dict[str, Path],
    pinned: dict[str, Variant],
) -> QuantReport:
    """Merge every suite file in out into report.json, with the gate decisions, and write it."""
    results = _assemble(registry, paths, out, pinned=pinned)
    decisions = evaluate(results, registry.reference, gate_policy)
    recommended = recommend(decisions, results)
    report = build_report(
        model_family=registry.family,
        reference=registry.reference,
        results=results,
        decisions=decisions,
        recommended=recommended,
        env_info=env_info,
    )
    write_report(report, out / "report.json")
    return report


def _report_path(path: Path) -> Path:
    """Accept a results directory (uses report.json inside it) or a report file."""
    candidate = path / "report.json" if path.is_dir() else path
    if not candidate.is_file():
        raise CliError(f"no report at {_portable(candidate)}; run `quantgate eval` first")
    return candidate


audit_app = typer.Typer(
    help="Refusal classifier audit: export a seeded sample for hand labelling.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(audit_app, name="audit")


@audit_app.command("export")
def audit_export(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Variant registry YAML.")
    ] = None,
    only: Annotated[
        str | None, typer.Option("--only", help="Comma-separated variants (default: all).")
    ] = None,
    n: Annotated[int, typer.Option("--n", help="Items sampled per variant.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 0,
    raw_root: Annotated[
        Path | None, typer.Option("--raw-root", help="Raw completions root (local).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="CSV to write (local, never committed).")
    ] = None,
    json_out: JsonOption = False,
) -> None:
    """Write a CSV of sampled refusal items for hand labelling. It holds no completion text.

    Each row gives the prompt id, the regex flag, the SHA-256 of the completion, and the local
    path and line of the raw record. The CSV contains local paths, so it must not be committed.
    """
    from quantgate.audit import default_audit_out, export_audit

    with _guard(json_out):
        if n <= 0:
            raise CliError("--n must be positive")
        registry = load_registry(_registry_arg(config))
        known = [v.name for v in registry.variants]
        names = known if only is None else [s.strip() for s in only.split(",") if s.strip()]
        unknown = sorted(set(names) - set(known))
        if unknown:
            raise CliError(f"unknown variant(s): {', '.join(unknown)}")
        summary = export_audit(
            (raw_root_path() if raw_root is None else raw_root).expanduser(),
            names,
            n,
            seed,
            (default_audit_out() if out is None else out).expanduser(),
        )
        lines = [f"Wrote {summary['rows']} row(s) to {summary['out']}."]
        for variant, count in summary["exported"].items():
            lines.append(f"  {variant}: {count} sampled")
        if summary["skipped"]:
            lines.append(f"  no raw records yet for: {', '.join(summary['skipped'])}")
        lines.append("The CSV holds local paths. Do not commit it.")
        _emit(json_out, {"ok": True, **summary}, "\n".join(lines))


@audit_app.command("score")
def audit_score(
    csv_path: Annotated[Path, typer.Argument(help="A labelled audit CSV (human_label filled).")],
    json_out: JsonOption = False,
) -> None:
    """Agreement of the regex flag with the human labels: precision, recall, specificity.

    The numbers come from the labelled CSV. Nothing is typed by hand.
    """
    from quantgate.audit import score_audit

    with _guard(json_out):
        if not csv_path.is_file():
            raise CliError(f"no audit CSV at {csv_path}")
        summary = score_audit(csv_path)
        lines = [
            f"rows {summary['rows']}, labelled {summary['labelled']}, "
            f"unclear {summary['unclear']}, blank {summary['blank']}",
            f"confusion: {summary['confusion']}",
        ]
        for key in ("precision_pct", "recall_pct", "specificity_pct"):
            r = summary[key]
            if r["value"] is None:
                lines.append(f"{key}: n/a (no denominator)")
            else:
                lines.append(
                    f"{key}: {r['value']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}] (n={r['n']})"
                )
        _emit(json_out, {"ok": True, **summary}, "\n".join(lines))


@app.command()
def gate(
    report: Annotated[Path, typer.Argument(help="Path to a QuantReport JSON.")],
    policy: Annotated[Path | None, typer.Option("--policy", "-p", help="Gate policy YAML.")] = None,
    json_out: JsonOption = False,
) -> None:
    """Apply the gate policy to a report. Exit 0 if every variant passes, 1 otherwise.

    The policy is applied fresh. If the decisions stored in the report differ, that is shown in
    the output.
    """
    with _guard(json_out):
        rep: QuantReport = load_report(_report_path(report))
        pol = load_policy(_policy_arg(policy))
        decisions = gate_report(rep, pol)
        if not decisions:
            raise CliError("the report has no candidate variants to gate")
        recommended = recommend(decisions, rep.variants)
        stored = {
            (d.variant, d.passed, tuple(d.failed_rules), d.hidden_danger) for d in rep.decisions
        }
        fresh = {(d.variant, d.passed, tuple(d.failed_rules), d.hidden_danger) for d in decisions}
        all_pass = all(d.passed for d in decisions)
        hidden = [d.variant for d in decisions if d.hidden_danger]
        payload = {
            "ok": all_pass,
            "reference": rep.reference,
            "decisions": [d.model_dump() for d in decisions],
            "recommended": recommended,
            "hidden_danger": hidden,
            "stored_decisions_agree": stored == fresh,
            "exit_code": 0 if all_pass else 1,
        }
        width = max(len(d.variant) for d in decisions)
        lines = [f"{'variant':<{width}}  gate                        hidden  failed rules"]
        for d in decisions:
            failed = ", ".join(d.failed_rules) if d.failed_rules else "-"
            danger = "YES" if d.hidden_danger else "no"
            lines.append(f"{d.variant:<{width}}  {gate_label(d):<28}  {danger:<6}  {failed}")
        lines.append(f"Recommended: {recommended or 'none'}")
        if stored != fresh:
            lines.append("Note: the stored decisions differ from this policy's decisions.")
        _emit(json_out, payload, "\n".join(lines))
        if not all_pass:
            raise typer.Exit(code=1)


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Results directory.")] = Path("results"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where Markdown, plots and the site are written.")
    ] = Path("site"),
    readme: Annotated[
        Path | None,
        typer.Option("--readme", help="Also splice the results block into this README."),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Rebuild report.json from every per-variant suite file first (no model run).",
        ),
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Variant registry YAML (with --rebuild).")
    ] = None,
    policy: Annotated[
        Path | None, typer.Option("--policy", "-p", help="Gate policy YAML (with --rebuild).")
    ] = None,
    json_out: JsonOption = False,
) -> None:
    """Render Markdown tables, Pareto plots and the static site from committed results."""
    with _guard(json_out):
        if rebuild:
            if not results.is_dir():
                raise CliError("--rebuild needs the results directory")
            registry_path = _registry_arg(config)
            registry = load_registry(registry_path)
            lock_entries = read_lock(_lock_default(registry_path))
            pinned = {v.name: apply_lock(v, lock_entries) for v in registry.variants}
            env_path = results / "env.json"
            env_info: dict[str, Any] = (
                json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
            )
            _aggregate_report(
                results,
                registry,
                load_policy(_policy_arg(policy)),
                env_info,
                {},
                pinned,
            )
        report_path = _report_path(results)
        rep = load_report(report_path)
        from quantgate.plots import write_report_plots  # matplotlib is an extra

        plot_paths = write_report_plots(rep, out_dir)
        written = write_site(rep, out_dir, plot_paths)
        env_src = report_path.parent / "env.json"
        if env_src.is_file():
            shutil.copyfile(env_src, out_dir / "env.json")
            written.append(out_dir / "env.json")
        written += plot_paths
        if readme is not None:
            text = readme.read_text(encoding="utf-8")
            readme.write_text(update_readme(text, render_readme_block(rep)), encoding="utf-8")
            written.append(readme)
        payload = {"ok": True, "written": [_portable(p) for p in written]}
        text = "Wrote:\n" + "\n".join(f"  {_portable(p)}" for p in written)
        _emit(json_out, payload, text)


DEMO_LADDER: tuple[tuple[str, float, int], ...] = (
    ("synthetic-fp16", 0.0, 1_000_000_000),
    ("synthetic-q8", 0.01, 530_000_000),
    ("synthetic-q4", 0.05, 380_000_000),
    ("synthetic-q2", 0.30, 220_000_000),
)


def _demo_items() -> tuple[list[str], list[McqItem], list[tuple[str, str]], list[str], list[str]]:
    """Scripted demo inputs: passages, MCQ items, GSM8K pairs, harmful and benign prompts."""
    passages = [f"synthetic passage {i} about quantization drift." for i in range(4)]
    mcq = [
        McqItem(
            item_id=f"demo-{i}",
            question=f"synthetic question {i}?",
            choices=["alpha", "beta", "gamma", "delta"],
            labels=["A", "B", "C", "D"],
            answer_key="A",
        )
        for i in range(20)
    ]
    # run_gsm8k takes raw answer fields, so the gold number follows a '####' marker.
    gsm = [(f"synthetic arithmetic problem {i}", "worked steps\n#### 7") for i in range(5)]
    harmful = [f"synthetic harmful-style prompt {i}" for i in range(10)]
    benign = [f"synthetic benign prompt {i}" for i in range(5)]
    return passages, mcq, gsm, harmful, benign


def _demo_results() -> list[VariantResult]:
    """Run every suite on FakeBackend perturbations.

    Raises NotImplementedError while a module it needs is still a stub.
    """
    passages, mcq, gsm, harmful, benign = _demo_items()
    ref_backend = FakeBackend(seed=0, perturbation=0.0)
    results: list[VariantResult] = []
    for name, perturbation, size in DEMO_LADDER:
        backend = (
            ref_backend if perturbation == 0.0 else FakeBackend(seed=0, perturbation=perturbation)
        )
        metrics: dict[str, MetricValue] = {}
        for mv in run_kl(ref_backend, backend, passages, 32):
            metrics[mv.name] = mv
        arc = run_mcq(backend, mcq)
        metrics[arc.name] = arc
        gsm_mv = run_gsm8k(backend, gsm, 16, 0)
        metrics[gsm_mv.name] = gsm_mv
        for mv in run_refusal(backend, harmful, benign, 16, 0, raw_dir=None):
            metrics[mv.name] = mv
        perf_cfg = PerfConfig(prompt_tokens=32, gen_tokens=8, warmup=0, repeats=2, threads=1)
        raw = measure_in_process(backend, perf_cfg)
        for mv in _perf_metrics(raw, perf_cfg.repeats, None, need_rss=False):
            metrics[mv.name] = mv
        variant = Variant(
            name=name,
            repo_id="synthetic/none",
            filename=f"{name}.gguf",
            file_bytes=size,
            is_reference=perturbation == 0.0,
        )
        results.append(VariantResult(variant=variant, metrics=metrics, raw_path=None))
    return results


@app.command()
def demo(
    json_out: JsonOption = False,
) -> None:
    """Run the full pipeline on a scripted FakeBackend in under 30 s, with no downloads.

    Output is labelled SYNTHETIC. Exits 2 with a clear message while a module it needs is still
    a stub.
    """
    with _guard(json_out):
        try:
            results = _demo_results()
        except NotImplementedError as exc:
            raise CliError(
                "demo cannot run yet: a module it needs is not implemented (NotImplementedError)"
            ) from exc
        policy_path = _policy_arg(None)
        policy = load_policy(policy_path)
        reference = DEMO_LADDER[0][0]
        decisions = evaluate(results, reference, policy)
        recommended = recommend(decisions, results)
        env_info: dict[str, Any] = {
            "synthetic": True,
            "label": "SYNTHETIC demo: FakeBackend perturbations, no model files, not a measurement",
            "run": {
                "sizes": "tiny, scripted FakeBackend, no model files",
                "suites": list(SUITE_NAMES),
                "policy": "bundled default" if is_bundled(policy_path) else _portable(policy_path),
            },
        }
        rep = build_report(
            model_family="synthetic-demo",
            reference=reference,
            results=results,
            decisions=decisions,
            recommended=recommended,
            env_info=env_info,
        )
        payload = {
            "ok": True,
            "synthetic": True,
            "report": json.loads(rep.model_dump_json()),
            "exit_code": 0,
        }
        _emit(json_out, payload, render_markdown(rep))


if __name__ == "__main__":  # pragma: no cover
    app()
