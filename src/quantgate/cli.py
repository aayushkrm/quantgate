"""Command-line interface: `quantgate fetch | eval | gate | report | demo`.

Every command accepts --json. Exit codes: 0 = pass, 1 = check failed (gate rule or hidden
danger), 2 = usage or runtime error. Demo output is labelled SYNTHETIC.
"""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="CI gate for quantized GGUF models: KL drift, accuracy, refusal and CPU speed.",
    no_args_is_help=True,
    add_completion=False,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]


@app.command()
def fetch(
    config: Annotated[Path, typer.Option("--config", "-c", help="Variant registry YAML.")] = Path(
        "configs/variants.yaml"
    ),
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Where GGUF files are stored (outside the repo).")
    ] = Path("~/models"),
    json_out: JsonOption = False,
) -> None:
    """Download every variant in the registry and verify sha256 and size."""
    raise NotImplementedError


@app.command("eval")
def eval_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Variant registry YAML.")] = Path(
        "configs/variants.yaml"
    ),
    suite: Annotated[
        str, typer.Option("--suite", help="Comma-separated subset of: kl,arc,gsm8k,refusal,perf.")
    ] = "kl,arc,gsm8k,refusal,perf",
    out: Annotated[Path, typer.Option("--out", help="Results directory.")] = Path("results"),
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Where GGUF files are stored.")
    ] = Path("~/models"),
    json_out: JsonOption = False,
) -> None:
    """Run the eval suite for every variant and write per-experiment JSON under --out."""
    raise NotImplementedError


@app.command()
def gate(
    report: Annotated[Path, typer.Argument(help="Path to a QuantReport JSON.")],
    policy: Annotated[Path, typer.Option("--policy", "-p", help="Gate policy YAML.")] = Path(
        "configs/gate.yaml"
    ),
    json_out: JsonOption = False,
) -> None:
    """Apply the gate policy to a report. Exit 0 if every variant passes, 1 otherwise."""
    raise NotImplementedError


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Results directory.")] = Path("results"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where Markdown, plots and the site are written.")
    ] = Path("site"),
    json_out: JsonOption = False,
) -> None:
    """Render Markdown tables, Pareto plots and the static site from committed results."""
    raise NotImplementedError


@app.command()
def demo(
    json_out: JsonOption = False,
) -> None:
    """Run the full pipeline on a scripted FakeBackend in under 30 s, with no downloads.

    Output is labelled SYNTHETIC.
    """
    raise NotImplementedError
