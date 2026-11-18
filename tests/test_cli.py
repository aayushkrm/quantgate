"""The CLI exposes the five commands and their help text works."""

import pytest
from typer.testing import CliRunner

from quantgate.cli import app

runner = CliRunner()

COMMANDS = ["fetch", "eval", "gate", "report", "demo"]


def test_top_level_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
