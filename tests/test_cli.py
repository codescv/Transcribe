import pytest
from typer.testing import CliRunner
from transcribe.cli import app
import sys
import os

runner = CliRunner()

def test_start_help():
    """Test that help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Start capturing screen audio and transcribing" in result.output

def test_start_continuous_summary_options():
    """Test that continuous summarization options are listed in help."""
    # Typer lists options for the command when calling [command] --help
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "--summary-output" in result.output
    assert "--summary-interval" in result.output
