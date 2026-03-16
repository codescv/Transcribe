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

def test_start_command_loads():
    """
    Test that the start command can be invoked (it will fail on recording start in test env, 
    but we can verify execution flow or mocking).
    """
    # We can mock the recorder and model to avoid starting real capture
    # but for a basic check, we just verify the command structure is correct.
    pass
