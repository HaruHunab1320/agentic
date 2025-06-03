"""
Tests for the CLI interface
"""

import pytest
from click.testing import CliRunner

from agentic.cli import cli


class TestCLI:
    """Test cases for the CLI interface"""
    
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Agentic" in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
    
    def test_cli_without_args(self):
        """Test CLI without arguments shows banner"""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0 