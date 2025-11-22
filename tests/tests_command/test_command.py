"""Tests of command lines"""

import subprocess


def test_command_lines():
    """Test command lines."""

    cmd = "pyclupan --help"
    subprocess.call(cmd.split())
    cmd = "pyclupan-calc --help"
    subprocess.call(cmd.split())
