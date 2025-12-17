"""Tests of command lines"""

import subprocess


def test_command_lines():
    """Test command lines."""

    cmd = "pyclupan --help"
    subprocess.call(cmd.split())
    # cmd = "pyclupan-sample --help"
    # subprocess.call(cmd.split())
    cmd = "pyclupan-cluster --help"
    subprocess.call(cmd.split())
    cmd = "pyclupan-calc --help"
    subprocess.call(cmd.split())
    cmd = "pyclupan-regression --help"
    subprocess.call(cmd.split())
    cmd = "pyclupan-mc --help"
    subprocess.call(cmd.split())
    cmd = "pyclupan-utils --help"
    subprocess.call(cmd.split())
