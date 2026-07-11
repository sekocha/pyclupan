"""Tests of command lines"""

import os
import subprocess
from pathlib import Path

cwd = Path(__file__).parent


def test_cluster():
    """Test run_cluster."""
    pos = str(cwd) + "/fcc-primitive"
    cmd = "pyclupan-cluster -p " + pos + " -e 0 1 --order 4 --cutoffs 3.0 3.0 3.0"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    assert result.returncode == 0
    os.remove("pyclupan_clusters.yaml")
