"""Tests of sampling derivatives"""

from pathlib import Path

from pyclupan.derivative.run_sample import run_sampling_derivatives

cwd = Path(__file__).parent


def test_run_sampling_derivatives():
    """Test run_sampling_derivatives."""
    files = [
        str(cwd) + "/pyclupan_derivatives_3.yaml",
        str(cwd) + "/pyclupan_derivatives_4.yaml",
    ]

    element_strings = ("Ag", "Au")
    run_sampling_derivatives(files, element_strings, method="all", save_poscars=False)
