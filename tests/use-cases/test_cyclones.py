"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest
import subprocess

# TODO: add tests for use case folder format:
#   - structure
#   - naming convention
#   - file exist


@pytest.mark.integration
def test_cyclones_train():
    """
    Test MNIST training workflow(s) by running it end-to-end.
    """
    workflows = [
        "./use-cases/cyclones/workflows/training-workflow.yml",
    ]

    for workflow in workflows:
        cmd = f"micromamba run -p ./.venv python run-workflow.py -f {workflow}"
        subprocess.run(cmd.split(), check=True)
        subprocess.run(cmd.split() + ["--cwl"], check=True)