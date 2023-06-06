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
def test_training_workflow():
    """
    Test MNIST training workflow by running it end-to-end.
    """
    cmd = (
        "micromamba run -p ./.venv python run-workflow.py "
        "-f ./use-cases/mnist/training-workflow.yml"
    )
    subprocess.run(cmd.split(), check=True)

    # CWL
    subprocess.run(cmd.split() + ['--cwl'], check=True)


@pytest.mark.integration
def test_inference_workflow():
    """
    Test MNIST inference workflow by running it end-to-end.
    """
    cmd = (
        "micromamba run -p ./.venv python run-workflow.py "
        "-f ./use-cases/mnist/inference-workflow.yml"
    )
    subprocess.run(cmd.split(), check=True)

    # CWL
    subprocess.run(cmd.split() + ['--cwl'], check=True)
