"""
Tests for MNIST use case.

Intended to be integration tests, to make sure that updates in the code base
do not break use cases' workflows.
"""

import pytest


@pytest.mark.integration
def test_training_workflow():
    """
    Run training workflow in an end-to-end manner and verify that
    everything works as expected.
    """
    # TODO: complete
    assert 1 == 1
