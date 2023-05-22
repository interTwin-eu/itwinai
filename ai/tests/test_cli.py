"""
Test itwinai CLI.
"""

import subprocess


def test_datasets_viz():
    """
    Test visualization of use case's dataset registry.
    """
    USE_CASE = 'use-cases/mnist/'
    subprocess.run(
        f'itwinai datasets --use-case {USE_CASE}'.split(),
        check=True
    )


def test_workflows_viz():
    """
    Test visualization of use case's workflows.
    """
    USE_CASE = './use-cases/mnist/'
    subprocess.run(
        f'itwinai workflows --use-case {USE_CASE}'.split(),
        check=True
    )
