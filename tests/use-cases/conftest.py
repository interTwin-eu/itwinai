import os
from typing import Callable
import pytest
import subprocess
import random
import string


FNAMES = [
    'pipeline.yaml',
    'startscript',
]


def rnd_string(len: int = 26):
    return ''.join(random.sample(string.ascii_lowercase, len))


@pytest.fixture
def tmp_test_dir():
    root = '/tmp/pytest'
    os.makedirs(root, exist_ok=True)
    test_dir = os.path.join(root, rnd_string())
    while os.path.exists(test_dir):
        test_dir = os.path.join(root, rnd_string())
    os.makedirs(test_dir, exist_ok=True)

    yield test_dir

    # Optional: remove dir here...


@pytest.fixture
def check_folder_structure() -> Callable:
    """
    Verify that the use case folder complies with some predefined
    structure.
    """
    def _check_structure(root: str):
        for fname in FNAMES:
            fpath = os.path.join(root, fname)
            assert os.path.isfile(fpath), f"'{fname}' is missing in '{fpath}'"
    return _check_structure


@pytest.fixture
def install_requirements() -> Callable:
    """Install requirements.txt, if present in root folder."""
    def _install_reqs(root: str, env_prefix: str):
        req_path = os.path.join(root, 'requirements.txt')
        if os.path.isfile(req_path):
            cmd = f"{env_prefix}/bin/pip install -r {req_path}"
            subprocess.run(cmd.split(), check=True)
    return _install_reqs
