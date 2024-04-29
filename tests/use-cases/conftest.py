import os
from typing import Callable
import pytest
import subprocess


FNAMES = [
    'pipeline.yaml',
    'startscript',
]


@pytest.fixture
def torch_env() -> str:
    """
    Return absolute path to torch virtual environment parsing it
    from environment variables, if provided, otherwise fall back
    to ``./.venv-pytorch``.
    """
    if os.environ.get('TORCH_ENV') is None:
        env_p = './.venv-pytorch'
    else:
        env_p = os.environ.get('TORCH_ENV')
    return os.path.join(os.getcwd(), env_p)


@pytest.fixture
def tf_env() -> str:
    """
    Return absolute path to tensorflow virtual environment parsing it
    from environment variables, if provided, otherwise fall back
    to ``./.venv-tf``.
    """
    if os.environ.get('TF_ENV') is None:
        env_p = './.venv-tf'
    else:
        env_p = os.environ.get('TF_ENV')
    return os.path.join(os.getcwd(), env_p)


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
