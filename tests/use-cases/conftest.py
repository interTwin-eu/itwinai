import os
import pytest
import subprocess

pytest.TORCH_PREFIX = './.venv-pytorch'
pytest.TF_PREFIX = './.venv-tf'

FNAMES = [
    'pipeline.yaml',
    'startscript',
]


@pytest.fixture
def check_folder_structure():
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
def install_requirements():
    """Install requirements.txt, if present in root folder."""
    def _install_reqs(root: str, env_prefix: str):
        req_path = os.path.join(root, 'requirements.txt')
        if os.path.isfile(req_path):
            cmd = (f"micromamba run -p {env_prefix} "
                   f"pip install -r {req_path}")
            subprocess.run(cmd.split(), check=True)
    return _install_reqs
