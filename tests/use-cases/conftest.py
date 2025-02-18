# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
import subprocess
from typing import Callable

import pytest

FNAMES = [
    "pipeline.yaml",
    "startscript",
]


@pytest.fixture
def check_folder_structure() -> Callable:
    """Verify that the use case folder complies with some predefined structure."""

    def _check_structure(root: str):
        for fname in FNAMES:
            fpath = os.path.join(root, fname)
            assert os.path.isfile(fpath), f"'{fname}' is missing in '{fpath}'"

    return _check_structure


@pytest.fixture
def install_requirements() -> Callable:
    """Install requirements.txt, if present in root folder."""

    def _install_reqs(root: str, env_prefix: str):
        req_path = os.path.join(root, "requirements.txt")
        if os.path.isfile(req_path):
            cmd = f"{env_prefix}/bin/pip install --no-cache-dir -r {req_path}"
            subprocess.run(cmd.split(), check=True)

    return _install_reqs
