# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
from pathlib import Path

import pytest


@pytest.fixture
def torch_env() -> str:
    """If TORCH_ENV env variable is defined, it overrides the default
    torch virtual environment name. Otherwise, fall back
    to './.venv-pytorch'.

    Returns absolute path to torch virtual environment.
    """
    env_path = Path(os.environ.get("TORCH_ENV", "./.venv"))
    return str(env_path.resolve())
    # if os.environ.get("TORCH_ENV") is None:
    #     env_p = "./.venv"
    # else:
    #     env_p = str(os.environ.get("TORCH_ENV"))
    # return os.path.abspath(env_p)


@pytest.fixture
def tf_env() -> str:
    """If TF_ENV env variable is defined, it overrides the default
    torch virtual environment name. Otherwise, fall back
    to './.venv-tf'.

    Returns absolute path to torch virtual environment.
    """
    env_path = Path(os.environ.get("TF_ENV", "./.venv"))
    return str(env_path.resolve())
    # if os.environ.get("TF_ENV") is None:
    #     env_p = "./.venv"
    # else:
    #     env_p = str(os.environ.get("TF_ENV"))
    # return os.path.abspath(env_p)
