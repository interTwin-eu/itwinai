import os
import pytest


@pytest.fixture
def torch_env() -> str:
    """
    If TORCH_ENV env variable is defined, it overrides the default
    torch virtual environment name. Otherwise, fall back
    to './.venv-pytorch'.

    Returns absolute path to torch virtual environment.
    """
    if os.environ.get('TORCH_ENV') is None:
        env_p = './.venv-pytorch'
    else:
        env_p = str(os.environ.get('TORCH_ENV'))
    return os.path.abspath(env_p)


@pytest.fixture
def tf_env() -> str:
    """
    If TF_ENV env variable is defined, it overrides the default
    torch virtual environment name. Otherwise, fall back
    to './.venv-tf'.

    Returns absolute path to torch virtual environment.
    """
    if os.environ.get('TF_ENV') is None:
        env_p = './.venv-tf'
    else:
        env_p = str(os.environ.get('TF_ENV'))
    return os.path.abspath(env_p)
