# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Utilities for itwinai package."""

import functools
import inspect
import logging
import os
import random
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Tuple, Type
from urllib.parse import urlparse

import yaml

py_logger = logging.getLogger(__name__)

# Directory names for logging and profiling data
COMPUTATION_DATA_DIR = "computation-data"
EPOCH_TIME_DIR = "epoch-time"
GPU_ENERGY_DIR = "gpu-energy-data"

adjectives = [
    "quantum",
    "relativistic",
    "wavy",
    "entangled",
    "chiral",
    "tachyonic",
    "superluminal",
    "anomalous",
    "hypercharged",
    "fermionic",
    "hadronic",
    "quarky",
    "holographic",
    "dark",
    "force-sensitive",
    "chaotic",
]

names = [
    "neutrino",
    "graviton",
    "muon",
    "gluon",
    "tachyon",
    "quasar",
    "pulsar",
    "blazar",
    "meson",
    "boson",
    "hyperon",
    "starlord",
    "groot",
    "rocket",
    "yoda",
    "skywalker",
    "sithlord",
    "midichlorian",
    "womp-rat",
    "beskar",
    "mandalorian",
    "ewok",
    "vibranium",
    "nova",
    "gamora",
    "drax",
    "ronan",
    "thanos",
    "cosmo",
]


def generate_random_name():
    return f"{random.choice(adjectives)}-{random.choice(names)}"


def load_yaml(path: str) -> Dict:
    """Load YAML file as dict.

    Args:
        path (str): path to YAML file.

    Raises:
        yaml.YAMLError: for loading/parsing errors.

    Returns:
        Dict: nested dict representation of parsed YAML file.
    """
    with open(path, encoding="utf-8") as yaml_file:
        loaded_config = yaml.safe_load(yaml_file)
    return loaded_config


def dynamically_import_class(name: str) -> Type:
    """
    Dynamically import class by module path.
    Adapted from https://stackoverflow.com/a/547867

    Args:
        name (str): path to the class (e.g., mypackage.mymodule.MyClass)

    Returns:
        __class__: class type.
    """
    try:
        module, class_name = name.rsplit(".", 1)
        mod = __import__(module, fromlist=[class_name])
        klass = getattr(mod, class_name)
    except ModuleNotFoundError as err:
        py_logger.error(
            f"Module not found when trying to dynamically import '{name}'. "
            "Make sure that the module's file is reachable from your current "
            "directory."
        )
        raise err
    except Exception as err:
        py_logger.error(
            f"Exception occurred when trying to dynamically import '{name}'. "
            "Make sure that the module's file is reachable from your current "
            "directory and that the class is present in that module."
        )
        raise err

    return klass


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> MutableMapping:
    """Flatten dictionary

    Args:
        d (MutableMapping): nested dictionary to flatten
        parent_key (str, optional): prefix for all keys. Defaults to ''.
        sep (str, optional): separator for nested key concatenation.
            Defaults to '.'.

    Returns:
        MutableMapping: flattened dictionary with new keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class SignatureInspector:
    """Provides the functionalities to inspect the signature of a function
    or a method.

    Args:
        func (Callable): function to be inspected.
    """

    INFTY: int = sys.maxsize

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.func_params = inspect.signature(func).parameters.items()

    @property
    def has_varargs(self) -> bool:
        """Checks if the function has ``*args`` parameter."""
        return any(map(lambda p: p[1].kind == p[1].VAR_POSITIONAL, self.func_params))

    @property
    def has_kwargs(self) -> bool:
        """Checks if the function has ``**kwargs`` parameter."""
        return any(map(lambda p: p[1].kind == p[1].VAR_KEYWORD, self.func_params))

    @property
    def required_params(self) -> Tuple[str]:
        """Names of required parameters. Class method's 'self' is skipped."""
        required_params = list(
            filter(
                lambda p: (
                    p[0] != "self"
                    and p[1].default == inspect._empty
                    and p[1].kind != p[1].VAR_POSITIONAL
                    and p[1].kind != p[1].VAR_KEYWORD
                ),
                self.func_params,
            )
        )
        return tuple(map(lambda p: p[0], required_params))

    @property
    def min_params_num(self) -> int:
        """Minimum number of arguments required."""
        return len(self.required_params)

    @property
    def max_params_num(self) -> int:
        """Max number of supported input arguments.
        If no limit, ``SignatureInspector.INFTY`` is returned.
        """
        if self.has_kwargs or self.has_varargs:
            return self.INFTY
        return len(self.func_params)


def clear_key(my_dict: Dict, dict_name: str, key: Hashable, complain: bool = True) -> Dict:
    """Remove key from dictionary if present and complain.

    Args:
        my_dict (Dict): Dictionary.
        dict_name (str): name of the dictionary.
        key (Hashable): Key to remove.
    """
    if key in my_dict:
        if complain:
            py_logger.warning(
                f"Field '{key}' should not be present in dictionary '{dict_name}'"
            )
        del my_dict[key]
    return my_dict


def make_config_paths_absolute(args: List[str]):
    """Process CLI arguments to make paths specified for `--config-path` or `-cp` absolute.
    Returns the modified arguments list.

    Args:
        args (List[str]): a list of system arguments

    Returns:
        List(str): the updated list of system arguments, where all the config path argument is
            absolute.
    """
    updated_args = args.copy()
    for i, arg in enumerate(updated_args):
        if arg.startswith("--config-path=") or arg.startswith("-cp="):
            prefix, path = arg.split("=", 1)
            abs_path = os.path.abspath(path)
            updated_args[i] = f"{prefix}={abs_path}"
            sys.path.append(abs_path)
            break
        elif arg in {"--config-path", "-cp"}:
            # Handle the case where the path is in the next argument
            abs_path = os.path.abspath(updated_args[i + 1])
            updated_args[i + 1] = abs_path
            sys.path.append(abs_path)
            break
    return updated_args


def get_root_cause(exception: Exception) -> Exception:
    """Recursively extract the first exception in the exception chain."""
    root = exception
    while root.__cause__ is not None:  # Traverse the exception chain
        root = root.__cause__
    return root


def to_uri(path_str: str | Path) -> str:
    """Parse a path and convert it to a URI.

    Args:
        path_str (str): path to convert.

    Returns:
        str: URI.
    """
    if isinstance(path_str, Path):
        return str(Path(path_str).resolve())
    parsed = urlparse(path_str)
    if parsed.scheme:
        # If it has a scheme, assume it's a URI and return as-is
        return path_str
    # Otherwise, make it absolute
    return str(Path(path_str).resolve())


def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
