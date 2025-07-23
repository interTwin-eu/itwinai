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
import io
import logging
import random
import sys
import time
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Tuple, Type
from urllib.parse import urlparse

import requests
import typer
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf, errors
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from .constants import adjectives, names

if TYPE_CHECKING:
    from .loggers import Logger

py_logger = logging.getLogger(__name__)


def normalize_tracking_uri(uri: str) -> str:
    """Normalize a tracking URI to a valid URI.
    If the URI is empty, it will be treated as the current directory.

    Args:
        uri (str): URI to normalize.

    Returns:
        str: Normalized URI.
    """
    if not uri:
        # treat uri as current path if empty string
        py_logger.warning("Empty tracking URI provided, using current directory.")
        uri = "."
    parsed = urlparse(uri)
    # If scheme is empty, treat as path
    if parsed.scheme == "":
        return Path(uri).resolve().as_uri()
    # If url is a file, normalize the path portion
    if parsed.scheme == "file":
        return Path(parsed.path).resolve().as_uri()
    # Otherwise (http, https, etc.) leave as it is
    return uri


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
        return any(p[1].kind == p[1].VAR_POSITIONAL for p in self.func_params)

    @property
    def has_kwargs(self) -> bool:
        """Checks if the function has ``**kwargs`` parameter."""
        return any(p[1].kind == p[1].VAR_KEYWORD for p in self.func_params)

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
        return tuple(p[0] for p in required_params)

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
            abs_path = Path(path).resolve()
            updated_args[i] = f"{prefix}={abs_path}"
            sys.path.append(str(abs_path))
            break
        elif arg in {"--config-path", "-cp"}:
            # Handle the case where the path is in the next argument
            abs_path = Path(updated_args[i + 1]).resolve()
            updated_args[i + 1] = str(abs_path)
            sys.path.append(str(abs_path))
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


def time_and_log(
    func: Callable,
    logger: "Logger",
    identifier: str,
    step: int | None = None,
    destroy_current_logger_context: bool = False,
) -> Any:
    """Time and log the execution of a function (using time.monotonic()).

    Args:
        func (Callable): function to execute, time and log, expects zero arguments. Use
            `partial` from `functools` if you need to add arguments.
        logger
        identifier (str): identifier for the logged metric
        step (int | None): step for logging. Defaults to None.
        destroy_current_logger_context (bool): Whether to destroy the current logger context.
            Default is False.

    Returns:
        result (Any): result of the function call
    """
    if step is None:
        py_logger.warning("No explicit step was provided for timing and logging!")

    # Using monotonic time to avoid time drift
    start_time = time.monotonic()
    result = func()
    end_time = time.monotonic()
    elapsed_time = end_time - start_time

    if logger.is_initialized and destroy_current_logger_context:
        py_logger.warning(f"Destroying logger context for timing {identifier}.")

        logger.destroy_logger_context()

        logger.is_initialized = False

    if not logger.is_initialized:
        py_logger.warning(
            f"Logger context not initialized for timing {identifier}. Setting context."
        )

        logger.create_logger_context()

    logger.log(
        item=elapsed_time,
        identifier=identifier,
        kind="metric",
        step=step,
    )

    return result


def filter_pipeline_steps(pipeline_cfg: DictConfig, pipe_steps: List[Any]) -> None:
    """Filters the steps in the pipeline configuration, `pipeline_cfg`, using `pipe_steps`,
    and validates the provided `pipe_steps`. Changes the `pipeline_cfg` object in-place. In
    the event of a validation error, the program is exited, as this is expected to be used
    from the CLI.
    """
    if "steps" not in pipeline_cfg:
        py_logger.error(
            "Pipelines have to contain the key `steps`, but the given pipeline does not. Make"
            " sure that your pipeline has at least one step. Are you sure you used the right"
            " pipe key?"
        )
        raise typer.Exit(1)

    any_numeric_steps = any(isinstance(step, int) for step in pipe_steps)
    any_string_steps = any(isinstance(step, str) for step in pipe_steps)
    if any_numeric_steps and isinstance(pipeline_cfg.steps, DictConfig):
        py_logger.error(
            "We don't support numbered indices for named steps. Filter steps using their"
            " names instead of indices."
        )
        raise typer.Exit(1)
    if any_string_steps and isinstance(pipeline_cfg.steps, ListConfig):
        py_logger.error(
            "The steps in the given pipeline are unnamed, but some of the `pipe_steps` are"
            " named. This is not supported. Use indices to filter steps when the steps are"
            " unnamed."
        )
        raise typer.Exit(1)

    try:
        pipeline_cfg.steps = [pipeline_cfg.steps[step] for step in pipe_steps]
        py_logger.info(f"Successfully selected steps {pipe_steps}")
    except errors.ConfigKeyError:
        py_logger.error(
            "Could not find all selected steps. Please ensure that all steps exist and that"
            f" you provided to the dotpath to them. \n\tSteps provided: {pipe_steps}."
            f"\n\tValid steps: {pipeline_cfg.steps.keys()}"
        )
        raise typer.Exit(1)


def retrieve_remote_omegaconf_file(url: str) -> DictConfig | ListConfig:
    """Fetches and parses a remote OmegaConf configuration file.

    Args:
       url: URL to the raw configuration file (YAML/JSON format), e.g. raw GitHub link.

    Returns:
       Parsed OmegaConf configuration as DictConfig or ListConfig.

    Raises:
       typer.Exit: If the request to the URL or the parsing fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except HTTPError as exception:
        py_logger.error(f"Failed to fetch data from '{url}' - '{exception.response.text}'")
        raise typer.Exit(1)
    except ConnectionError:
        py_logger.error(
            f"Failed to connect to '{url}' - Check your internet connection or if the server"
            " is available. Did you write the URL correctly?"
        )
        raise typer.Exit(1)
    except Timeout:
        py_logger.error(f"Request to '{url}' timed out - The server took too long to respond")
        raise typer.Exit(1)
    except RequestException as exception:
        py_logger.error(f"Failed to fetch data from '{url}' - {str(exception)}")
        raise typer.Exit(1)

    response_io_stream = io.StringIO(response.text)
    try:
        cfg = OmegaConf.load(response_io_stream)
    except Exception as exception:
        py_logger.error(
            f"Failed to load configuration from '{url}'. Did you remember to pass a raw file?"
            f"\nException: '{str(exception)}'"
        )
        raise typer.Exit(1)

    return cfg
