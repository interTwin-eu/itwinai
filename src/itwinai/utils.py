"""
Utilities for itwinai package.
"""
from typing import Dict, Type, Callable, Tuple, Hashable
import sys
import inspect
from collections.abc import MutableMapping
import yaml


def load_yaml(path: str) -> Dict:
    """Load YAML file as dict.

    Args:
        path (str): path to YAML file.

    Raises:
        yaml.YAMLError: for loading/parsing errors.

    Returns:
        Dict: nested dict representation of parsed YAML file.
    """
    with open(path, "r", encoding="utf-8") as yaml_file:
        try:
            loaded_config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
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
        print(
            f"Module not found when trying to dynamically import '{name}'. "
            "Make sure that the module's file is reachable from your current "
            "directory."
        )
        raise err
    except Exception as err:
        print(
            f"Exception occurred when trying to dynamically import '{name}'. "
            "Make sure that the module's file is reachable from your current "
            "directory and that the class is present in that module."
        )
        raise err

    return klass


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
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
        return any(map(
            lambda p: p[1].kind == p[1].VAR_POSITIONAL,
            self.func_params
        ))

    @property
    def has_kwargs(self) -> bool:
        """Checks if the function has ``**kwargs`` parameter."""
        return any(map(
            lambda p: p[1].kind == p[1].VAR_KEYWORD,
            self.func_params
        ))

    @property
    def required_params(self) -> Tuple[str]:
        """Names of required parameters. Class method's 'self' is skipped."""
        required_params = list(filter(
            lambda p: (p[0] != 'self' and p[1].default == inspect._empty
                       and p[1].kind != p[1].VAR_POSITIONAL
                       and p[1].kind != p[1].VAR_KEYWORD),
            self.func_params
        ))
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


def str_to_slice(interval: str) -> slice:
    """Transform string interval to Python slice.
    Example: "1:17:3" -> slice(1,17,3)

    Args:
        interval (str): interval to parse.

    Raises:
        ValueError: when interval is invalid.

    Returns:
        slice: parsed slice.
    """
    import re
    # TODO: add support for slices starting with empty index
    # e.g., :20:3
    if not re.match(r"\d+(:\d+)?(:\d+)?", interval):
        raise ValueError(
            f"Received invalid interval for slice: '{interval}'"
        )
    if ":" in interval:
        return slice(*map(
            lambda x: int(x.strip()) if x.strip() else None,
            interval.split(':')
        ))
    return int(interval)


def clear_key(
        my_dict: Dict,
        dict_name: str,
        key: Hashable,
        complain: bool = True
) -> Dict:
    """Remove key from dictionary if present and complain.

    Args:
        my_dict (Dict): Dictionary.
        dict_name (str): name of the dictionary.
        key (Hashable): Key to remove.
    """
    if key in my_dict:
        if complain:
            print(
                f"Field '{key}' should not be present "
                f"in dictionary '{dict_name}'"
            )
        del my_dict[key]
    return my_dict
