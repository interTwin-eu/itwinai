"""
Utilities for itwinai package.
"""
import os
import yaml

from collections.abc import MutableMapping
from typing import Dict
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def load_yaml(path: str) -> Dict:
    """Load YAML file as dict.

    Args:
        path (str): path to YAML file.

    Raises:
        exc: yaml.YAMLError for loading/parsing errors.

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


def load_yaml_with_deps_from_file(path: str) -> DictConfig:
    """
    Load YAML file with OmegaConf and merge it with its dependencies
    specified in the `conf-dependencies` field.
    Assume that the dependencies live in the same folder of the
    YAML file which is importing them.

    Args:
        path (str): path to YAML file.

    Raises:
        exc: yaml.YAMLError for loading/parsing errors.

    Returns:
        DictConfig: nested representation of parsed YAML file.
    """
    yaml_conf = load_yaml(path)
    use_case_dir = os.path.dirname(path)
    deps = []
    if yaml_conf.get("conf-dependencies"):
        for dependency in yaml_conf["conf-dependencies"]:
            deps.append(load_yaml(os.path.join(use_case_dir, dependency)))

    return OmegaConf.merge(yaml_conf, *deps)


def load_yaml_with_deps_from_dict(dict_conf, use_case_dir) -> DictConfig:
    deps = []

    if dict_conf.get("conf-dependencies"):
        for dependency in dict_conf["conf-dependencies"]:
            deps.append(load_yaml(os.path.join(use_case_dir, dependency)))

    return OmegaConf.merge(dict_conf, *deps)


def dynamically_import_class(name: str):
    """
    Dynamically import class by module path.
    Adapted from https://stackoverflow.com/a/547867

    Args:
        name (str): path to the class (e.g., mypackage.mymodule.MyClass)

    Returns:
        __class__: class object.
    """
    module, class_name = name.rsplit(".", 1)
    mod = __import__(module, fromlist=[class_name])
    klass = getattr(mod, class_name)
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
