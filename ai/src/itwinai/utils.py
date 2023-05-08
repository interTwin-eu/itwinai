"""
Utilities for itwinai package.
"""

from collections.abc import MutableMapping


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
        d: MutableMapping,
        parent_key: str = '',
        sep: str = '.'
) -> MutableMapping:
    """
    Flatten dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
