"""
Utilities for itwinai package.
"""


def dynamically_import_class(name: str):
    """
    Dynamically import class by module path.
    Adapted from https://stackoverflow.com/a/547867

    Args:
        name (str): path to the class (e.g., mypackage.mymodule.MyClass)

    Returns:
        __class__: class object.
    """
    module, class_name = name.rsplit('.', 1)
    mod = __import__(module, fromlist=[class_name])
    klass = getattr(mod, class_name)
    return klass
