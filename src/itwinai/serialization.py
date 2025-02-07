# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import abc
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .type import MLModel
from .utils import SignatureInspector


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


class SerializationError(Exception):
    """Serialization error"""


class Serializable:
    #: Dictionary storing constructor arguments. Needed to serialize the
    #: class to dictionary. Set by ``self.save_parameters()`` method.
    parameters: Dict[Any, Any] = None

    def save_parameters(self, **kwargs) -> None:
        """Simplified way to store constructor arguments in as class
        attributes. Keeps track of the parameters to enable
        YAML/JSON serialization.
        """
        if self.parameters is None:
            self.parameters = {}
        self.parameters.update(kwargs)

        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)

    @staticmethod
    def locals2params(locals: Dict, pop_self: bool = True) -> Dict:
        """Remove ``self`` from the output of ``locals()``.

        Args:
            locals (Dict): output of ``locals()`` called in the constructor
                of a class.
            pop_self (bool, optional): whether to remove ``self``.
                Defaults to True.

        Returns:
            Dict: cleaned ``locals()``.
        """
        if pop_self:
            locals.pop("self", None)
        return locals

    def update_parameters(self, **kwargs) -> None:
        """Updates stored parameters."""
        self.save_parameters(**kwargs)

    def to_dict(self) -> Dict:
        """Returns a dict serialization of the current object."""
        self._validate_parameters()
        class_path = self._get_class_path()
        init_args = dict()
        for par_name, par in self._saved_constructor_parameters().items():
            init_args[par_name] = self._recursive_serialization(par, par_name)
        init_args["_target_"] = class_path

        return init_args

    def _validate_parameters(self) -> None:
        if self.parameters is None:
            raise SerializationError(
                f"{self.__class__.__name__} cannot be serialized "
                "because its constructor arguments were not saved. "
                "Please add 'self.save_parameters(param_1=param_1, "
                "..., param_n=param_n)' as first instruction of its "
                "constructor."
            )

        init_inspector = SignatureInspector(self.__init__)
        for par_name in init_inspector.required_params:
            if self.parameters.get(par_name) is None:
                raise SerializationError(
                    f"Required parameter '{par_name}' of "
                    f"{self.__class__.__name__} class not present in "
                    "saved parameters. "
                    "Please add 'self.save_parameters(param_1=param_1, "
                    "..., param_n=param_n)' as first instruction of its "
                    f"constructor, including also '{par_name}'."
                )

    def _get_class_path(self) -> str:
        class_path = fullname(self)
        if "<locals>" in class_path:
            raise SerializationError(
                f"{self.__class__.__name__} is "
                "defined locally, which is not supported for serialization. "
                "Move the class to a separate Python file and import it "
                "from there."
            )
        return class_path

    def _saved_constructor_parameters(self) -> Dict[str, Any]:
        """Extracts the current constructor parameters from all
        the saved parameters, as some of them may had been added by
        superclasses.

        Returns:
            Dict[str, Any]: subset of saved parameters containing only
            the constructor parameters for this class.
        """
        init_params = inspect.signature(self.__init__).parameters.items()
        init_par_nam = map(lambda x: x[0], init_params)
        return {
            par_name: self.parameters[par_name]
            for par_name in init_par_nam
            if self.parameters.get(par_name, inspect._empty) != inspect._empty
        }

    def _recursive_serialization(self, item: Any, item_name: str) -> Any:
        if isinstance(item, (tuple, list, set)):
            return [self._recursive_serialization(x, item_name) for x in item]
        elif isinstance(item, dict):
            return {k: self._recursive_serialization(v, item_name) for k, v in item.items()}
        elif is_jsonable(item):
            return item
        elif isinstance(item, Serializable):
            return item.to_dict()
        else:
            raise SerializationError(
                f"{self.__class__.__name__} cannot be serialized "
                f"because its constructor argument '{item_name}' "
                "is not a Python built-in type and it does not "
                "extend 'itwinai.serialization.Serializable' class."
            )

    def to_json(self, file_path: Union[str, Path], nested_key: str) -> None:
        """Save a component to JSON file.

        Args:
            file_path (Union[str, Path]): JSON file path.
            nested_key (str): root field containing the serialized object.
        """
        with open(file_path, "w") as fp:
            json.dump({nested_key: self.to_dict()}, fp)

    def to_yaml(self, file_path: Union[str, Path], nested_key: str) -> None:
        """Save a component to YAML file.

        Args:
            file_path (Union[str, Path]): YAML file path.
            nested_key (str): root field containing the serialized object.
        """
        with open(file_path, "w") as fp:
            yaml.dump({nested_key: self.to_dict()}, fp)


class ModelLoader(abc.ABC, Serializable):
    """Loads a machine learning model from somewhere."""

    def __init__(self, model_uri: str) -> None:
        super().__init__()
        self.model_uri = model_uri

    @abc.abstractmethod
    def __call__(self) -> MLModel:
        """Loads model from model URI."""
