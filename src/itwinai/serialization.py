from typing import Dict, Any
import abc
import json
import inspect

from .types import MLModel


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


class SerializationError(Exception):
    ...


class Serializable:
    parameters: Dict[Any, Any] = None

    def save_parameters(self, **kwargs) -> None:
        """Simplified way to store constructor arguments in as class
        attributes. Keeps track of the parameters to enable
        YAML/JSON serialization.
        """
        if self.parameters is None:
            self.parameters = {}
        self.parameters.update(kwargs)

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def update_parameters(self, **kwargs) -> None:
        """Updates stored parameters."""
        self.save_parameters(**kwargs)

    def to_dict(self) -> Dict:
        """Returns a dict serialization of the current object."""
        self._validate_parameters()
        class_path = fullname(self)
        init_args = dict()
        for par_name, par in self._saved_constructor_parameters().items():
            init_args[par_name] = self._recursive_serialization(par, par_name)
        return dict(class_path=class_path, init_args=init_args)

    def _validate_parameters(self) -> None:
        if self.parameters is None:
            raise SerializationError(
                f"{self.__class__.__name__} cannot be serialized "
                "because its constructor arguments were not saved. "
                "Please add 'self.save_parameters(param_1=param_1, "
                "..., param_n=param_n)' as first instruction of its "
                "constructor."
            )

        init_params = inspect.signature(self.__init__).parameters.items()

        # Check that all non-default parameters are in self.parameters
        non_default_par = list(filter(
            lambda p: p[0] != 'self' and p[1].default == inspect._empty,
            init_params
        ))
        non_default_par_names = list(map(lambda p: p[0], non_default_par))
        for par_name in non_default_par_names:
            if self.parameters.get(par_name) is None:
                raise SerializationError(
                    f"Required parameter '{par_name}' of "
                    f"{self.__class__.__name__} class not present in "
                    "saved parameters. "
                    "Please add 'self.save_parameters(param_1=param_1, "
                    "..., param_n=param_n)' as first instruction of its "
                    f"constructor, including also '{par_name}'."
                )

        # # Check that all params in self.parameters match with the signature
        # init_par_nam = set(map(lambda x: x[0], init_params))
        # sav_par_nam = set(self.parameters.keys())
        # if len(init_par_nam.intersection(sav_par_nam)) != len(sav_par_nam):
        #     raise SerializationError(
        #         "Some parameters saved with "
        #         "'self.save_parameters(param_1=param_1, "
        #         "..., param_n=param_n)' "
        #         "Are unused not present in the constructor of "
        #         f"'{self.__class__.__name__}' class. Please remove them."
        #     )

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
            par_name: self.parameters[par_name] for par_name in init_par_nam
            if self.parameters.get(par_name, inspect._empty) != inspect._empty
        }

    def _recursive_serialization(self, item: Any, item_name: str) -> Any:
        if isinstance(item, (tuple, list)):
            return [self._recursive_serialization(x, item_name) for x in item]
        elif isinstance(item, dict):
            return {
                k: self._recursive_serialization(v, item_name)
                for k, v in item.items()
            }
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


class ModelLoader(abc.ABC, Serializable):
    """Loads a machine learning model from somewhere."""

    def __init__(self, model_uri: str) -> None:
        super().__init__()
        self.model_uri = model_uri

    @abc.abstractmethod
    def __call__(self) -> MLModel:
        """Loads model from model URI."""
