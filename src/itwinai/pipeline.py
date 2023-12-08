"""
This module provides the functionalities to execute workflows defined in
in form of pipelines.

TODO:
- Define input and output for components, as in KubeFlow, so that it is
not ambiguous when creating a DAG how to split/merge outputs/inputs.
An alternative is to define additional splitter/merger blocks to manage
the routing of inputs/outputs:

>>> class Router:
>>>     ...
>>> class Splitter(Router):
>>>     ...
>>> class Merger(Router):
>>>     ...

- Create a CLI parser allowing to execute pipelines directly from their
config file serialization, directly from CLI, with dynamic override of
fields, as done with Lightning CLI.
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, Tuple
import inspect
from .components import BaseComponent, monitor_exec


class Pipeline(BaseComponent):
    """Executes a set of components arranged as a pipeline."""

    steps: Iterable[BaseComponent]
    constructor_args: Dict

    def __init__(
        self,
        steps: Iterable[BaseComponent],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.constructor_args = kwargs

    def __getitem__(self, subscript) -> Pipeline:
        if isinstance(subscript, slice):
            s = self.steps[subscript.start:subscript.stop: subscript.step]
            sliced = self.__class__(
                steps=s,
                name=self.name,
                **self.constructor_args
            )
            return sliced
        else:
            return self.steps[subscript]

    def __len__(self) -> int:
        return len(self.steps)

    @monitor_exec
    def execute(self, *args) -> Any:
        """"Execute components sequentially."""
        for step in self.steps:
            step: BaseComponent
            args = self._pack_args(args)
            self.validate_args(args, step)
            args = step.execute(*args)

        return args

    @staticmethod
    def _pack_args(args) -> Tuple:
        """Wraps args in a tuple, if needed."""
        args = () if args is None else args
        if not isinstance(args, tuple):
            args = (args,)
        return args

    @staticmethod
    def validate_args(input_args: Tuple, component: BaseComponent):
        """Verify that the number of input args provided to some component
        match with the number of the non-default args in the component.

        Args:
            input_args (Tuple): input args to be fed to the component.
            component (BaseComponent): component to be executed.

        Raises:
            RuntimeError: in case of args mismatch.
        """
        comp_params = inspect.signature(component.execute).parameters.items()
        non_default_par = list(filter(
            lambda p: p[0] != 'self' and p[1].default == inspect._empty,
            comp_params
        ))
        non_default_par_names = list(map(lambda p: p[0], non_default_par))
        if len(non_default_par) != len(input_args):
            raise RuntimeError(
                "Mismatch into the number of non-default parameters "
                f"of execute method of '{component.name}' component "
                f"({non_default_par_names}), and the number of arguments "
                f"it received as input: {input_args}."
            )
