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
from typing import Iterable, Dict, Any, Tuple, Union, Optional
import inspect
from .components import BaseComponent, monitor_exec


class Pipeline(BaseComponent):
    """Executes a set of components arranged as a pipeline."""

    steps: Union[Dict[str, BaseComponent], Iterable[BaseComponent]]

    def __init__(
        self,
        steps: Union[Dict[str, BaseComponent], Iterable[BaseComponent]],
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.save_parameters(steps=steps, name=name)
        self.steps = steps

    def __getitem__(self, subscript: Union[str, int, slice]) -> Pipeline:
        if isinstance(subscript, slice):
            # First, convert to list if is a dict
            if isinstance(self.steps, dict):
                steps = list(self.steps.items())
            else:
                steps = self.steps
            # Second, perform slicing
            s = steps[subscript.start:subscript.stop: subscript.step]
            # Third, reconstruct dict, if it is a dict
            if isinstance(self.steps, dict):
                s = dict(s)
            # Fourth, return sliced sub-pipeline, preserving its
            # initial structure
            sliced = self.__class__(
                steps=s,
                name=self.name
            )
            return sliced
        else:
            return self.steps[subscript]

    def __len__(self) -> int:
        return len(self.steps)

    @monitor_exec
    def execute(self, *args) -> Any:
        """"Execute components sequentially."""
        if isinstance(self.steps, dict):
            steps = list(self.steps.values())
        else:
            steps = self.steps

        for step in steps:
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
