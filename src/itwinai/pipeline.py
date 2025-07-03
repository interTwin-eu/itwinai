# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""This module provides the functionalities to execute workflows defined in
in form of pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from .components import BaseComponent, monitor_exec
from .utils import SignatureInspector


class Pipeline(BaseComponent):
    """Executes a set of components arranged as a pipeline.

    Args:
        steps (Dict[str, BaseComponent] | Iterable[BaseComponent]): can be a list or a
            dictionary of steps of type :class:`~itwinai.components.BaseComponent`.
        name (str, optional): name of the pipeline. Defaults to None.
    """

    #: Pipeline steps. Can be a list of ``BaseComponent`` or a dictionary
    #: allowing the user to name each ``BaseComponent``.
    steps: Dict[str, BaseComponent] | Iterable[BaseComponent]

    def __init__(
        self,
        steps: Dict[str, BaseComponent] | Iterable[BaseComponent],
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.save_parameters(steps=steps, name=name)
        self.steps = steps

    def __getitem__(self, subscript: str | int | slice) -> Pipeline:
        if isinstance(subscript, slice):
            # First, convert to list if is a dict
            if isinstance(self.steps, dict):
                steps = list(self.steps.items())
            else:
                steps = self.steps
            # Second, perform slicing
            s = steps[subscript.start : subscript.stop : subscript.step]
            # Third, reconstruct dict, if it is a dict
            if isinstance(self.steps, dict):
                s = dict(s)
            # Fourth, return sliced sub-pipeline, preserving its
            # initial structure
            sliced = self.__class__(steps=s, name=self.name)
            return sliced
        else:
            return self.steps[subscript]

    def __len__(self) -> int:
        return len(self.steps)

    @monitor_exec
    def execute(self, *args) -> Any:
        """Execute components sequentially."""
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
        inspector = SignatureInspector(component.execute)
        if inspector.min_params_num > len(input_args):
            raise TypeError(
                f"Component '{component.name}' received too few "
                f"input arguments: {input_args}. Expected at least "
                f"{inspector.min_params_num}, with names: "
                f"{inspector.required_params}."
            )
        if (
            inspector.max_params_num != inspector.INFTY
            and len(input_args) > inspector.max_params_num
        ):
            raise TypeError(
                f"Component '{component.name}' received too many "
                f"input arguments: {input_args}. Expected at most "
                f"{inspector.max_params_num}."
            )
