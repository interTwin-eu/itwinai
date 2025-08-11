# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import functools
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Tuple

import matplotlib
import pandas as pd
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from itwinai.constants import PROFILER_TRACES_DIR_NAME, PROFILING_AVG_NAME

if TYPE_CHECKING:
    from itwinai.torch.trainer import TorchTrainer


# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")

py_logger = logging.getLogger(__name__)


def profile_torch_trainer(method: Callable) -> Callable:
    """Decorator for execute method for components. Profiles function calls and
    stores the result for future analysis (e.g. computation vs. other plots).
    """

    def gather_profiling_data(key_averages: Iterable) -> pd.DataFrame:
        profiling_data = []
        for event in key_averages:
            profiling_data.append(
                {
                    "name": event.key,
                    "node_id": event.node_id,
                    "self_cuda_time_total": event.self_device_time_total,
                    "cuda_time_total": event.device_time_total,
                    "cuda_time_total_str": event.device_time_total_str,
                    "calls": event.count,
                }
            )
        return pd.DataFrame(profiling_data)

    def adjust_wait_and_warmup_epochs(
        training_epochs: int, wait_epochs: int, warmup_epochs: int
    ) -> Tuple[int, int, int]:
        """Validates if the given wait and warmup epochs are compatible and if not,
        adjusts them so they fit. The largest one is iteratively decreased until
        a compatible value is reached.

        Returns:
            int: The resulting number of epochs for doing active profiling
            int: The resulting number of wait epochs, possibly adjusted
            int: The resulting number of warmup epochs, possibly adjusted
        """
        active_epochs = training_epochs - wait_epochs - warmup_epochs
        if active_epochs > 0:
            return active_epochs, wait_epochs, warmup_epochs

        # This can probably be done with a simple math expression, but this was
        # simpler to implement and won't really cause much overhead anyway...
        while active_epochs <= 0:
            if wait_epochs > warmup_epochs:
                wait_epochs -= 1
            else:
                warmup_epochs -= 1
            active_epochs = training_epochs - wait_epochs - warmup_epochs

        if wait_epochs < 0 or warmup_epochs < 0:
            raise ValueError(
                f"Unable to adjust wait and warmup epochs to accomodate the"
                f"given number of training epochs. Was given the following values: "
                f"Training epochs: {training_epochs}, wait epochs: {wait_epochs}"
                f", warmup epochs: {warmup_epochs}"
            )
        py_logger.warning(
            f"Adjusted the given wait and warmup epochs for the profiler - "
            f"wait epochs: {wait_epochs}, warmup epochs: {warmup_epochs}."
        )
        return active_epochs, wait_epochs, warmup_epochs

    @functools.wraps(method)
    def profiled_method(self: "TorchTrainer", *args, **kwargs) -> Any:
        if not self.enable_torch_profiling:
            py_logger.info(
                "Profiling of computation with the PyTorch profiler has been disabled!"
            )
            return method(self, *args, **kwargs)

        active_epochs, wait_epochs, warmup_epochs = adjust_wait_and_warmup_epochs(
            training_epochs=self.epochs,
            wait_epochs=self.profiling_wait_epochs,
            warmup_epochs=self.profiling_warmup_epochs,
        )
        # Set correct values for the profiling epochs
        self.profiling_wait_epochs = wait_epochs
        self.profiling_warmup_epochs = warmup_epochs
        if self.store_torch_profiling_traces:
            trace_handler = tensorboard_trace_handler(
                dir_name=f"{PROFILER_TRACES_DIR_NAME}/{self.run_name}/torch-traces",
                worker_name=f"worker_{self.strategy.global_rank()}",
            )
        else:
            py_logger.warning("Profiling computation without storing the traces!")
            trace_handler = None
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=schedule(
                wait=wait_epochs,
                warmup=warmup_epochs,
                active=active_epochs,
            ),
            on_trace_ready=trace_handler,
            with_modules=True,
        ) as profiler:
            self.profiler = profiler
            result = method(self, *args, **kwargs)

        strategy = self.strategy
        strategy_name = strategy.name

        global_rank = strategy.global_rank()
        num_gpus_global = strategy.global_world_size()

        # Extracting and storing the profiling data
        key_averages = profiler.key_averages()

        profiling_dataframe = gather_profiling_data(key_averages=key_averages)
        profiling_dataframe["strategy"] = strategy_name
        profiling_dataframe["num_gpus"] = num_gpus_global
        profiling_dataframe["global_rank"] = global_rank

        # write profiling averages to mlflow logger
        if self.mlflow_logger is None:
            py_logger.info("No MLflow logger is set, not logging the profiling data!")
            return result

        temp_dir = tempfile.gettempdir()
        csv_path = Path(temp_dir) / f"{PROFILING_AVG_NAME}.csv"
        profiling_dataframe.to_csv(csv_path, index=False)
        self.mlflow_logger.log(
            item=str(csv_path),
            identifier=PROFILING_AVG_NAME,
            kind="artifact",
        )
        csv_path.unlink()  # Remove after logging

        return result

    return profiled_method
