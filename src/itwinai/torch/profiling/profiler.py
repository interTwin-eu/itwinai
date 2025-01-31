# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import functools
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import matplotlib
import pandas as pd
from torch.profiler import ProfilerActivity, profile, schedule

from itwinai.torch.trainer import TorchTrainer

# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")


def profile_torch_trainer(method: Callable) -> Callable:
    """Decorator for execute method for components. Profiles the communication time
    vs. computation time and stores the result for future analysis.
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
        print(
            f"Warning: adjusted the given wait and warmup epochs for the profiler - "
            f"wait epochs: {wait_epochs}, warmup epochs: {warmup_epochs}."
        )
        return active_epochs, wait_epochs, warmup_epochs

    @functools.wraps(method)
    def profiled_method(self: TorchTrainer, *args, **kwargs) -> Any:
        active_epochs, wait_epochs, warmup_epochs = adjust_wait_and_warmup_epochs(
            training_epochs=self.epochs,
            wait_epochs=self.profiling_wait_epochs,
            warmup_epochs=self.profiling_warmup_epochs,
        )
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=schedule(
                wait=wait_epochs,
                warmup=warmup_epochs,
                active=active_epochs,
            ),
            with_modules=True
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

        profiling_log_dir = Path("scalability-metrics/communication-data")
        profiling_log_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{strategy_name}_{num_gpus_global}_{global_rank}.csv"
        output_path = profiling_log_dir / filename

        print(f"Writing communication profiling dataframe to '{output_path}'.")
        profiling_dataframe.to_csv(output_path)

        return result

    return profiled_method
