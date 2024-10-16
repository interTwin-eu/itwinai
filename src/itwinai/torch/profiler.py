from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import pandas as pd
from torch.profiler import ProfilerActivity, profile, schedule

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)
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
                    "self_cpu_time_total": event.self_cpu_time_total,
                    "cpu_time_total": event.cpu_time_total,
                    "cpu_time_total_str": event.cpu_time_total_str,
                    "self_cuda_time_total": event.self_cuda_time_total,
                    "cuda_time_total": event.cuda_time_total,
                    "cuda_time_total_str": event.cuda_time_total_str,
                    "calls": event.count,
                }
            )
        return pd.DataFrame(profiling_data)

    @functools.wraps(method)
    def profiled_method(self: TorchTrainer, *args, **kwargs) -> Any:

        profiler = profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            with_modules=True,
            schedule=schedule(
                # skip_first=1
                wait=1,
                warmup=2,
                active=100,
            ),
        )
        profiler.start()

        self.profiler = profiler
        try:
            result = method(self, *args, **kwargs)
        finally:
            profiler.stop()

        strategy = self.strategy
        if isinstance(strategy, NonDistributedStrategy):
            strategy_str = "non-dist"
        elif isinstance(strategy, TorchDDPStrategy):
            strategy_str = "ddp"
        elif isinstance(strategy, DeepSpeedStrategy):
            strategy_str = "deepspeed"
        elif isinstance(strategy, HorovodStrategy):
            strategy_str = "horovod"
        else:
            strategy_str = "unk"

        global_rank = strategy.global_rank()
        num_gpus_global = strategy.global_world_size()

        # Extracting and storing the profiling data
        key_averages = profiler.key_averages()
        profiling_dataframe = gather_profiling_data(key_averages=key_averages)
        profiling_dataframe["strategy"] = strategy_str
        profiling_dataframe["num_gpus"] = num_gpus_global
        profiling_dataframe["global_rank"] = global_rank

        profiling_log_dir = Path("profiling_logs")
        profiling_log_dir.mkdir(parents=True, exist_ok=True)

        filename = f"profile_{strategy_str}_{num_gpus_global}_{global_rank}.csv"
        output_path = profiling_log_dir / filename

        print(f"Writing profiling dataframe to {output_path}")
        profiling_dataframe.to_csv(output_path)
        strategy.clean_up()

        return result

    return profiled_method
