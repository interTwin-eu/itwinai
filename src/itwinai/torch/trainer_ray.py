import os
from abc import abstractmethod
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Literal, Optional,
                    Tuple, Union)
import tempfile

import ray.train as train
import ray.train.torch
import torch

import ray.tune as tune
from torch.utils.data import Dataset, Sampler
from pydantic.utils import deep_update

from itwinai.components import Trainer, monitor_exec
from itwinai.torch.distributed import RayDDPStrategy, RayDeepSpeedStrategy
from itwinai.torch.raytune import get_raytune_schedule, get_raytune_search_alg
from itwinai.loggers import Logger


DEFAULT_CONFIG = {
    "scaling_config": {
        "num_workers": 4,  # Default to 4 workers
        "use_gpu": True,
        "resources_per_worker": {
            "CPU": 5,
            "GPU": 1
        }
    },
    "tune_config": {
        "num_samples": 1,     # Number of trials to run, increase for more thorough tuning
        "metric": "loss",
        "mode": "min"
    },
    "run_config": {
        "checkpoint_at_end": True,   # Save checkpoint at the end of each trial
        "checkpoint_freq": 10,       # Save checkpoint every 10 iterations
        "storage_path": "ray_results"  # Directory to save results, logs, and checkpoints
    },
    "train_loop_config": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 10,
        "shuffle_train": False,
        "shuffle_validation": False,
        "shuffle_test": False,
        "pin_gpu_memory": False,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "optim_momentum": 0.9,
        "optim_weight_decay": 0,
        "num_workers_dataloader": 4,
        "random_seed": 21
    }
}


class RayTorchTrainer(Trainer):
    def __init__(
            self,
            config: Dict,
            strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = 'ddp',
            name: Optional[str] = None,
            logger: Optional[Logger] = None
    ) -> None:
        super().__init__(name=name)
        self.logger = logger
        self.strategy = self._initialize_strategy(strategy)

        self._set_configs(config=config)

        self._initialize_ray()

    def _set_configs(self, config: Dict):

        self.config = deep_update(DEFAULT_CONFIG, config)

        self._set_scaling_config()
        self._set_tune_config()
        self._set_run_config()
        self._set_train_loop_config()

    def _initialize_ray(self):
        try:
            ip_head = os.environ.get("ip_head")
            head_node_ip = os.environ.get("head_node_ip")

            if not ip_head or not head_node_ip:
                raise EnvironmentError(
                    "Ray initialization requires 'ip_head' and 'head_node_ip' to be set.")

            if not ray.is_initialized():
                ray.init(
                    address=ip_head,
                    _node_ip_address=head_node_ip
                )

        except Exception as e:
            raise RuntimeError(f"Error initializing Ray: {str(e)}")

    def _initialize_strategy(self, strategy: str):
        if strategy == 'ddp':
            return RayDDPStrategy()
        elif strategy == "deepspeed":
            return RayDeepSpeedStrategy()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        batch_size: int = 1,
        shuffle_train: bool | None = False,
        shuffle_test: bool | None = False,
        shuffle_validation: bool | None = False,
        sampler: Union[Sampler, Iterable, None] = None,
        collate_fn: Callable[[List], Any] | None = None
    ) -> None:

        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn,
            sampler=sampler
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=batch_size,
                shuffle=shuffle_validation,
                collate_fn=collate_fn
            )
        else:
            self.validation_dataloader = None
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=shuffle_test,
                collate_fn=collate_fn
            )
        else:
            self.test_dataloader = None

    @abstractmethod
    def train(config, data=None):
        pass

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """
        """

        train_with_data = tune.with_parameters(
            self.train,
            data=[train_dataset, validation_dataset, test_dataset]
        )
        trainer = ray.train.torch.TorchTrainer(
            train_with_data,
            scaling_config=self.scaling_config,
            run_config=self.run_config
        )
        param_space = {
            "train_loop_config": self.train_loop_config
        }
        tuner = tune.Tuner(
            trainer,
            param_space=param_space,
            tune_config=self.tune_config
        )

        result_grid = tuner.fit()

        return train_dataset, validation_dataset, test_dataset, result_grid

    def set_epoch(self, epoch: int) -> None:
        self.train_dataloader.sampler.set_epoch(epoch)
        if self.validation_dataloader is not None:
            self.validation_dataloader.sampler.set_epoch(epoch)
        if self.test_dataloader is not None:
            self.test_dataloader.sampler.set_epoch(epoch)

    def _set_tune_config(self):
        tune_config = self.config.get("tune_config")
        if tune_config:
            num_samples = tune_config.get("num_samples", 10)
            metric = tune_config.get("metric", "loss")
            mode = tune_config.get("mode", "min")

            search_alg = get_raytune_search_alg(
                tune_config) if "search_alg" in tune_config else None
            scheduler = get_raytune_schedule(
                tune_config) if "scheduler" in tune_config else None

            # Only set metric and mode if search_alg and scheduler aren't defined
            self.tune_config = tune.TuneConfig(
                num_samples=num_samples,
                metric=metric if not search_alg and not scheduler else None,
                mode=mode if not search_alg and not scheduler else None,
                search_alg=search_alg,
                scheduler=scheduler,
            )
        else:
            # TODO: Communicate what that means
            print("INFO: No Tune Config configured.")
            self.tune_config = None

    def _set_scaling_config(self):
        scaling_config = self.config.get("scaling_config")
        if scaling_config:
            self.scaling_config = ray.train.ScalingConfig(
                **scaling_config
            )
        else:
            # TODO: Communicate what that means
            print("INFO: No Scaling Config configured.")
            self.scaling_config = None

    def _set_run_config(self):
        # Set RunConfig if it exists, otherwise assume local execution
        run_config = self.config.get("run_config")
        if run_config:
            # TODO: Look up structure of Scaling Config
            storage_path = Path(run_config.get("storage_path"))
            if storage_path:
                self.run_config = ray.train.RunConfig(
                    storage_path=Path.absolute(storage_path)
                )
        else:
            print("INFO: No RunConfig provided. Assuming local execution.")
            self.run_config = None

    def _set_train_loop_config(self):
        train_loop_config = self.config.get("train_loop_config")
        if train_loop_config:
            self.train_loop_config = self._set_searchspace(train_loop_config)
        else:
            print("INFO: No training_loop_config detected. \
                  No parameters are being tuned or passed to the training function.")
            self.train_loop_config = {}

    def _set_searchspace(self, train_loop_dict: Dict):
        train_loop_config = {}

        for name, values in train_loop_dict.items():

            if not isinstance(values, dict):
                # Constant parameters can be added as-is
                train_loop_config[name] = values
                continue

            param_type = values.get("type")

            if param_type == "choice":
                train_loop_config[name] = tune.choice(values["options"])

            elif param_type == "uniform":
                train_loop_config[name] = tune.uniform(
                    float(values["min"]), float(values["max"]))

            elif param_type == "quniform":
                train_loop_config[name] = tune.quniform(
                    values["min"], values["max"], values["q"])

            elif param_type == "loguniform":
                train_loop_config[name] = tune.loguniform(values["min"], values["max"])

            elif param_type == "qloguniform":
                train_loop_config[name] = tune.qloguniform(
                    values["min"], values["max"], values["q"])

            elif param_type == "randint":
                train_loop_config[name] = tune.randint(values["min"], values["max"])

            elif param_type == "qrandint":
                train_loop_config[name] = tune.qrandint(
                    values["min"], values["max"], values["q"])

            elif param_type == "lograndint":
                train_loop_config[name] = tune.lograndint(values["min"], values["max"])

            elif param_type == "qlograndint":
                train_loop_config[name] = tune.qlograndint(
                    values["min"], values["max"], values["q"])

            elif param_type == "randn":
                train_loop_config[name] = tune.randn(values["mean"], values["stddev"])

            elif param_type == "qrandn":
                train_loop_config[name] = tune.qrandn(
                    values["mean"], values["stddev"], values["q"])

            elif param_type == "grid_search":
                train_loop_config[name] = tune.grid_search(values["options"])

            else:
                raise ValueError(f"Unsupported search space type: {param_type}")

        return train_loop_config

    # TODO: Maybe I could make this more general
    def checkpoint_and_report(self, epoch, tuning_metrics, checkpointing_data=None):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % self.config.get("checkpoint_freq", 1)

            if checkpointing_data and should_checkpoint:
                torch.save(
                    checkpointing_data,
                    os.path.join(temp_checkpoint_dir, str(epoch))
                )
                checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)

        train.report(tuning_metrics, checkpoint=checkpoint)

    def initialize_logger(self, hyperparams: Optional[Dict], rank):
        print(f"Logger initializing with rank {rank}")

        if self.logger:
            self.logger.create_logger_context(rank=rank)

            if hyperparams:
                self.logger.save_hyperparameters(hyperparams)
            else:
                print("INFO: Not logging any hyperparameters.")

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """
        if self.logger:
            self.logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs
            )
        else:
            print("INFO: The log method was called, but no logger was configured for this Trainer.")
