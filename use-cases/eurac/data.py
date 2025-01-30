from copy import deepcopy
from typing import Dict, List, Tuple

from hydra.utils import instantiate
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import Wflow1d
from hython.scaler import Scaler
from omegaconf import OmegaConf

from itwinai.components import DataSplitter, monitor_exec


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        # == common ==
        dataset: str,
        scaling_variant: str,
        experiment_name: str,
        experiment_run: str,
        data_source: dict,
        work_dir: str,
        dynamic_inputs: List[str] | None = None,
        static_inputs: List[str] | None = None,
        target_variables: List[str] | None = None,
        scaling_static_range: Dict | None = None,
        mask_variables: List[str] | None = None,
        train_temporal_range: List[str] = None,
        valid_temporal_range: List[str] = None,
        train_downsampler: Dict | None = None,
        valid_downsampler: Dict | None = None,
        # == calibration ==
        data_dynamic_inputs: str | None = None,
        data_static_inputs: str | None = None,
        data_target_variables: str | None = None,
        data_target_mask: str | None = None,
        min_sample_target: int | None = None,
        seq_length: int | None = None,
        # == training ==
    ) -> None:

        if train_temporal_range is None:
            train_temporal_range = ["", ""]
        if valid_temporal_range is None:
            valid_temporal_range = ["", ""]

        self.save_parameters(**self.locals2params(locals()))

        self.cfg = deepcopy(self.locals2params(locals()))

        self.cfg = instantiate(OmegaConf.create(self.cfg))

    @monitor_exec
    def execute(self) -> Tuple[Wflow1d, Wflow1d, None]:
        scaler = Scaler(self.cfg)

        train_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, True, "train")

        val_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, False, "valid")

        return train_dataset, val_dataset, None
