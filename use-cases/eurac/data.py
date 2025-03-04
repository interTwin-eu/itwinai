from copy import deepcopy
from typing import Dict, List, Tuple

from hython.datasets.wflow_sbm import WflowSBM
from hython.datasets import get_dataset
from hython.scaler import Scaler
from hython.config import Config

from itwinai.components import DataSplitter, monitor_exec


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        # == common ==
        hython_trainer: str,
        dataset: str,
        downsampling_temporal_dynamic: bool,
        scaling_variant: str,
        experiment_name: str,
        experiment_run: str,
        data_source: dict,
        work_dir: str,
        data_lazy_load: bool,
        scaling_use_cached: bool | None = None,
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
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(self) -> Tuple[WflowSBM, WflowSBM, None]:

        cfg = Config()

        for i in self.parameters:
            setattr(cfg, i, self.parameters[i])

        scaler = Scaler(cfg, cfg.scaling_use_cached)

        period = "train"
        istrain = True
        if "cal" in cfg.hython_trainer:
            period = "cal"

        train_dataset = get_dataset(cfg.dataset)(cfg, scaler, istrain, period)

        val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid")

        return train_dataset, val_dataset, None
