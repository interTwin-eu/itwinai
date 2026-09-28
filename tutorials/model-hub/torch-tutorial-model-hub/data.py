# --------------------------------------------------------------------------------------
# Part of the RI-SCALE Project: https://www.riscale.eu/
# --------------------------------------------------------------------------------------
"""Pipeline steps that produce the synthetic datasets used by this tutorial."""

from typing import Tuple

from synthetic_data import SyntheticCheckpointDataset, SyntheticInferenceDataset
from torch.utils.data import Dataset


class SyntheticCheckpointDatasetSplitter:
    """Produces the train/validation split consumed by `training_pipeline`."""

    def __init__(self, train_samples: int = 64, val_samples: int = 16):
        self.train_samples = train_samples
        self.val_samples = val_samples

    def execute(self) -> Tuple[Dataset, Dataset]:
        train_dataset = SyntheticCheckpointDataset(num_samples=self.train_samples)
        validation_dataset = SyntheticCheckpointDataset(num_samples=self.val_samples)
        return train_dataset, validation_dataset


class SyntheticInferenceDatasetGenerator:
    """Produces the inference dataset consumed by `inference_pipeline`."""

    def __init__(self, inference_samples: int = 16):
        self.inference_samples = inference_samples

    def execute(self) -> Dataset:
        return SyntheticInferenceDataset(num_samples=self.inference_samples)
