# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Oleksandr Krochak
# --------------------------------------------------------------------------------------

from itwinai.components import DataGetter, DataProcessor, DataSplitter, monitor_exec
import os
from pulsar_simulation.generate_data_pipeline import generate_example_payloads_for_training
from typing import Optional, Tuple
import torch
import glob

from torch.utils.data import Dataset, TensorDataset, random_split
from src.pulsar_analysis.train_neural_network_model import ImageMaskPair
from src.pulsar_analysis.preprocessing import PrepareFreqTimeImage
from src.pulsar_analysis.pipeline_methods import PipelineImageToMask

class synthesizeData(DataGetter):
    def __init__(self, name: Optional[str] = None,
                 tag: str = "test_v0_", num_payloads: int = 50, plot: bool = 0, num_cpus: int = 4, 
                 param_root: str = "syn_runtime/", payload_root: str = "syn_payload/") -> None:
       
        """Initialize the synthesizeData class.
    
        Args:
            name [optional] (str):  name of the data getter component.
            param_root      (str):  folder where synthetic param data will be saved.
            payload_root    (str):  folder where synthetic payload data will be saved.
            tag             (str):  tag which is used as prefix for the generated files.
            num_cpus        (int):  number of CPUs used for parallel processing.
            num_payloads    (int):  number of generated examples.
            plot            (bool): if True, plotting routine is activated \
                                               (set False when running 'main.py' directly after)
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()), pop_self=False)

        # TODO find a smart way to compute the right value for num_cpus

        if not (os.path.exists(param_root) and os.path.exists(payload_root)):
            os.makedirs(param_root, exist_ok=True)
            os.makedirs(payload_root, exist_ok=True)

    @monitor_exec
    def execute(self) -> None:
        """Generate synthetic data and save it to disk. Relies on the pulsar_simulation package."""
        generate_example_payloads_for_training(tag         = self.parameters["tag"], 
                                            num_payloads   = self.parameters["num_payloads"],
                                            plot_a_example = self.parameters["plot"], 
                                            param_folder   = self.parameters["param_root"],
                                            payload_folder = self.parameters["payload_root"],
                                            num_cpus       = self.parameters["num_cpus"],
                                            reinit_ray     = False) 
                                            


class GenericDataset(Dataset):
    
    """Class to represent Pulsar datasets"""

    def __init__(
        self,
        image_tag: str,
        mask_tag: str,
        image_directory: str,
        mask_directory: str,
        mask_maker_engine: Optional[PipelineImageToMask] = None,
        image_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=False, do_resize=True
        ),
        mask_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=True, do_resize=True
        ),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self._image_tag = image_tag
        self._mask_tag = mask_tag
        self._image_directory = image_directory
        self._mask_directory = mask_directory
        self._image_engine = image_engine
        self._mask_engine = mask_engine
        self._device = device
        self._mask_maker_engine = mask_maker_engine

    def loadImagePair(self, index) -> ImageMaskPair:
        imgAddress = self._image_directory + self._image_tag.replace(
            "*", str(index)
        )
        maskAddress = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        if self._mask_maker_engine is None: # this method is taken from ImageToMaskDataset class
            image_mask_pair = ImageMaskPair.load_from_payload_address(
                image_payload_address=imgAddress,
                mask_payload_address=maskAddress,
                image_engine=self._image_engine,
                mask_engine=self._mask_engine,
            )
        else:     # this method is taken from InMaskToDataset class
            image_mask_pair = ImageMaskPair.load_from_payload_and_make_in_mask(
                image_payload_address=imgAddress,
                mask_payload_address=maskAddress,
                mask_maker_engine=self._mask_maker_engine,
                image_engine=self._image_engine,
                mask_engine=self._mask_engine,
            )
        return image_mask_pair
        
    def __getitem__(self, index):
        img, mask = self.loadImagePair(index)()
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return img.float(), mask.float()
    
    def __get_descriptions__(self, index):
        return self.loadImagePair(index).descriptions

    def __len__(self):
        search_pattern = os.path.join(self._image_directory, self._image_tag)
        matching_files = glob.glob(search_pattern)
        num_items = len(matching_files)
        return num_items

    def plot(self, index):
        """Plot InMask and Mask pair

        Args:
            index (int): index of the pair to plot
        """
        image_payload_address = self._image_directory + self._image_tag.replace(
            "*", str(index)
        )
        mask_payload_address = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        image_mask_pair = ImageMaskPair.load_from_payload_and_make_in_mask(
            image_payload_address=image_payload_address,
            mask_payload_address=mask_payload_address,
            mask_maker_engine=self._mask_maker_engine,
            image_engine=self._image_engine,
            mask_engine=self._mask_engine,
        )
        image_mask_pair.plot()

# testData = synthesizeData(num_payloads=10)
# testData.execute()

# trainData = synthesizeData(tag='train_v0_', num_payloads=10)
# trainData.execute()

class TimeSeriesDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the splitter for time-series datasets.

        Args:
            train_proportion (int | float): Proportion of files for the training set.
            validation_proportion (int | float): Proportion for validation.
            test_proportion (int | float): Proportion for testing.
            rnd_seed (int | None): Seed for randomization (optional).
            name (str | None): Name of the data splitter.
        """
        super().__init__(train_proportion, validation_proportion, test_proportion, name)
        self.save_parameters(**self.locals2params(locals()))
        self.rnd_seed = rnd_seed

    @monitor_exec
    def execute(self, whole_dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """Execute the dataset splitting process.

        Finds all pickled files in the root folder, then splits them into
        training, validation, and test sets based on the specified proportions.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Training, validation, and test datasets.
        """

        # Split file paths into train, validation, and test sets
        generator = torch.Generator().manual_seed(self.rnd_seed)
        [train_dataset, validation_dataset, test_dataset] = random_split(
            whole_dataset,
            [self.train_proportion, self.validation_proportion, self.test_proportion],
            generator=generator,
        )
        print(f"Shape of item: {train_dataset.__getitem__(idx=5)[0].shape}")
        return train_dataset, validation_dataset, test_dataset
