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
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, TensorDataset, random_split
from src.pulsar_analysis.train_neural_network_model import ImageMaskPair, SignalToLabelDataset
from src.pulsar_analysis.preprocessing import PrepareFreqTimeImage, BinarizeToMask
from src.pulsar_analysis.pipeline_methods import PipelineImageToMask, \
        PipelineImageToFilterDelGraphtoIsPulsar, PipelineImageToFilterToCCtoLabels
from src.pulsar_analysis.neural_network_models import UNet

class SynthesizeData(DataGetter):
    def __init__(self, 
                 name: Optional[str] = None,
                 tag: str = "test_v0_", 
                 num_payloads: int = 50, 
                 plot: bool = 0, 
                 num_cpus: int = 4, 
                 param_root: str = "./syn_runtime/", 
                 payload_root: str = "./syn_payload/") -> None:
       
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
        # Re-use some arguments to initialize PrepareFreqTimeImage class inside this class
        do_rot_phase_avg: bool = True,
        do_resize: bool = True,
        resize_size: Tuple = (128,128),
        mask_binarize_func: str = "thresh",
        mask_maker_engine: Optional[dict] = None,
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
        self._device = device


        # Initialize PrepareFreqTimeImage classes
        self._img_engine = PrepareFreqTimeImage(
                    do_rot_phase_avg=do_rot_phase_avg,
                    do_binarize=False,
                    do_resize=do_resize,
                    resize_size=resize_size,
                    )
        self._mask_engine = PrepareFreqTimeImage(
                    do_rot_phase_avg=True,
                    do_binarize=True,
                    do_resize=True,
                    resize_size=resize_size,
                    binarize_engine = BinarizeToMask(binarize_func=mask_binarize_func)
                    #BinarizeToMask(binarize_func='gaussian_blur') # or 'exponential'
                    )
        
        self._image_engine = image_engine
        self._mask_engine = mask_engine

        # Optional initialization of mask_maker_engine
        if mask_maker_engine is not None:
            if mask_maker_engine["model"] == "UNet":
                mme_model = UNet()
            else:
                raise ValueError("Uknown model type for mask_maker_engine")
            self._mask_maker_engine = PipelineImageToMask(
                image_to_mask_network=mme_model,
                trained_image_to_mask_network_path=mask_maker_engine["model_path"],
            )
        else:
            self._mask_maker_engine = None

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

    def execute(self) -> Dataset:
        return self

class SignalDataset(SignalToLabelDataset):
    # this class is defined to provide a new init method for 
    # easy initialization from config.yaml file.
    def __init__(
        self,
        mask_tag: str,
        mask_directory: str,
        do_rot_phase_avg: bool = True,
        do_binarize: bool = True,
        do_resize: bool = True,
        resize_size: Tuple = (128,128),
        mask_binarize_func: str = "thresh",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self._mask_tag = mask_tag
        self._mask_directory = mask_directory

        self._device = device
        self._mask_engine = PrepareFreqTimeImage(
                    do_rot_phase_avg=do_rot_phase_avg,
                    do_binarize=do_binarize,
                    do_resize=do_resize,
                    resize_size=resize_size,
                    binarize_engine = BinarizeToMask(binarize_func=mask_binarize_func)
                    )
        
    def execute(self) -> Dataset:
        return self

class DatasetSplitter(DataSplitter):
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

class pipelinePulsarInterface(PipelineImageToFilterDelGraphtoIsPulsar):
    def execute(self) -> PipelineImageToFilterDelGraphtoIsPulsar:
        return self
    
class pipelineLabelsInterface(PipelineImageToFilterToCCtoLabels):
    def execute(self) -> PipelineImageToFilterToCCtoLabels:
        return self

class testSuite:
    def __init__(
        self,
        image_to_mask_network: torch.nn.Module,
        trained_image_to_mask_network_path: str,
        mask_filter_network: torch.nn.Module,
        trained_mask_filter_network_path: str,
        signal_to_label_network: torch.nn.Module,
        trained_signal_to_label_network: str,
        img_dir: str,
        lbl_dir: str,
        size: int,
        offset: int,
        ):
            self.img_dir = img_dir
            self.lbl_dir = lbl_dir
            self.size = size
            self.offset = offset

            self.DelGraphtoIsPulsar = PipelineImageToFilterDelGraphtoIsPulsar(
                image_to_mask_network,
                trained_image_to_mask_network_path,
                mask_filter_network,
                trained_mask_filter_network_path,
                signal_to_label_network,
                trained_signal_to_label_network   
            )

            self.ToCCtoLabels = PipelineImageToFilterToCCtoLabels(
                image_to_mask_network,
                trained_image_to_mask_network_path,
                mask_filter_network,
                trained_mask_filter_network_path,
                min_cc_size_threshold=5
            )

    def execute(self):
        data = np.load(file=self.img_dir,mmap_mode='r')
        data_label = np.load(file=self.lbl_dir,mmap_mode='r')
        data_subset = data[self.offset+1:self.offset+self.size,:,:]
        data_label_subset = data_label[self.offset+1:self.offset+self.size]

        self.DelGraphtoIsPulsar.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_details=True,
            plot_randomly=True,
            batch_size=2
        )

        self.ToCCtoLabels.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_randomly=True,
            batch_size=2
        )

        # plt.show()
        for i in plt.get_fignums():
            fig = plt.figure(i)
            fig.savefig(f"plots/figure_{i}.png")
            
        return print("Test Suite executed")       

class ModelSaver:
    def execute(self, model, path) -> None:
        torch.save(model.state_dict(), path)
        print(f"Model saved at {path}")
        return None