# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Alex Krochak
#
# Credit:
# - Alex Krochak <o.krochak@fz-juelich.de> - FZJ
# --------------------------------------------------------------------------------------

import os
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt

from itwinai.components import DataGetter, DataSplitter, monitor_exec
from torch.utils.data import Dataset, random_split

from typing import Optional, Tuple, Literal
from collections import OrderedDict
from pulsar_analysis.train_neural_network_model import ImageMaskPair, SignalLabelPair
from pulsar_analysis.preprocessing import PrepareFreqTimeImage, BinarizeToMask
from pulsar_analysis.pipeline_methods import (
    PipelineImageToMask,
    PipelineImageToFilterDelGraphtoIsPulsar,
    PipelineImageToFilterToCCtoLabels,
)
from pulsar_analysis.neural_network_models import UNet
from pulsar_simulation.generate_data_pipeline import generate_example_payloads_for_training
import ray

class SynthesizeData(DataGetter):
    def __init__(
        self,
        name: Optional[str] = None,
        tag: str = "test_v0_",
        num_payloads: int = 50,
        plot: bool = False,
        num_cpus: int = 4,
        param_root: str = "./syn_runtime/",
        payload_root: str = "./syn_payload/",
    ) -> None:
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
            returns None.
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()), pop_self=False)
        # TODO find a smart way to compute the right value for num_cpus
        os.makedirs(param_root, exist_ok=True)
        os.makedirs(payload_root, exist_ok=True)

    @monitor_exec
    def execute(self) -> None:
        """Generate synthetic data and save it to disk.
        Relies on the pulsar_simulation package."""

        # Ray is assumed to be running for the data generation. 
        if ray.is_initialized() == False: ray.init(num_cpus=self.parameters["num_cpus"],
                                                   include_dashboard=False)
        generate_example_payloads_for_training(
            tag=self.parameters["tag"],
            num_payloads=self.parameters["num_payloads"],
            plot_a_example=self.parameters["plot"],
            param_folder=self.parameters["param_root"],
            payload_folder=self.parameters["payload_root"],
            num_cpus=self.parameters["num_cpus"],
            reinit_ray=False,
        )
        if ray.is_initialized(): ray.shutdown()  # shutdown ray after the data generation is done


class PulsarDataset(Dataset):
    """Class to represent common datasets. Variable 'engine_settings' is supposed to
    provide the settings for the image, mask and mask_maker engines, depending on
    the type of dataset. For example, for UNet dataset, the settings could be:

    engine_settings = {
        "image": {
            "do_rot_phase_avg": True,
            "do_binarize": False,
            "do_resize": True,
            "resize_size": (128, 128),
        },
        "mask": {
            "do_rot_phase_avg": True,
            "do_binarize": True,
            "do_resize": True,
            "resize_size": (128, 128),
        },
    }

    These settings are passed to the PrepareFreqTimeImage classes. The number and configuration
    of these are dependent on network type, thus initialization is inherently dependent on
    the provided "type" argument ! To clarify:
    - UNet: image and mask engines are initialized
        see: ImageToMaskDataset in src.pulsar_analysis.train_neural_network_model.py:150
    - FilterCNN: image, mask and mask_maker engines are initialized
        see: InMaskToMaskDataset in src.pulsar_analysis.train_neural_network_model.py:238
    - CNN1D: only mask engine is initialized.
        see: SignalToLabelDataset in src.pulsar_analysis.train_neural_network_model.py:496
    """

    def __init__(
        self,
        type: Literal["unet", "filtercnn", "cnn1d"],
        image_tag: str,
        mask_tag: str,
        image_directory: str,
        mask_directory: str,
        engine_settings: dict,
    ):
        self._type = type
        self._image_tag = image_tag
        self._mask_tag = mask_tag
        self._image_directory = image_directory
        self._mask_directory = mask_directory
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._image_engine = None
        self._mask_engine = None
        self._mask_maker_engine = None

        ### Compute dataset length ###
        search_pattern = os.path.join(self._image_directory, self._image_tag)
        matching_files = glob.glob(search_pattern)
        self._len = len(matching_files)

        ### Check that the engine settings are appropriate for chosen dataset type ###
        if self._type == "unet" and engine_settings.keys() != {"image","mask"}:
            raise ValueError("Wrong engine settings for UNet dataset. \n"
                "Provide 'image' and 'mask' engine settings.")

        elif self._type == "filtercnn" and \
        engine_settings.keys() != {"image","mask","mask_maker"}:
            raise ValueError("Wrong engine settings for FilterCNN dataset. \n"
            "Provide 'image', 'mask' and 'mask_maker' engine settings.")
        
        elif self._type == "cnn1d" and engine_settings.keys() != {"mask"}:
            raise ValueError("Wrong engine settings for CNN1D dataset.\n"
                             "Provide 'mask' engine settings.")

        ### Initialize the engines, one to three dependent on the dataset type ###
        ###        Mask engine settings are needed for all dataset types       ###

        # retrieve the binarization function from the engine settings and remove it
        bin_func = engine_settings["mask"].pop("binarize_func")
        bin_eng = BinarizeToMask(binarize_func=bin_func)

        self._mask_engine = PrepareFreqTimeImage(
            **engine_settings["mask"], binarize_engine=bin_eng
        )

        if self._type == "unet" or self._type == "filtercnn":  # init image engine
            self._image_engine = PrepareFreqTimeImage(**engine_settings["image"])

        if self._type == "filtercnn":  # init mask_maker engine
            # similarly, to bin. func., retrieve model type
            model_name = engine_settings["mask_maker"].pop("model")
            if model_name == "UNet":
                mme_model = UNet()
            else:
                raise ValueError(
                    "Unknown model type in engine_settings['mask_maker']['model']"
                )
            self._mask_maker_engine = PipelineImageToMask(
                **engine_settings["mask_maker"], image_to_mask_network=mme_model
            )

    def load_image_pair(self, img_id: int) -> ImageMaskPair | Tuple[torch.tensor, torch.tensor]:
        """Load an a data point from disk. Loading method depends on the network type:
        - For UNet and FilterCNN architectures, data point consits of image and mask pair.
        - For CNN1D architecture, data point consists of a mask only.
        """
        img_address = self._image_directory + self._image_tag.replace("*", str(img_id))
        mask_address = self._mask_directory + self._mask_tag.replace("*", str(img_id))

        if self._type == "unet":
            pair = ImageMaskPair.load_from_payload_address(
                image_payload_address=img_address,
                mask_payload_address=mask_address,
                image_engine=self._image_engine,
                mask_engine=self._mask_engine,
            )

        elif self._type == "filtercnn":
            pair = ImageMaskPair.load_from_payload_and_make_in_mask(
                image_payload_address=img_address,
                mask_payload_address=mask_address,
                mask_maker_engine=self._mask_maker_engine,
                image_engine=self._image_engine,
                mask_engine=self._mask_engine,
            )

        elif self._type == "cnn1d":
            signal_label_pair = SignalLabelPair.load_from_payload_address(
                mask_payload_address=mask_address, mask_engine=self._mask_engine
            )

            signal, label = signal_label_pair()
            label_vector = list(label.values())
            pulsar_present = label_vector[0]

            if pulsar_present > 0:
                label_vector = [1]
            else:
                label_vector = [0]

            label_vector = torch.tensor(label_vector, dtype=torch.float32)
            pair = torch.tensor(signal, dtype=torch.float32).unsqueeze(0), label_vector

        return pair

    def __getitem__(self, item_id: int) -> ImageMaskPair | Tuple[torch.tensor, torch.tensor]:
        """Use load_image_pair method to retrieve the data point with the given item_id.
        Output is network architecture dependent.
        Returns either image-mask pair (UNET, FCNN) or signal-label pair (CNN1D)."""

        ### Implementation is slightly different per architecture
        ### due to the original use-case code
        if self._type == "cnn1d":
            return self.load_image_pair(item_id)
        elif self._type == "unet" or self._type == "filtercnn":
            img, mask = self.load_image_pair(item_id)()
            img = img.unsqueeze(0)
            mask = mask.unsqueeze(0)
            return img.float(), mask.float()

    def __get_descriptions__(self, item_id: int):
        """Provide descriptions of the data point with the given item_id."""
        return self.load_image_pair(item_id).descriptions

    def __len__(self):
        """Return the length of the dataset pre-computed during the initialization."""
        return self._len

    def plot(self, item_id: int):
        """Plot the data point with the provided item_id.
        Plotting method is model-dependent."""
        ##TODO: make this method HPC-friendy
        if self._type == "unet" or self._type == "filtercnn":
            image_payload_address = self._image_directory + self._image_tag.replace(
                "*", str(item_id)
            )
            mask_payload_address = self._mask_directory + self._mask_tag.replace(
                "*", str(item_id)
            )

            image_mask_pair = ImageMaskPair.load_from_payload_and_make_in_mask(
                image_payload_address=image_payload_address,
                mask_payload_address=mask_payload_address,
                mask_maker_engine=self._mask_maker_engine,
                image_engine=self._image_engine,
                mask_engine=self._mask_engine,
            )
            image_mask_pair.plot()

        elif self._type == "cnn1d":
            signal, label_vector = self[item_id]
            _, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
            ax.plot(signal.detach().numpy().flatten())
            ax.set_xlabel("freqs (a.u)")
            ax.set_ylabel("lags")
            ax.set_aspect("auto")
            ax.set_ylim(0, 1)
            ax.grid()
            ax.set_title(f"Pulsar Present {label_vector[0]>=0.9}")

    def execute(self) -> Dataset:
        """Read the dataset from disk and return it to the trainer in-memory."""
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
    def execute(self, whole_dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """Execute the dataset splitting process.

        Finds all pickled files in the root folder, then splits them into
        training, validation, and test sets based on the specified proportions.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Training, validation, and test datasets.
        """

        # Split file paths into train, validation, and test sets
        generator = torch.Generator().manual_seed(self.rnd_seed)
        train_dataset, validation_dataset, test_dataset = random_split(
            whole_dataset,
            [self.train_proportion, self.validation_proportion, self.test_proportion],
            generator=generator,
        )
        return train_dataset, validation_dataset, test_dataset


class PipelinePulsarInterface(PipelineImageToFilterDelGraphtoIsPulsar):
    def execute(self) -> PipelineImageToFilterDelGraphtoIsPulsar:
        return self


class PipelineLabelsInterface(PipelineImageToFilterToCCtoLabels):
    def execute(self) -> PipelineImageToFilterToCCtoLabels:
        return self


class TestSuite:
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

        self.del_graph_to_is_pulsar = PipelineImageToFilterDelGraphtoIsPulsar(
            image_to_mask_network,
            trained_image_to_mask_network_path,
            mask_filter_network,
            trained_mask_filter_network_path,
            signal_to_label_network,
            trained_signal_to_label_network,
        )

        self.to_cc_to_labels = PipelineImageToFilterToCCtoLabels(
            image_to_mask_network,
            trained_image_to_mask_network_path,
            mask_filter_network,
            trained_mask_filter_network_path,
            min_cc_size_threshold=5,
        )

    def execute(self):
        data = np.load(file=self.img_dir, mmap_mode="r")
        data_label = np.load(file=self.lbl_dir, mmap_mode="r")
        data_subset = data[self.offset + 1 : self.offset + self.size, :, :]
        data_label_subset = data_label[self.offset + 1 : self.offset + self.size]

        self.del_graph_to_is_pulsar.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_details=True,
            plot_randomly=True,
            batch_size=2,
        )

        self.to_cc_to_labels.test_on_real_data_from_npy_files(
            image_data_set=data_subset,
            image_label_set=data_label_subset,
            plot_randomly=True,
            batch_size=2,
        )

        # plt.show()
        os.makedirs("plots", exist_ok=True)
        for i in plt.get_fignums():
            fig = plt.figure(i)
            fig.savefig(f"plots/figure_{i}.png")

        print("Test Suite executed")


class ModelSaver:
    def execute(self, model, path) -> None:
        m_dict = model.state_dict()
        # ensure correct saving syntax expected by the loader
        m_new = OrderedDict(
            [
                (k.replace("module.", ""), v) if k.startswith("module") else (k, v)
                for k, v in m_dict.items()
            ]
        )
        torch.save(m_new, path)
        print(f"Model saved at {path}")
