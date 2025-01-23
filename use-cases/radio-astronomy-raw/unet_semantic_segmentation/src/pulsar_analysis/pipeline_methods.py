import os
import glob
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from .information_packet_formats import Payload
from .preprocessing import PrepareFreqTimeImage, BinarizeToMask
from .postprocessing import DelayGraph, ConnectedComponents, FitSegmentedTraces


class ImageReader:
    """Class ImageReader acts as an engine to load/prepare images from payload files or numpy arrays"""

    def __init__(
        self,
        file_type: type = Payload([]),
        resize_size: tuple = (128, 128),
        do_average: bool = False,
        do_binarize: bool = False,
    ):
        self.__filetype = file_type
        self.__resize_size = resize_size
        self.__do_average = do_average
        self.__do_binarize = do_binarize

    def __call__(self, filename: str):
        if type(self.__filetype) == Payload:
            image = self.read_from_payload(
                filename=filename,
                resize_size=self.__resize_size,
                do_average=self.__do_average,
                do_binarize=self.__do_binarize,
            )
        else:
            print(f"type not recognized {type(self.__filetype)}")
            image = None
        return image

    @staticmethod
    def read_from_payload(
        filename: str,
        resize_size: tuple = (128, 128),
        do_average: bool = False,
        do_binarize: bool = False,
    ):
        """Method to load freq-time image from payload files

        Args:
            filename (str): full address to payload file
            resize_size (tuple, optional): output shape of the loaded image. Defaults to (128, 128).
            do_average (bool, optional): If True then the image is created by averaging phase values over many rotations. Defaults to False.
            do_binarize (bool, optional): If True then the image is binarized. Defaults to False.

        Returns:
            (np.ndarray): loaded image
        """
        # image_payload = Payload.read_payload_from_jsonfile(filename=filename)
        image_reader_engine = PrepareFreqTimeImage(
            do_rot_phase_avg=do_average,
            do_binarize=do_binarize,
            do_resize=True,
            resize_size=resize_size,
        )
        image = image_reader_engine(payload_address=filename)
        image = image - min(image.flatten())
        if np.max(image.flatten()) > 0:
            image = image / np.max(image.flatten())
        return image


class LabelReader:
    """This class acts as an engine to read the labels from files of type payload"""

    def __init__(self, file_type: type = Payload([])):
        self.__filetype = file_type

    def __call__(self, filename: str):
        if type(self.__filetype) == Payload:
            description = self.read_from_payload(filename=filename)
        else:
            print(f"type not recognized {type(self.__filetype)}")
            description = None
        return description

    @staticmethod
    def read_from_payload(filename: str):
        """Method to read the label from pyload file

        Args:
            filename (str): full path to the payload file

        Returns:
            dict: dictionary containing details of the payload file
        """
        image_payload = Payload.read_payload_from_jsonfile(filename=filename)
        # image_reader_engine=PrepareFreqTimeImage(do_rot_phase_avg=True,do_binarize=False,do_resize=True)
        # image = image_reader_engine(payload_address=filename)
        description = image_payload.description
        return description


class ImageDataSet:
    """This class is used to represent or memory map a set of images"""

    def __init__(
        self,
        image_tag: str,
        image_directory: str,
        image_reader_engine: ImageReader = ImageReader(file_type=Payload([])),
    ):
        self._image_tag = image_tag
        self._image_directory = image_directory
        self._image_reader_engine = image_reader_engine

    def __getitem__(self, idx):
        image_address = self._image_directory + self._image_tag.replace("*", str(idx))
        image = self._image_reader_engine(filename=image_address)
        return image

    def __len__(self):
        search_pattern = os.path.join(self._image_directory, self._image_tag)
        matching_files = glob.glob(search_pattern)
        num_items = len(matching_files)
        return num_items

    def plot(self, idx):
        """plots the image from the set represented by the idx

        Args:
            idx (int): index of the image

        Returns:
            plt.axis: axis of the plot
        """
        image = self[idx]
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(image)
        ax.set_xlabel("Phase (a.u)")
        ax.set_ylabel("freq channel")
        ax.set_aspect("auto")
        return plt.gca()


class LabelDataSet:
    """This class is used to represent or memory map a set of labels of the images"""

    def __init__(
        self,
        image_tag: str,
        image_directory: str,
        label_reader_engine: LabelReader = LabelReader(file_type=Payload([])),
    ):
        self._image_tag = image_tag
        self._image_directory = image_directory
        self._label_reader_engine = label_reader_engine

    def __getitem__(self, idx):
        image_address = self._image_directory + self._image_tag.replace("*", str(idx))
        description = self._label_reader_engine(filename=image_address)
        return description

    def __len__(self):
        search_pattern = os.path.join(self._image_directory, self._image_tag)
        matching_files = glob.glob(search_pattern)
        num_items = len(matching_files)
        return num_items

    def plot(self, idx):
        """prints the label of idx image

        Args:
            idx (int): index representing the image

        Returns:
            dict: description
        """
        description = self[id]
        print(f"label extracted is {description}")
        return description


class PipelineImageToMask:
    """Class implementing methods in sequence to generate segmented freq-time Image from freq-time Image"""

    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
    ):
        self.__image_to_mask_network = image_to_mask_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__image_to_mask_network = self.__image_to_mask_network.to(self.device)
        self.__image_to_mask_network.load_state_dict(
            torch.load(trained_image_to_mask_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__image_to_mask_network.eval()

    def __call__(self, image: np.ndarray):
        mask = self.image_to_mask_method(image=image)
        return mask

    def image_to_mask_method(self, image: np.ndarray):
        """Method to convert image to mask

        Args:
            image (np.ndarray): image

        Returns:
            image (np.ndarray): mask
        """
        image = (
            torch.tensor(image, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.no_grad():
            pred = self.__image_to_mask_network(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_binarized = binarizer(image=pred_numpy_copy)
        return pred_binarized
    
    def plot(self, image: np.ndarray):
        """plots the image from the set represented by the idx

        Args:
            image (np.ndarray): image

        Returns:
            plt.axis: axis of the plot
        """
        image = image
        mask = self(image=image)
        _, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        ax.imshow(mask)
        ax.set_xlabel("Phase (a.u)")
        ax.set_ylabel("freq channel")
        ax.set_aspect("auto")
        return plt.gca()


class PipelineImageToDelGraphtoIsPulsar:
    """Class implementing methods in sequence to generate segmented freq-time Image then Delay graph then to determine if pulsar is there"""

    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
        signal_to_label_network: nn.Module,
        trained_signal_to_label_network: str,
    ):
        self.__image_to_mask_network = image_to_mask_network
        self.__signal_to_label_network = signal_to_label_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__image_to_mask_network = self.__image_to_mask_network.to(self.device)
        self.__image_to_mask_network.load_state_dict(
            torch.load(trained_image_to_mask_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__image_to_mask_network.eval()

        self.__signal_to_label_network = self.__signal_to_label_network.to(self.device)
        self.__signal_to_label_network.load_state_dict(
            torch.load(trained_signal_to_label_network,map_location=torch.device(self.device),weights_only=True)
        )
        self.__signal_to_label_network.eval()

    def __call__(
        self, image: np.ndarray, return_bool: bool = True, return_steps: bool = False
    ):
        mask = self.image_to_mask_method(image=image)
        signal = self.mask_to_signal_method(mask=mask)
        label = self.signal_to_label_method(signal=signal)

        if return_bool and not return_steps:
            if label > 0.5:
                return True
            else:
                return False
        elif return_bool and return_steps:
            if label > 0.5:
                bool_val = True
            else:
                bool_val = False
            return bool_val, mask, signal, label
        elif not return_bool and return_steps:
            return mask, signal, label
        else:
            return label

    def image_to_mask_method(self, image: np.ndarray):
        """Method to convert image to mask

        Args:
            image (np.ndarray): image

        Returns:
            image (np.ndarray): mask
        """
        image = (
            torch.tensor(image, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.no_grad():
            pred = self.__image_to_mask_network(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_binarized = binarizer(image=pred_numpy_copy)
        return pred_binarized

    def mask_to_signal_method(self, mask):
        """Method to convert mask to delaygraph and extract lags as signal

        Args:
            image (np.ndarray): mask

        Returns:
            np.ndarray: x_lags as signal
        """
        delay_graph_engine = DelayGraph()
        x_lags, __ = delay_graph_engine(dispersed_freq_time=mask)
        signal = x_lags.flatten()
        return signal

    def signal_to_label_method(self, signal: np.ndarray):
        """Method to determine if pulsar is there based on signal

        Args:
            signal (np.ndarray): x_lags as signal

        Returns:
            float: probability if pulsar is there
        """
        signal = torch.Tensor(signal).unsqueeze(0).unsqueeze(0)
        # self.__signal_to_label_network = self.__signal_to_label_network.to(self.device)
        # self.__signal_to_label_network.load_state_dict(torch.load(self.trained_model_path))
        with torch.no_grad():
            pred = self.__signal_to_label_network(signal.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        return pred_numpy

    def validate_efficiency(
        self,
        image_data_set: ImageDataSet,
        label_data_set: LabelDataSet,
    ):
        """Method to validate efficiency of the pipeline from image and label dataset

        Args:
            image_data_set (ImageDataSet): image dataset
            label_data_set (LabelDataSet): label dataset

        Returns:
            float: efficiency measure
        """
        total_items = len(image_data_set)
        correct_pred_noter = np.zeros(total_items)
        efficiency_measure = 0
        only_pulsar_signal_noter = 0
        only_pulsar_signal_predicted = 0
        pbar = tqdm(total=total_items, desc="Starting...")
        for idx in range(total_items):
            image = image_data_set[idx]
            is_pulsar_predicted = self(image=image, return_bool=True)
            is_pulsar_there = label_data_set[idx]["Pulsar"] == 1
            if is_pulsar_there:
                only_pulsar_signal_noter += 1
            if is_pulsar_there and is_pulsar_predicted:
                only_pulsar_signal_predicted += 1
            if is_pulsar_there == is_pulsar_predicted:
                correct_pred_noter[idx] = 1
            efficiency_measure = sum(correct_pred_noter) / (idx + 1)
            if only_pulsar_signal_noter > 0:
                efficiency_measure_only_pulsar = (
                    only_pulsar_signal_predicted / only_pulsar_signal_noter
                )
            else:
                efficiency_measure_only_pulsar = 0
            pbar.set_description(
                f"efficiency_measure {efficiency_measure:0.2f},{efficiency_measure_only_pulsar:0.2f}"
            )
            pbar.update(1)
        return efficiency_measure
    
    def display_results_in_batch(
        self,
        image_data_set: ImageDataSet,
        mask_data_set: ImageDataSet,
        label_data_set: LabelDataSet,
        randomize: bool = True,
        ids_toshow: list = [0, 1],
        batch_size: int = 2,
    ):
        """Plot results of the pipeline with step outputs and comparison with pre-labelled dataset

        Args:
            image_data_set (ImageDataSet): Image dataset
            mask_data_set (ImageDataSet): Mask dataset of the image dataset
            label_data_set (LabelDataSet): Label dataset of the images
            randomize (bool, optional): If True, randomly chooses images from the dataset. Defaults to True.
            ids_toshow (list, optional): If radomize = False, then choose ids_show from dataset. Defaults to [0, 1].
            batch_size (int, optional): If randomize=True, then chooses batch_size images from set. Defaults to 2.
        """
        total_items = len(image_data_set)
        if randomize:
            ids_toshow = np.random.permutation(np.arange(total_items))[0:batch_size]
        else:
            ids_toshow = ids_toshow
            batch_size = len(ids_toshow)
        _, ax = plt.subplots(batch_size, 4, figsize=(3.5 * 4, 3.5 * batch_size))
        for i, idx in enumerate(ids_toshow):
            image_current = image_data_set[idx]
            mask_current = mask_data_set[idx]
            given_image_label_current = label_data_set[idx]
            bool_val, mask,  signal, label = self.__call__(
                image=image_current, return_steps=True
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")

            ax[i, 1].imshow(mask_current)
            ax[i, 1].set_title(f"Original Mask")

            ax[i, 2].imshow(mask)
            ax[i, 2].set_title(f"Segmented Mask")

            ax[i, 3].plot(signal.flatten())
            ax[i, 3].set_ylim(0, 1)
            ax[i, 3].grid()
            ax[i, 3].set_title(f"Pulse Prob: {label:0.2f}")

    def test_on_real_data_from_npy_files(
        self,
        image_data_set: np.memmap,
        image_label_set: np.memmap | None = None,
        plot_details: bool = False,
        plot_randomly: bool = True,
        batch_size: int = 5,
    ):
        """Method to test pipeline on .npy file dataset

        Args:
            image_data_set (np.memmap): image dataset as numpy array
            image_label_set (np.memmap | None, optional): label dataset as numpy array. Defaults to None.
            plot_details (bool, optional): if True then plot the results. Defaults to False.
            plot_randomly (bool, optional): If True then randomly choose images from dataset. Defaults to True.
            batch_size (int, optional): number of images to test. minimum is 2. Defaults to 5.


        """
        if plot_details == False:
            is_pulsar_predicted = np.array(
                [
                    self(image=image_data_set[id, :, :], return_bool=True)
                    for id in range(image_data_set.shape[0])
                ]
            )
            if type(image_label_set) == np.memmap:
                is_pulsar_there = np.array(
                    [
                        "Pulse" in image_label_set[id]
                        for id in range(image_data_set.shape[0])
                    ]
                )
                success_in_detecting_pulsar = np.logical_and(
                    is_pulsar_predicted, is_pulsar_there
                )
                return success_in_detecting_pulsar
            else:
                return is_pulsar_predicted
        else:
            if plot_randomly:
                random_idxs = np.random.permutation(np.arange(image_data_set.shape[0]))[
                    0:batch_size
                ]
            else:
                random_idxs = np.arange(image_data_set.shape[0])[0:batch_size]
            _, ax = plt.subplots(batch_size, 3, figsize=(4 * 3, 4 * batch_size))
            for i, idx in enumerate(random_idxs):
                image_current = image_data_set[idx, :, :]
                image_current = image_current.astype(dtype=np.float32)
                image_current = image_current / np.max(image_current.flatten())
                if type(image_label_set) == np.memmap:
                    given_image_label_current = image_label_set[idx]
                else:
                    given_image_label_current = "NA"

                bool_val, mask, signal, label = self.__call__(
                    image=image_current, return_bool=True, return_steps=True
                )
                ax[i, 0].imshow(image_current)
                ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")
                ax[i, 1].imshow(mask)
                ax[i, 1].set_title(f"Segmented Mask")
                ax[i, 2].plot(signal.flatten())
                ax[i, 2].set_ylim(0, 1)
                ax[i, 2].grid()
                ax[i, 2].set_title(f"Pulse Prob: {label:0.2f}")
                # print(f'is_pulsar: {label:0.2f} and given_label is {given_image_label_current}')


class PipelineImageToFilterDelGraphtoIsPulsar:
    """Class implementing methods in sequence to generate segmented freq-time Image, filter it, then Delay graph then to determine if pulsar is there"""

    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
        mask_filter_network: nn.Module,
        trained_mask_filter_network_path: str,
        signal_to_label_network: nn.Module,
        trained_signal_to_label_network: str,
    ):
        self.__image_to_mask_network = image_to_mask_network
        self.__mask_filter_network = mask_filter_network
        self.__signal_to_label_network = signal_to_label_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__image_to_mask_network = self.__image_to_mask_network.to(self.device)
        self.__image_to_mask_network.load_state_dict(
            torch.load(trained_image_to_mask_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__image_to_mask_network.eval()

        self.__mask_filter_network = self.__mask_filter_network.to(self.device)
        self.__mask_filter_network.load_state_dict(
            torch.load(trained_mask_filter_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__mask_filter_network.eval()

        self.__signal_to_label_network = self.__signal_to_label_network.to(self.device)
        self.__signal_to_label_network.load_state_dict(
            torch.load(trained_signal_to_label_network,map_location=torch.device(self.device),weights_only=True)
        )
        self.__signal_to_label_network.eval()

    def __call__(
        self, image: np.ndarray, return_bool: bool = True, return_steps: bool = False
    ):
        mask = self.image_to_mask_method(image=image)
        filtered_mask = self.filter_mask_method(pred_binarized=mask)
        signal = self.mask_to_signal_method(mask=filtered_mask)
        label = self.signal_to_label_method(signal=signal)

        if return_bool and not return_steps:
            if label > 0.5:
                return True
            else:
                return False
        elif return_bool and return_steps:
            if label > 0.5:
                bool_val = True
            else:
                bool_val = False
            return bool_val, mask, filtered_mask, signal, label
        elif not return_bool and return_steps:
            return mask, filtered_mask, signal, label
        else:
            return label

    def image_to_mask_method(self, image: np.ndarray):
        """Method to convert image to mask

        Args:
            image (np.ndarray): image

        Returns:
            image (np.ndarray): mask
        """
        image = (
            torch.tensor(image, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.no_grad():
            pred = self.__image_to_mask_network(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_binarized = binarizer(image=pred_numpy_copy)
        return pred_binarized

    def filter_mask_method(self, pred_binarized: np.ndarray):
        """Method to filter out wrong segments in the segmented mask

        Args:
            pred_binarized (np.ndarray): segmented mask to filter

        Returns:
            np.ndarray: filtered segmented mask
        """
        if type(pred_binarized) == torch.Tensor:
            pred_binarized = pred_binarized.float().unsqueeze(0)
        else:
            pred_binarized = (
                torch.tensor(pred_binarized, requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )

        with torch.no_grad():
            pred_filtered_raw = self.__mask_filter_network(
                pred_binarized.to(self.device)
            )
        pred_filtered_raw = pred_filtered_raw.to("cpu")
        pred_filtered_raw_numpy = pred_filtered_raw.squeeze().numpy()
        pred_filtered_raw_numpy_copy = deepcopy(pred_filtered_raw_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_filtered_raw_binarized = binarizer(image=pred_filtered_raw_numpy_copy)
        return pred_filtered_raw_binarized

    def mask_to_signal_method(self, mask):
        """Method to convert mask to delaygraph and extract lags as signal

        Args:
            image (np.ndarray): mask

        Returns:
            np.ndarray: x_lags as signal
        """
        delay_graph_engine = DelayGraph()
        x_lags, __ = delay_graph_engine(dispersed_freq_time=mask)
        signal = x_lags.flatten()
        # signal = x_lags.flatten()

        return signal

    def signal_to_label_method(self, signal: np.ndarray):
        """Method to determine if pulsar is there based on signal

        Args:
            signal (np.ndarray): x_lags as signal

        Returns:
            float: probability if pulsar is there
        """
        signal = torch.Tensor(signal).unsqueeze(0).unsqueeze(0)
        # self.__signal_to_label_network = self.__signal_to_label_network.to(self.device)
        # self.__signal_to_label_network.load_state_dict(torch.load(self.trained_model_path))
        with torch.no_grad():
            pred = self.__signal_to_label_network(signal.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        return pred_numpy

    def validate_efficiency(
        self,
        image_data_set: ImageDataSet,
        label_data_set: LabelDataSet,
    ):
        """Method to validate efficiency of the pipeline from image and label dataset

        Args:
            image_data_set (ImageDataSet): image dataset
            label_data_set (LabelDataSet): label dataset

        Returns:
            float: efficiency measure
        """
        total_items = len(image_data_set)
        correct_pred_noter = np.zeros(total_items)
        efficiency_measure = 0
        only_pulsar_signal_noter = 0
        
        only_pulsar_signal_predicted_if_there = 0
        only_pulsar_signal_predicted_if_not_there = 0
        no_pulsar_signal_noter = 0
        pbar = tqdm(total=total_items, desc="Starting...")
        for idx in range(total_items):
            image = image_data_set[idx]
            is_pulsar_predicted = self(image=image, return_bool=True)
            is_pulsar_there = label_data_set[idx]["Pulsar"] == 1
            if is_pulsar_there:
                only_pulsar_signal_noter += 1
            else:
                no_pulsar_signal_noter += 1
            if is_pulsar_there and is_pulsar_predicted:
                only_pulsar_signal_predicted_if_there += 1

            if not(is_pulsar_there) and is_pulsar_predicted:
                only_pulsar_signal_predicted_if_not_there += 1
            
            if is_pulsar_there == is_pulsar_predicted:
                correct_pred_noter[idx] = 1
            efficiency_measure = sum(correct_pred_noter) / (idx + 1)

            if only_pulsar_signal_noter > 0:
                efficiency_measure_only_pulsar = (
                    only_pulsar_signal_predicted_if_there / only_pulsar_signal_noter
                )
            else:
                efficiency_measure_only_pulsar = 0

            if no_pulsar_signal_noter > 0:
                wrong_measure_but_no_pulsar = (
                    only_pulsar_signal_predicted_if_not_there /no_pulsar_signal_noter
                )
            else:
                wrong_measure_but_no_pulsar = 0

            pbar.set_description(
                f"efficiency_measure {efficiency_measure:0.2f},detected_if_pulsar {efficiency_measure_only_pulsar:0.2f}, wrongly_detected_as_no_pulsar {wrong_measure_but_no_pulsar:0.2f}"
            )
            pbar.update(1)
        return efficiency_measure

    def display_results_in_batch(
        self,
        image_data_set: ImageDataSet,
        mask_data_set: ImageDataSet,
        label_data_set: LabelDataSet,
        randomize: bool = True,
        ids_toshow: list = [0, 1],
        batch_size: int = 2,
    ):
        """Plot results of the pipeline with step outputs and comparison with pre-labelled dataset

        Args:
            image_data_set (ImageDataSet): Image dataset
            mask_data_set (ImageDataSet): Mask dataset of the image dataset
            label_data_set (LabelDataSet): Label dataset of the images
            randomize (bool, optional): If True, randomly chooses images from the dataset. Defaults to True.
            ids_toshow (list, optional): If radomize = False, then choose ids_show from dataset. Defaults to [0, 1].
            batch_size (int, optional): If randomize=True, then chooses batch_size images from set. Defaults to 2.
        """
        total_items = len(image_data_set)
        if randomize:
            ids_toshow = np.random.permutation(np.arange(total_items))[0:batch_size]
        else:
            ids_toshow = ids_toshow
            batch_size = len(ids_toshow)
        _, ax = plt.subplots(batch_size, 5, figsize=(3.5 * 5, 3.5 * batch_size))
        for i, idx in enumerate(ids_toshow):
            image_current = image_data_set[idx]
            mask_current = mask_data_set[idx]
            given_image_label_current = label_data_set[idx]
            bool_val, mask, filtered_mask, signal, label = self.__call__(
                image=image_current, return_steps=True
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")

            ax[i, 1].imshow(mask_current)
            ax[i, 1].set_title(f"Original Mask")

            ax[i, 2].imshow(mask)
            ax[i, 2].set_title(f"Segmented Mask")

            ax[i, 3].imshow(filtered_mask)
            ax[i, 3].set_title(f"Filtered Mask")

            ax[i, 4].plot(signal.flatten())
            ax[i, 4].set_ylim(0, 1)
            ax[i, 4].grid()
            ax[i, 4].set_title(f"Pulse Prob: {label:0.2f}")

    def test_on_real_data_from_npy_files(
        self,
        image_data_set: np.memmap,
        image_label_set: np.memmap | None = None,
        plot_details: bool = False,
        plot_randomly: bool = True,
        batch_size: int = 5,
    ):
        """Method to test pipeline on .npy file dataset

        Args:
            image_data_set (np.memmap): image dataset as numpy array
            image_label_set (np.memmap | None, optional): label dataset as numpy array. Defaults to None.
            plot_details (bool, optional): if True then plot the results. Defaults to False.
            plot_randomly (bool, optional): If True then randomly choose images from dataset. Defaults to True.
            batch_size (int, optional): number of images to test. minimum is 2. Defaults to 5.


        """
        if plot_details == False:
            is_pulsar_predicted = np.array(
                [
                    self(image=image_data_set[id, :, :], return_bool=True)
                    for id in range(image_data_set.shape[0])
                ]
            )
            if type(image_label_set) == np.memmap:
                is_pulsar_there = np.array(
                    [
                        "Pulse" in image_label_set[id]
                        for id in range(image_data_set.shape[0])
                    ]
                )
                success_in_detecting_pulsar = np.logical_and(
                    is_pulsar_predicted, is_pulsar_there
                )
                return success_in_detecting_pulsar
            else:
                return is_pulsar_predicted
        else:
            if plot_randomly:
                random_idxs = np.random.permutation(np.arange(image_data_set.shape[0]))[
                    0:batch_size
                ]
            else:
                random_idxs = np.arange(image_data_set.shape[0])[0:batch_size]
            _, ax = plt.subplots(batch_size, 4, figsize=(4 * 4, 4 * batch_size))
            for i, idx in enumerate(random_idxs):
                image_current = image_data_set[idx, :, :]
                image_current = image_current.astype(dtype=np.float32)
                image_current = image_current / np.max(image_current.flatten())
                if type(image_label_set) == np.memmap:
                    given_image_label_current = image_label_set[idx]
                else:
                    given_image_label_current = "NA"

                bool_val, mask, filtered_mask, signal, label = self.__call__(
                    image=image_current, return_bool=True, return_steps=True
                )
                ax[i, 0].imshow(image_current)
                ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")

                ax[i, 1].imshow(mask)
                ax[i, 1].set_title(f"Segmented Mask")

                ax[i, 2].imshow(filtered_mask)
                ax[i, 2].set_title(f"Filtered Mask")

                ax[i, 3].plot(signal.flatten())
                ax[i, 3].set_ylim(0, 1)
                ax[i, 3].grid()
                ax[i, 3].set_title(f"Pulse Prob: {label:0.2f}")
                # print(f'is_pulsar: {label:0.2f} and given_label is {given_image_label_current}')
            

class PipelineImageToCCtoLabels:
    """Class implementing methods in sequence to generate segmented freq-time Image, then CC then to determine CCs to categories"""

    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
        min_cc_size_threshold: int = 10,
    ):
        self.__image_to_mask_network = image_to_mask_network
        self.__min_cc_size_threshold = min_cc_size_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__image_to_mask_network = self.__image_to_mask_network.to(self.device)
        self.__image_to_mask_network.load_state_dict(
            torch.load(trained_image_to_mask_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__image_to_mask_network.eval()

    def __call__(self, image: np.ndarray, return_steps: bool = False):
        pred_binarized = self.image_to_mask_method(image=image)
        labelled_skeleton = self.mask_to_labelled_skeleton_method(mask=pred_binarized)
        results = self.labelled_skeleton_to_labels_method(
            labelled_skeleton=labelled_skeleton, return_detailed_results=return_steps
        )
        if return_steps:
            return results, pred_binarized, labelled_skeleton
        else:
            return results

    def image_to_mask_method(self, image: np.ndarray):
        """Method to convert image to mask

        Args:
            image (np.ndarray): image

        Returns:
            image (np.ndarray): mask
        """
        image = (
            torch.tensor(image, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.no_grad():
            pred = self.__image_to_mask_network(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_binarized = binarizer(image=pred_numpy_copy)
        return pred_binarized

    def mask_to_labelled_skeleton_method(self, mask: np.ndarray):
        """Method to make labelled skeleton from mask

        Args:
            mask (np.ndarray): segmented mask

        Returns:
            (np.ndarray): labelled skeleton
        """
        small_component_size = self.__min_cc_size_threshold
        cc_obj = ConnectedComponents(small_component_size=small_component_size)
        labelled_skeleton = cc_obj(dispersed_freq_time_segmented=mask)
        return labelled_skeleton

    def labelled_skeleton_to_labels_method(
        self, labelled_skeleton: np.ndarray, return_detailed_results: bool = False
    ):
        """Method to analyze each labels of the labelled skeleton

        Args:
            labelled_skeleton (np.ndarray): labelled skeleton
            return_detailed_results (bool, optional): _description_. Defaults to False.

        Returns:
            results in list of each labels
        """
        results = FitSegmentedTraces.return_detected_categories(
            labelled_skeleton=labelled_skeleton,
            return_detailed_results=return_detailed_results,
        )
        return results

    def validate_efficiency(
        self, image_data_set: ImageDataSet, label_data_set: LabelDataSet
    ):
        """Method to validate efficiency of the pipeline from image and label dataset

        Args:
            image_data_set (ImageDataSet): image dataset
            label_data_set (LabelDataSet): label dataset

        Returns:
            float: efficiency measure
        """
        total_items = len(image_data_set)
        correct_pred_noter = np.zeros(total_items)
        efficiency_measure = 0
        pbar = tqdm(total=total_items, desc="Starting...")
        for idx in range(total_items):
            image = image_data_set[idx]
            results = self(image=image)
            category_vals = list(results.values())
            category_types = list(results.keys())
            max_probable_category = np.argmax(category_vals)
            if "Pulsar" in category_types[max_probable_category]:
                is_pulsar_predicted = True
            else:
                is_pulsar_predicted = False
            # is_pulsar_predicted = self(image=image, return_bool=True)
            is_pulsar_there = label_data_set[idx]["Pulsar"] == 1
            if is_pulsar_there == is_pulsar_predicted:
                correct_pred_noter[idx] = 1
            efficiency_measure = sum(correct_pred_noter) / (idx + 1)
            pbar.set_description(f"efficiency_measure {efficiency_measure:0.2f}")
            pbar.update(1)
        return efficiency_measure

    def display_results_in_batch(
        self,
        image_data_set: ImageDataSet,
        mask_data_set:ImageDataSet,
        label_data_set: LabelDataSet,
        randomize: bool = True,
        ids_toshow: list = [0, 1],
        batch_size: int = 2,
    ):
        """Plot results of the pipeline with step outputs and comparison with pre-labelled dataset

        Args:
            image_data_set (ImageDataSet): Image dataset
            label_data_set (LabelDataSet): Label dataset of the images
            mask_data_set (ImageDataSet): Mask dataset of the image dataset
            randomize (bool, optional): If True, randomly chooses images from the dataset. Defaults to True.
            ids_toshow (list, optional): If radomize = False, then choose ids_show from dataset. Defaults to [0, 1].
            batch_size (int, optional): If randomize=True, then chooses batch_size images from set. Defaults to 2.
        """
        total_items = len(image_data_set)
        if randomize:
            ids_toshow = np.random.permutation(np.arange(total_items))[0:batch_size]
        else:
            ids_toshow = ids_toshow
        _, ax = plt.subplots(batch_size, 4, figsize=(4 * 4, 4 * batch_size))
        for i, idx in enumerate(ids_toshow):
            image_current = image_data_set[idx]
            mask_current = mask_data_set[idx]
            given_image_label_current = label_data_set[idx]
            results, pred_binarized, labelled_skeleton = self.__call__(
                image=image_current, return_steps=True
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}",fontsize=10)

            ax[i, 1].imshow(mask_current)
            ax[i, 1].set_title(f"idx: {idx}:{given_image_label_current}",fontsize=10)

            ax[i, 2].imshow(pred_binarized)
            ax[i, 2].set_title(f"Segmented Mask")

            if results:
                # fig1, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax[i, 3].imshow(image_current)
                for result in results:
                    x_coors_sorted = result[0]
                    y_coors_sorted = result[1]
                    func = result[3]
                    popt = result[2]
                    category = result[4]
                    num_points = result[5]
                    # print(x_coors_sorted,y_coors_sorted)
                    start_rect = (np.min(x_coors_sorted), np.min(y_coors_sorted))
                    end_rect = (np.max(x_coors_sorted), np.max(y_coors_sorted))
                    min_rect_dim = 10
                    height = [
                        (
                            np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            if np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    width = [
                        (
                            np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            if np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    if category == "Pulsar":
                        color = "r"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='r', facecolor='none')
                    else:
                        color = "b"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='g', facecolor='none')
                    rect = patches.Rectangle(
                        start_rect,
                        width[0],
                        height[0],
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                        label=f"{category}{num_points}",
                    )
                    ax[i, 3].plot(x_coors_sorted, y_coors_sorted, ".g")
                    fitted_y_coors = np.array([func(x, *popt) for x in x_coors_sorted])
                    ax[i, 3].plot(x_coors_sorted, fitted_y_coors, color)
                    ax[i, 3].add_patch(rect)
                    ax[i, 3].annotate(
                        f"{category}{num_points}", start_rect, fontsize=10, color=color
                    )
                ax[i, 3].set_xlabel("lags (a.u)")
                ax[i, 3].set_xlim(left=0, right=labelled_skeleton.shape[1])
                ax[i, 3].set_ylim(bottom=0, top=labelled_skeleton.shape[0])
                ax[i, 3].set_ylabel("freq channel")
                ax[i, 3].grid("True")
                ax[i, 3].invert_yaxis()
                ax[i, 3].set_aspect("auto")
        pass

    def test_on_real_data_from_npy_files(
        self,
        image_data_set: np.memmap,
        image_label_set: np.memmap | None = None,
        plot_randomly: bool = True,
        batch_size: int = 5,
    ):
        """Method to test pipeline on .npy file dataset

        Args:
            image_data_set (np.memmap): image dataset as numpy array
            image_label_set (np.memmap | None, optional): label dataset as numpy array. Defaults to None.
            plot_details (bool, optional): if True then plot the results. Defaults to False.
            plot_randomly (bool, optional): If True then randomly choose images from dataset. Defaults to True.
            batch_size (int, optional): number of images to test. minimum is 2. Defaults to 5.


        """

        if plot_randomly:
            random_idxs = np.random.permutation(np.arange(image_data_set.shape[0]))[
                0:batch_size
            ]
        else:
            random_idxs = np.arange(image_data_set.shape[0])[0:batch_size]
        _, ax = plt.subplots(batch_size, 3, figsize=(4 * 3, 4 * batch_size))
        for i, idx in enumerate(random_idxs):
            image_current = image_data_set[idx, :, :]
            image_current = image_current.astype(dtype=np.float32)
            image_current = image_current / np.max(image_current.flatten())
            if type(image_label_set) == np.memmap:
                given_image_label_current = image_label_set[idx]
            else:
                given_image_label_current = "NA"

            results, pred_binarized, labelled_skeleton = self.__call__(
                image=image_current, return_steps=True
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")

            ax[i, 1].imshow(pred_binarized)
            ax[i, 1].set_title(f"Segmented Mask")

            if results:
                # fig1, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax[i, 2].imshow(image_current)
                for result in results:
                    x_coors_sorted = result[0]
                    y_coors_sorted = result[1]
                    func = result[3]
                    popt = result[2]
                    category = result[4]
                    num_points = result[5]
                    # print(x_coors_sorted,y_coors_sorted)
                    start_rect = (np.min(x_coors_sorted), np.min(y_coors_sorted))
                    end_rect = (np.max(x_coors_sorted), np.max(y_coors_sorted))
                    min_rect_dim = 10
                    height = [
                        (
                            np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            if np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    width = [
                        (
                            np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            if np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    if category == "Pulsar":
                        color = "r"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='r', facecolor='none')
                    else:
                        color = "b"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='g', facecolor='none')
                    rect = patches.Rectangle(
                        start_rect,
                        width[0],
                        height[0],
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                        label=f"{category}{num_points}",
                    )
                    ax[i, 2].plot(x_coors_sorted, y_coors_sorted, ".g")
                    fitted_y_coors = np.array([func(x, *popt) for x in x_coors_sorted])
                    ax[i, 2].plot(x_coors_sorted, fitted_y_coors, color)
                    ax[i, 2].add_patch(rect)
                    ax[i, 2].annotate(
                        f"{category}{num_points}", start_rect, fontsize=10, color=color
                    )
                ax[i, 2].set_xlabel("lags (a.u)")
                ax[i, 2].set_xlim(left=0, right=labelled_skeleton.shape[1])
                ax[i, 2].set_ylim(bottom=0, top=labelled_skeleton.shape[0])
                ax[i, 2].set_ylabel("freq channel")
                ax[i, 2].grid("True")
                ax[i, 2].invert_yaxis()
                ax[i, 2].set_aspect("auto")


class PipelineImageToFilterToCCtoLabels:
    """Class implementing methods in sequence to generate segmented freq-time Image, filter it, then CC then to determine CCs to categories"""

    def __init__(
        self,
        image_to_mask_network: nn.Module,
        trained_image_to_mask_network_path: str,
        mask_filter_network: nn.Module,
        trained_mask_filter_network_path: str,
        min_cc_size_threshold: int = 10,
    ):
        self.__image_to_mask_network = image_to_mask_network
        self.__mask_filter_network = mask_filter_network
        self.__min_cc_size_threshold = min_cc_size_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__image_to_mask_network = self.__image_to_mask_network.to(self.device)
        self.__image_to_mask_network.load_state_dict(
            torch.load(trained_image_to_mask_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__image_to_mask_network.eval()

        self.__mask_filter_network = self.__mask_filter_network.to(self.device)
        self.__mask_filter_network.load_state_dict(
            torch.load(trained_mask_filter_network_path,map_location=torch.device(self.device),weights_only=True)
        )
        self.__mask_filter_network.eval()

    def __call__(self, image: np.ndarray, return_steps: bool = False):
        pred_binarized = self.image_to_mask_method(image=image)
        pred_binarized_copy = deepcopy(pred_binarized)
        pred_binarized_filtered = self.filter_mask_method(
            pred_binarized=pred_binarized_copy
        )
        # pred_binarized_filtered = pred_binarized_copy
        labelled_skeleton = self.mask_to_labelled_skeleton_method(
            mask=pred_binarized_filtered
        )
        results = self.labelled_skeleton_to_labels_method(
            labelled_skeleton=labelled_skeleton, return_detailed_results=return_steps
        )
        if return_steps:
            return results, pred_binarized, pred_binarized_filtered, labelled_skeleton
        else:
            return results

    def image_to_mask_method(self, image: np.ndarray):
        """Method to convert image to mask

        Args:
            image (np.ndarray): image

        Returns:
            image (np.ndarray): mask
        """
        image = (
            torch.tensor(image, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.no_grad():
            pred = self.__image_to_mask_network(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_binarized = binarizer(image=pred_numpy_copy)
        return pred_binarized

    def mask_to_labelled_skeleton_method(self, mask: np.ndarray):
        """Method to make labelled skeleton from mask

        Args:
            mask (np.ndarray): segmented mask

        Returns:
            (np.ndarray): labelled skeleton
        """
        small_component_size = self.__min_cc_size_threshold
        cc_obj = ConnectedComponents(small_component_size=small_component_size)
        labelled_skeleton = cc_obj(dispersed_freq_time_segmented=mask)
        return labelled_skeleton

    def filter_mask_method(self, pred_binarized: np.ndarray):
        """Method to filter out wrong segments in the segmented mask

        Args:
            pred_binarized (np.ndarray): segmented mask to filter

        Returns:
            np.ndarray: filtered segmented mask
        """
        if type(pred_binarized) == torch.Tensor:
            pred_binarized = pred_binarized.float().unsqueeze(0)
        else:
            pred_binarized = (
                torch.tensor(pred_binarized, requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
        # pred_binarized = torch.tensor(pred_binarized, requires_grad=False).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            pred_filtered_raw = self.__mask_filter_network(
                pred_binarized.to(self.device)
            )
        pred_filtered_raw = pred_filtered_raw.to("cpu")
        pred_filtered_raw_numpy = pred_filtered_raw.squeeze().numpy()
        pred_filtered_raw_numpy_copy = deepcopy(pred_filtered_raw_numpy)
        # binarizer = BinarizeToMask(binarize_func="gaussian_blur")
        binarizer = BinarizeToMask(binarize_func="thresh")
        pred_filtered_raw_binarized = binarizer(image=pred_filtered_raw_numpy_copy)
        return pred_filtered_raw_binarized

    def labelled_skeleton_to_labels_method(
        self, labelled_skeleton: np.ndarray, return_detailed_results: bool = False
    ):
        """Method to analyze each labels of the labelled skeleton

        Args:
            labelled_skeleton (np.ndarray): labelled skeleton
            return_detailed_results (bool, optional): _description_. Defaults to False.

        Returns:
            results in list of each labels
        """
        results = FitSegmentedTraces.return_detected_categories(
            labelled_skeleton=labelled_skeleton,
            return_detailed_results=return_detailed_results,
        )
        return results

    def validate_efficiency(
        self, image_data_set: ImageDataSet, label_data_set: LabelDataSet
    ):
        """Method to validate efficiency of the pipeline from image and label dataset

        Args:
            image_data_set (ImageDataSet): image dataset
            label_data_set (LabelDataSet): label dataset

        Returns:
            float: efficiency measure
        """
        total_items = len(image_data_set)
        correct_pred_noter = np.zeros(total_items)
        efficiency_measure = 0
        pbar = tqdm(total=total_items, desc="Starting...")
        for idx in range(total_items):
            image = image_data_set[idx]
            results = self(image=image)
            category_vals = list(results.values())
            category_types = list(results.keys())
            max_probable_category = np.argmax(category_vals)
            if "Pulsar" in category_types[max_probable_category]:
                is_pulsar_predicted = True
            else:
                is_pulsar_predicted = False
            # is_pulsar_predicted = self(image=image, return_bool=True)
            is_pulsar_there = label_data_set[idx]["Pulsar"] == 1
            if is_pulsar_there == is_pulsar_predicted:
                correct_pred_noter[idx] = 1
            efficiency_measure = sum(correct_pred_noter) / (idx + 1)
            pbar.set_description(f"efficiency_measure {efficiency_measure:0.2f}")
            pbar.update(1)
        return efficiency_measure

    def display_results_in_batch(
        self,
        image_data_set: ImageDataSet,
        mask_data_set: ImageDataSet,
        label_data_set: LabelDataSet,
        randomize: bool = True,
        ids_toshow: list = [0, 1],
        batch_size: int = 2,
    ):
        """Plot results of the pipeline with step outputs and comparison with pre-labelled dataset

        Args:
            image_data_set (ImageDataSet): Image dataset
            label_data_set (LabelDataSet): Label dataset of the images
            randomize (bool, optional): If True, randomly chooses images from the dataset. Defaults to True.
            ids_toshow (list, optional): If radomize = False, then choose ids_show from dataset. Defaults to [0, 1].
            batch_size (int, optional): If randomize=True, then chooses batch_size images from set. Defaults to 2.
        """
        total_items = len(image_data_set)
        if randomize:
            ids_toshow = np.random.permutation(np.arange(total_items))[0:batch_size]
        else:
            ids_toshow = ids_toshow
        _, ax = plt.subplots(batch_size, 5, figsize=(3 * 5, 3 * batch_size))
        for i, idx in enumerate(ids_toshow):
            image_current = image_data_set[idx]
            mask_current = mask_data_set[idx]
            # image_current= image_current.astype(dtype=np.float32)
            # image_current = image_current/np.max(image_current.flatten())
            given_image_label_current = label_data_set[idx]

            results, pred_binarized, pred_binarized_filtered, labelled_skeleton = (
                self.__call__(image=image_current, return_steps=True)
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}",fontsize=10)

            ax[i, 1].imshow(mask_current)
            ax[i, 1].set_title(f"Original Mask",fontsize=10)

            ax[i, 2].imshow(pred_binarized)
            ax[i, 2].set_title(f"Segmented Mask")

            ax[i, 3].imshow(pred_binarized_filtered)
            ax[i, 3].set_title(f"Filtered Mask")

            if results:
                # fig1, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax[i, 4].imshow(image_current)
                for result in results:
                    x_coors_sorted = result[0]
                    y_coors_sorted = result[1]
                    func = result[3]
                    popt = result[2]
                    category = result[4]
                    num_points = result[5]
                    # print(x_coors_sorted,y_coors_sorted)
                    start_rect = (np.min(x_coors_sorted), np.min(y_coors_sorted))
                    end_rect = (np.max(x_coors_sorted), np.max(y_coors_sorted))
                    min_rect_dim = 10
                    height = [
                        (
                            np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            if np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    width = [
                        (
                            np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            if np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    if category == "Pulsar":
                        color = "r"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='r', facecolor='none')
                    else:
                        color = "b"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='g', facecolor='none')
                    rect = patches.Rectangle(
                        start_rect,
                        width[0],
                        height[0],
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                        label=f"{category}{num_points}",
                    )
                    ax[i, 4].plot(x_coors_sorted, y_coors_sorted, ".g")
                    fitted_y_coors = np.array([func(x, *popt) for x in x_coors_sorted])
                    ax[i, 4].plot(x_coors_sorted, fitted_y_coors, color)
                    ax[i, 4].add_patch(rect)
                    ax[i, 4].annotate(
                        f"{category}{num_points}", start_rect, fontsize=10, color=color
                    )
                ax[i, 4].set_xlabel("lags (a.u)")
                ax[i, 4].set_xlim(left=0, right=labelled_skeleton.shape[1])
                ax[i, 4].set_ylim(bottom=0, top=labelled_skeleton.shape[0])
                ax[i, 4].set_ylabel("freq channel")
                ax[i, 4].grid("True")
                ax[i, 4].invert_yaxis()
                ax[i, 4].set_aspect("auto")

    def test_on_real_data_from_npy_files(
        self,
        image_data_set: np.memmap,
        image_label_set: np.memmap | None = None,
        plot_randomly: bool = True,
        batch_size: int = 5,
    ):
        """Method to test pipeline on .npy file dataset

        Args:
            image_data_set (np.memmap): image dataset as numpy array
            image_label_set (np.memmap | None, optional): label dataset as numpy array. Defaults to None.
            plot_details (bool, optional): if True then plot the results. Defaults to False.
            plot_randomly (bool, optional): If True then randomly choose images from dataset. Defaults to True.
            batch_size (int, optional): number of images to test. minimum is 2. Defaults to 5.


        """

        if plot_randomly:
            random_idxs = np.random.permutation(np.arange(image_data_set.shape[0]))[
                0:batch_size
            ]
        else:
            random_idxs = np.arange(image_data_set.shape[0])[0:batch_size]
        _, ax = plt.subplots(batch_size, 4, figsize=(4 * 4, 4 * batch_size))
        for i, idx in enumerate(random_idxs):
            image_current = image_data_set[idx, :, :]
            image_current = image_current.astype(dtype=np.float32)
            image_current = image_current / np.max(image_current.flatten())
            if type(image_label_set) == np.memmap:
                given_image_label_current = image_label_set[idx]
            else:
                given_image_label_current = "NA"

            results, pred_binarized, pred_binarized_filtered, labelled_skeleton = (
                self.__call__(image=image_current, return_steps=True)
            )
            ax[i, 0].imshow(image_current)
            ax[i, 0].set_title(f"idx: {idx}:{given_image_label_current}")

            ax[i, 1].imshow(pred_binarized)
            ax[i, 1].set_title(f"Segmented Mask")

            ax[i, 2].imshow(pred_binarized_filtered)
            ax[i, 2].set_title(f"Filtered Mask")

            if results:
                # fig1, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax[i, 3].imshow(image_current)
                for result in results:
                    x_coors_sorted = result[0]
                    y_coors_sorted = result[1]
                    func = result[3]
                    popt = result[2]
                    category = result[4]
                    num_points = result[5]
                    # print(x_coors_sorted,y_coors_sorted)
                    start_rect = (np.min(x_coors_sorted), np.min(y_coors_sorted))
                    end_rect = (np.max(x_coors_sorted), np.max(y_coors_sorted))
                    min_rect_dim = 10
                    height = [
                        (
                            np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            if np.max(y_coors_sorted) - np.min(y_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    width = [
                        (
                            np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            if np.max(x_coors_sorted) - np.min(x_coors_sorted)
                            > min_rect_dim
                            else min_rect_dim
                        )
                        for _ in range(1)
                    ]
                    if category == "Pulsar":
                        color = "r"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='r', facecolor='none')
                    else:
                        color = "b"
                        # rect = patches.Rectangle(start_rect, width[0] , height[0] , linewidth=1, edgecolor='g', facecolor='none')
                    rect = patches.Rectangle(
                        start_rect,
                        width[0],
                        height[0],
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                        label=f"{category}{num_points}",
                    )
                    ax[i, 3].plot(x_coors_sorted, y_coors_sorted, ".g")
                    fitted_y_coors = np.array([func(x, *popt) for x in x_coors_sorted])
                    ax[i, 3].plot(x_coors_sorted, fitted_y_coors, color)
                    ax[i, 3].add_patch(rect)
                    ax[i, 3].annotate(
                        f"{category}{num_points}", start_rect, fontsize=10, color=color
                    )
                ax[i, 3].set_xlabel("lags (a.u)")
                ax[i, 3].set_xlim(left=0, right=labelled_skeleton.shape[1])
                ax[i, 3].set_ylim(bottom=0, top=labelled_skeleton.shape[0])
                ax[i, 3].set_ylabel("freq channel")
                ax[i, 3].grid("True")
                ax[i, 3].invert_yaxis()
                ax[i, 3].set_aspect("auto")
            #plt.show()