import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from .information_packet_formats import Payload
from .neural_network_models import (
    UNet,
    CustomLossUNet,
    OneDconvEncoder,
    Simple1DCnnClassifier,
    CustomLossClassifier,
)
from .preprocessing import PrepareFreqTimeImage, BinarizeToMask
from .postprocessing import DelayGraph
from .pipeline_methods import PipelineImageToMask


class ImageMaskPair:
    """Class to pair Image and Mask"""

    def __init__(
        self,
        image: torch.Tensor = None,
        mask: torch.Tensor = None,
        descriptions: tuple[dict] = ({}, {}),
    ):
        self.image = image
        self.mask = mask
        self.descriptions = descriptions

    def __call__(self):
        return (self.image, self.mask)

    def __repr__(self) -> str:
        return f"ImageMaskPair with (image,mask) dimension {(self.image.shape,self.mask.shape)} and descriptions {(self.descriptions)}"

    def update_descriptions(self, descriptions: tuple[dict]):
        """Method to add descriptions about image and mask

        Args:
            descriptions (tuple[dict]): description
        """
        self.descriptions = descriptions

    def plot(self):
        """Plot Image and its mask

        Returns:
            ndarray: current axis of the plot
        """
        _, ax = plt.subplots(1, 2, figsize=(4*2, 4))
        # plt.figure()
        ax[0].imshow((self.image.detach().numpy()))
        ax[1].imshow((self.mask.detach().numpy()))
        ax[0].set_xlabel("Phase (a.u)")
        ax[0].set_ylabel("freq channel")
        ax[0].set_aspect("auto")
        ax[1].set_xlabel("Phase (a.u)")
        ax[1].set_ylabel("freq channel")
        ax[1].set_aspect("auto")
        return plt.gca()

    @classmethod
    def load_from_payload_address(
        cls,
        image_payload_address: str,
        mask_payload_address: str,
        image_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=False, do_resize=True
        ),
        mask_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=True, do_resize=True
        ),
    ):
        """Method to load from payload files

        Args:
            image_payload_address (str): full address to the image payload file
            mask_payload_address (str): full address to the mask payload file
            image_engine (PrepareFreqTimeImage, optional): engine to load image from payload file. Defaults to PrepareFreqTimeImage(do_rot_phase_avg=True,do_binarize=False,do_resize=True).
            mask_engine (PrepareFreqTimeImage, optional): engine to load image from payload file. Defaults to PrepareFreqTimeImage(do_rot_phase_avg=True,do_binarize=True,do_resize=True).

        Returns:
            (ImageMaskPair): ImageMaskPair object with loaded image and mask
        """
        cls_obj = cls()
        image_payload = Payload.read_payload_from_jsonfile(
            filename=image_payload_address
        )
        image = image_engine(payload_address=image_payload_address)
        image = image - min(image.flatten())
        image = image / max(image.flatten())
        cls_obj.image = torch.tensor(image, requires_grad=False)
        mask_payload = Payload.read_payload_from_jsonfile(filename=mask_payload_address)
        mask = mask_engine(payload_address=mask_payload_address)
        cls_obj.mask = torch.tensor(mask, requires_grad=False)
        cls_obj.update_descriptions(
            descriptions=(image_payload.description, mask_payload.description)
        )
        return cls_obj

    @classmethod
    def load_from_payload_and_make_in_mask(
        cls,
        image_payload_address: str,
        mask_payload_address: str,
        mask_maker_engine: PipelineImageToMask,
        image_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=False, do_resize=True
        ),
        mask_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=True, do_resize=True
        ),
    ):
        """Method to load image from payload and its mask. The loaded image is then converted to mask using a mask generator to be used as input mask (in_mask). This in_mask and mask pair is used for training filter networks

        Args:
            image_payload_address (str): full address to the image payload file
            mask_payload_address (str): full address to the mask payload file
            mask_maker_engine (PipelineImageToMask): engine to make mask from image
            image_engine (PrepareFreqTimeImage, optional): engine to load image from payload file. Defaults to PrepareFreqTimeImage(do_rot_phase_avg=True,do_binarize=False,do_resize=True).
            mask_engine (PrepareFreqTimeImage, optional): engine to load image from payload file. Defaults to PrepareFreqTimeImage(do_rot_phase_avg=True,do_binarize=True,do_resize=True).

        Returns:
            (ImageMaskPair): ImageMaskPair object with loaded image and mask
        """
        cls_obj = cls()
        image_payload = Payload.read_payload_from_jsonfile(
            filename=image_payload_address
        )
        image = image_engine(payload_address=image_payload_address)
        image = image - min(image.flatten())
        image = image / max(image.flatten())
        in_mask = mask_maker_engine(image=image)
        cls_obj.image = torch.tensor(in_mask, requires_grad=False)
        mask_payload = Payload.read_payload_from_jsonfile(filename=mask_payload_address)
        mask = mask_engine(payload_address=mask_payload_address)
        cls_obj.mask = torch.tensor(mask, requires_grad=False)
        cls_obj.update_descriptions(
            descriptions=(image_payload.description, mask_payload.description)
        )
        return cls_obj


class ImageToMaskDataset(Dataset):
    """Class to represent Image Mask dataset"""

    def __init__(
        self,
        image_tag: str,
        mask_tag: str,
        image_directory: str,
        mask_directory: str,
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

    def __getitem__(self, index):
        image_payload_address = self._image_directory + self._image_tag.replace(
            "*", str(index)
        )
        mask_payload_address = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        image_mask_pair = ImageMaskPair.load_from_payload_address(
            image_payload_address=image_payload_address,
            mask_payload_address=mask_payload_address,
            image_engine=self._image_engine,
            mask_engine=self._mask_engine,
        )
        image, mask = image_mask_pair()
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask

    def __get_descriptions__(self, index):
        image_payload_address = self._image_directory + self._image_tag.replace(
            "*", str(index)
        )
        mask_payload_address = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        image_mask_pair = ImageMaskPair.load_from_payload_address(
            image_payload_address=image_payload_address,
            mask_payload_address=mask_payload_address,
            image_engine=self._image_engine,
            mask_engine=self._mask_engine,
        )
        return image_mask_pair.descriptions

    def __len__(self):
        search_pattern = os.path.join(self._image_directory, self._image_tag)
        matching_files = glob.glob(search_pattern)
        num_items = len(matching_files)
        return num_items

    def plot(self, index):
        """Plot image mask pair represented by index

        Args:
            index (int): index of the pair
        """
        image_payload_address = self._image_directory + self._image_tag.replace(
            "*", str(index)
        )
        mask_payload_address = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        image_mask_pair = ImageMaskPair.load_from_payload_address(
            image_payload_address=image_payload_address,
            mask_payload_address=mask_payload_address,
            image_engine=self._image_engine,
            mask_engine=self._mask_engine,
        )
        image_mask_pair.plot()


class InMaskToMaskDataset(Dataset):
    """Class to represent InMask Mask dataset"""

    def __init__(
        self,
        image_tag: str,
        mask_tag: str,
        image_directory: str,
        mask_directory: str,
        mask_maker_engine: PipelineImageToMask,
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
        self._mask_maker_engine = mask_maker_engine
        self._device = device

    def __getitem__(self, index):
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
        in_mask, mask = image_mask_pair()
        in_mask = in_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return in_mask, mask

    def __get_descriptions__(self, index):
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
        return image_mask_pair.descriptions

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


class TrainImageToMaskNetworkModel:
    """Class involving methods to train Image (or InMask) to Mask Network"""

    def __init__(
        self,
        num_epochs: int,
        loss_criterion: torch.nn.Module = CustomLossUNet(),
        # loss_criterion:torch.nn.Module=torch.nn.BCELoss(),
        store_trained_model_at: str = "./syn_data/model/trained_unet_test_v0.pt",
        model: torch.nn.Module = UNet(),
    ):
        # self._model = model
        self._loss_criterion = loss_criterion
        self._epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self.trained_model_path = store_trained_model_at

    def __call__(self, image_mask_pairset: ImageToMaskDataset):
        self.train_model(image_mask_pairset=image_mask_pairset)

    def train_model(self, image_mask_pairset: ImageToMaskDataset):
        """Method to train the network

        Args:
            image_mask_pairset (ImageToMaskDataset): Image Mask Pair dataset

        Returns:
            (list,list): epoch number and loss in each epoch
        """
        self._model.train()  # Set the model to training mode
        loss_noter = []
        epoch_noter = []
        criterion = self._loss_criterion
        optimizer = self._optimizer
        dataset_loader = DataLoader(
            dataset=image_mask_pairset, batch_size=10, shuffle=True, pin_memory=True
        )
        for epoch in range(self._epochs):
            pbar = tqdm(total=len(dataset_loader), desc="Starting...")
            for images, masks in tqdm(dataset_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self._model(images.float())
                loss = criterion(outputs, masks.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss_measure {loss:0.2f}")
                pbar.update(1)
            loss_noter.append(loss.item())
            epoch_noter.append(epoch)
            print(
                f"Epoch [{epoch+1}/{self._epochs}], Mean Loss: {np.mean(loss_noter):0.4f}"
            )
            torch.save(self._model.state_dict(), self.trained_model_path)
            pbar.close()
        return epoch_noter, loss_noter

    def test_model(self, image: torch.tensor, plot_pred: bool = False):
        """Method to test the network

        Args:
            image (torch.tensor): image to convert to mask
            plot_pred (bool, optional): If True, plots the prediction from the NN. Defaults to False.

        Returns:
            (np.ndarray): prediction by the network as predicted mask
        """
        if type(image) == torch.Tensor:
            image = image.float().unsqueeze(0)
        else:
            #image = image - np.min(image.flatten())
            #if np.max(image.flatten())>0:
            #    image = image/np.max(image.flatten())
            image = (
                torch.tensor(image, requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
        self._model = self._model.to(self.device)
        self._model.load_state_dict(torch.load(self.trained_model_path,map_location=torch.device(self.device)))
        self._model.eval()
        with torch.no_grad():
            pred = self._model(image.to(self.device))
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func='gaussian_blur')
        binarizer = BinarizeToMask(binarize_func="thresh")
        # binarizer = BinarizeToMask(binarize_func='thresh_blur')
        # binarizer = BinarizeToMask(binarize_func='exponential')
        # binarizer = BinarizeToMask(binarize_func=None)
        pred_binarized = binarizer(image=pred_numpy_copy)
        if plot_pred:
            fig, (ax0, ax1,ax2) = plt.subplots(1, 3, figsize=(12, 4))
            ax0.imshow(image.squeeze().squeeze())
            ax0.set_title(f"Input Image")
            ax1.imshow(pred_numpy)
            ax1.set_title(f"Real Prediction")
            ax2.imshow(pred_binarized)
            ax2.set_title(f"Binarized Prediction")
            # fig.set_size_inches(10, 30)
            plt.show()
        return pred_binarized


class SignalLabelPair:
    """Class to represent Signal and label pair"""

    def __init__(
        self,
        signal: torch.Tensor | None = None,
        label: dict | None = None,
    ):
        self.signal = signal
        self.label = label

    def __call__(self):
        return (self.signal, self.label)

    def plot(self):
        """Method to plot Signal and label

        Returns:
            np.ndarray: current axis of the plot
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.figure()
        ax.plot((self.signal))
        ax.set_xlabel("Phase (a.u)")
        ax.set_ylabel("freq channel")
        ax.set_title(f"{self.label}")
        return plt.gca()

    @classmethod
    def load_from_payload_address(
        cls,
        mask_payload_address: str,
        mask_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=True, do_resize=True
        ),
    ):
        """Method to load mask from payload and convert it to signal

        Args:
            mask_payload_address (str): full path to the payload file of the mask
            mask_engine (PrepareFreqTimeImage, optional): engine to make mask from mask payload file. Defaults to PrepareFreqTimeImage( do_rot_phase_avg=True, do_binarize=True, do_resize=True ).

        Returns:
            SignalLabelPair: Signal Label Pair
        """

        cls_obj = cls()
        mask_payload = Payload.read_payload_from_jsonfile(filename=mask_payload_address)
        mask = mask_engine(payload_address=mask_payload_address)
        delay_graph_engine: DelayGraph = DelayGraph()
        x_lags, __ = delay_graph_engine(dispersed_freq_time=mask)
        cls_obj.signal = x_lags.flatten()
        cls_obj.label = mask_payload.description
        return cls_obj


class SignalToLabelDataset(Dataset):
    """Class to represent Signal Label pair dataset"""

    def __init__(
        self,
        mask_tag: str,
        mask_directory: str,
        mask_engine: PrepareFreqTimeImage = PrepareFreqTimeImage(
            do_rot_phase_avg=True, do_binarize=True, do_resize=True
        ),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self._mask_tag = mask_tag
        self._mask_directory = mask_directory
        self._mask_engine = mask_engine
        self._device = device

    def __getitem__(self, index):
        # image_payload_address = self._image_directory + self._image_tag.replace('*',str(index))
        mask_payload_address = self._mask_directory + self._mask_tag.replace(
            "*", str(index)
        )
        signal_label_pair = SignalLabelPair.load_from_payload_address(
            mask_payload_address=mask_payload_address, mask_engine=self._mask_engine
        )
        signal, label = signal_label_pair()
        label_vector = list(label.values())
        pulsar_present = label_vector[0]
        # print(label_vector)
        # total_sum = sum(label_vector)
        if pulsar_present > 0:
            label_vector = [1]
            # label_vector.append(0)
        else:
            label_vector = [0]
        label_vector = torch.tensor(label_vector, dtype=torch.float32)
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0), label_vector

    def __len__(self):
        search_pattern = os.path.join(self._mask_directory, self._mask_tag)
        matching_files = glob.glob(search_pattern)
        num_items = len(matching_files)
        return num_items

    def plot(self, index):
        """Method to plot Signal and label pair from the dataset

        Args:
            index (int): index of the pair

        Returns:
            np.ndarray: current axes of the plot
        """
        signal, label_vector = self[index]
        _, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        ax.plot(signal.detach().numpy().flatten())
        ax.set_xlabel("freqs (a.u)")
        ax.set_ylabel("lags")
        ax.set_aspect("auto")
        ax.set_ylim(0, 1)
        ax.grid()
        ax.set_title(f"Pulsar Present {label_vector[0]>=0.9}")
        return plt.gca()


class TrainSignalToLabelModel:
    """Class involving methods to train nn to classify signal into labels"""

    def __init__(
        self,
        num_epochs: int,
        # loss_criterion:torch.nn.Module=torch.nn.MSELoss(),
        loss_criterion: torch.nn.Module = torch.nn.BCELoss(reduction="mean"),
        # loss_criterion:torch.nn.Module=CustomLossClassifier(),
        store_trained_model_at: str = "./syn_data/model/trained_OneDconvEncoder_test_v0.pt",
        model: torch.nn.Module = Simple1DCnnClassifier(),
    ):
        self._loss_criterion = loss_criterion
        self._epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device, dtype=torch.float32)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self.trained_model_path = store_trained_model_at

    def __call__(self, signal_label_pairset: SignalToLabelDataset):
        self.train_model(signal_label_pairset=signal_label_pairset)

    def train_model(self, signal_label_pairset: SignalToLabelDataset):
        """Method to train the NN

        Args:
            signal_label_pairset (SignalToLabelDataset): signal label pair dataset

        Returns:
            (list,list): epoch number and the loss in each epoch
        """
        self._model.train()  # Set the model to training mode
        loss_noter = []
        epoch_noter = []
        criterion = self._loss_criterion
        optimizer = self._optimizer
        dataset_loader = DataLoader(
            dataset=signal_label_pairset, batch_size=20, shuffle=True, pin_memory=True
        )
        for epoch in range(self._epochs):
            pbar = tqdm(total=len(dataset_loader), desc="Starting...")
            for signal, label_vector in dataset_loader:
                signal = signal.to(self.device, dtype=torch.float32)
                label_vector = label_vector.to(self.device, dtype=torch.float32)
                outputs = self._model(signal.float())

                loss = criterion(outputs, label_vector.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss_measure {loss:0.2f}")
                pbar.update(1)
            # print('debug',outputs,label_vector)
            loss_noter.append(loss.item())
            epoch_noter.append(epoch)
            print(
                f"Epoch [{epoch+1}/{self._epochs}], Mean Loss: {np.mean(loss_noter):0.4f}"
            )
            torch.save(self._model.state_dict(), self.trained_model_path)
            pbar.close()
        return epoch_noter, loss_noter

    def test_model_from_signal(self, signal: torch.tensor, plot_pred: bool = False):
        """Method to test the nn to classify a signal

        Args:
            signal (torch.tensor): signal to classify
            plot_pred (bool, optional): If True, plots the signal and with prediction probability of each categories. Defaults to False.

        Returns:
            np.ndarray: Predicted labels are with probabilities
        """
        signal = signal.detach().numpy().flatten()
        # print('Lsignal',signal)
        # print('test_model_from_signal',signal.shape,type(signal))
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # print('test_model_sig sig.shape',signal.shape)
        signal = signal.to(self.device, dtype=torch.float32)

        self._model = self._model.to(self.device, dtype=torch.float32)
        self._model.load_state_dict(torch.load(self.trained_model_path))
        self._model.eval()
        with torch.no_grad():
            pred = self._model(signal)
            # print('test_model_from_signal pred',{pred})
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        # pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func='gaussian_blur')
        # pred_binarized = binarizer(image=pred_numpy_copy)
        if plot_pred:
            print(f"Predicted labels are with probabilities: {pred_numpy}")
        return pred_numpy

    def test_model(self, mask: np.ndarray, plot_pred: bool = False):
        """Method to test model from mask

        Args:
            mask (np.ndarray): mask from which category probability is predicted after generating the signal
            plot_pred (bool, optional): If True plots the results. Defaults to False.

        Returns:
            np.ndarray: Predicted labels are with probabilities
        """
        delay_graph_engine = DelayGraph()
        x_lags, __ = delay_graph_engine(mask)
        # print(x_lags)
        signal = x_lags.flatten()

        # print('Ssignal',signal)
        # print('test_model',signal.shape,type(signal))
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # print('test_model sig.shape',signal.shape)
        signal = signal.to(self.device, dtype=torch.float32)

        self._model = self._model.to(self.device, dtype=torch.float32)
        self._model.load_state_dict(torch.load(self.trained_model_path,map_location=torch.device(self.device),weights_only=True))
        self._model.eval()
        with torch.no_grad():
            pred = self._model(signal)
            # print('test_model pred',{pred})
        pred = pred.to("cpu")
        pred_numpy = pred.squeeze().numpy()
        # pred_numpy_copy = deepcopy(pred_numpy)
        # binarizer = BinarizeToMask(binarize_func='gaussian_blur')
        # pred_binarized = binarizer(image=pred_numpy_copy)
        if plot_pred:
            print(f"Predicted labels are with probabilities: {pred_numpy}")
        return pred_numpy
