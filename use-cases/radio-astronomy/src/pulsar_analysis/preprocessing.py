import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


from .information_packet_formats import Payload


class BinarizeToMask:
    """Class to define methods to binarize images"""

    def __init__(self, binarize_func: str = "gaussian_blur"):
        self.binarizing_func = binarize_func

    def __call__(self, image: np.ndarray):
        return self.binarize_protocol(image=image)

    def plot(self, image: np.ndarray):
        """Method to plot the binarized image

        Args:
            image (np.ndarray): "2D image Values range from 0-1"

        Returns:
            image (np.ndarray): binarized mask
        """
        image = self.__call__(image=image)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.figure()
        ax.imshow(image)
        ax.set_xlabel("Phase (a.u)")
        ax.set_ylabel("freq channel")
        ax.set_aspect("auto")
        return plt.gca()

    def binarize_protocol(self, image: np.ndarray):
        """call the binarize protocol method in an if loop condition (self.binarizing_func == 'binarize_func')

        Args:
            image (np.ndarray): image to binarize

        Returns:
            image (np.ndarray): binarized image
        """
        if self.binarizing_func == "exponential":
            image_copy = BinarizeToMask.exponential_method(image=image)

        elif self.binarizing_func == "gaussian_blur":
            image_copy = BinarizeToMask.gaussian_blurr(image=image)
        else:
            threshold = 0.5
            # foreground = image - background*1.05
            blurred_image = gaussian_filter(image, sigma=1 * 5) 
            image = image - blurred_image
            image = image - np.min(image.flatten())
            if np.max(image.flatten()) > 0:
                image = image / np.max(image.flatten())
            image_copy = image
            # image = 1/(1 + np.exp(-image))
            image_copy[image < threshold] = 0
            image_copy[image >= threshold] = 1
        # print(f"DEBUG:threshold {threshold}")

        return image_copy

    @staticmethod
    def gaussian_blurr(image: np.ndarray, sigma: int = 3):
        """Gaussian method to be called with binarize_func: str = "gaussian_blur" while initializing

        Args:
            image (np.ndarray): image to binarize
            sigma (int, optional): gaussian kernel. Defaults to 3.

        Returns:
            image (np.ndarray): binarized image
        """
        blurred_image = gaussian_filter(image, sigma=1 * sigma)
        background = gaussian_filter(blurred_image, sigma=3 * sigma)
        # background = gaussian_filter(background, sigma=1 * sigma)
        # background = gaussian_filter(background, sigma=1 * sigma)
        foreground = blurred_image - background * 1.05
        foreground = foreground - np.min(foreground.flatten())
        if np.max(foreground.flatten()) > 0:
            foreground = foreground / np.max(foreground.flatten())
        foreground[foreground <= 0.5] = 0
        foreground[foreground > 0.5] = 1
        return foreground

    @staticmethod
    def exponential_method(image: np.ndarray):
        """Exponential method to be called with binarize_func: str = "gaussian_blur" while initializing

        Args:
            image (np.ndarray): image to binarize

        Returns:
            image (np.ndarray): binarized image
        """
        intensities = image.flatten()
        hist, bin_edges = np.histogram(intensities, density=True, bins=25)
        bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
        # if self.binarizing_func == "exponential":
        func = BinarizeToMask.__exponential
        try:
            params, _ = curve_fit(
                func, bin_centers, hist, p0=[bin_centers[0], 1, 0], maxfev=5000
            )
            threshold = np.exp(1) * params[0]
        except:
            threshold = max(intensities) / 2
            print(
                f"WARNING: curve fitting couldnt found a fit therefore thresh is max_thresh/2 {threshold}"
            )

        image[image <= threshold] = 0
        image[image > threshold] = 1
        return image

    @staticmethod
    def __exponential(x, lam, amp_ex, offset):
        exponential_part = amp_ex * np.exp(-(x / lam)) + offset
        return exponential_part


class PrepareFreqTimeImage:
    """Class to implement methods to load and pre process radio payloads to freq-time image"""

    def __init__(
        self,
        do_rot_phase_avg: bool = True,
        do_resize: bool = True,
        do_binarize: bool = False,
        resize_size: tuple = (128, 128),
        binarize_engine: BinarizeToMask = BinarizeToMask(),
    ):
        self.do_rot_phase_avg = do_rot_phase_avg
        self.do_resize = do_resize
        self.resize_size = resize_size
        self.do_binarize = do_binarize
        self.binarize_engine = binarize_engine

    def __call__(self, payload_address: str):
        payload = Payload.read_payload_from_jsonfile(payload_address)
        return self.preparation_protocol(payload=payload)

    def __repr__(self) -> str:
        return f"PrepareFreqTimeImage class object with attributes {self.__dict__}"

    def preparation_protocol(self, payload: Payload):
        """Protocol to call methods to prepare freq-time graphs

        Args:
            payload (Payload): Payload class object made during the simulation

        Returns:
            image (np.ndarray): freq-time image
        """
        if self.do_rot_phase_avg:
            avg_dataframe = self.average_payload_rotphase(payload=payload)
        else:
            avg_dataframe = np.array(payload.dataframe).T
        if self.do_resize:
            avg_dataframe = resize(avg_dataframe, self.resize_size)
        if self.do_binarize:
            avg_dataframe = self.binarize_engine(image=avg_dataframe)
        return avg_dataframe

    def plot(self, payload_address: str):
        """plot the freq-time graph loaded from payload file

        Args:
            payload_address (str): path to the payload file
        """
        dataframe = self.__call__(payload_address=payload_address)
        fig1, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        # plt.figure()
        ax.imshow(dataframe)
        ax.set_xlabel("Phase (a.u)")
        ax.set_ylabel("freq channel")
        ax.set_aspect("auto")

    def average_payload_rotphase(self, payload: Payload):
        """Method to averge payload to 0-360 rotphase having multiple rotations

        Args:
            payload (Payload): Payload class object made during the simulation

        Returns:
            image (np.ndarray): freq-time image
        """
        rot_phases = np.array(payload.rot_phases)
        dataframe = np.array(payload.dataframe).T
        # slice_num = int(rot_phases/360)
        rot_phase_min = min(rot_phases)
        rot_phases = rot_phases - rot_phase_min
        rot_phase_max = max(rot_phases)
        rot_phase_scale = rot_phases[rot_phases < 360]
        if rot_phase_max < 360:
            avg_dataframe = dataframe
        else:
            current_dataframe_slice = np.zeros(
                (dataframe.shape[0], len(rot_phase_scale))
            )
            current_dataframe_noter = np.ones(
                (dataframe.shape[0], len(rot_phase_scale))
            )
            for i in range(int(np.ceil(rot_phase_max / 360.0))):
                current_slice_rot_phases = rot_phases[
                    np.logical_and(rot_phases < 360 * (i + 1), rot_phases >= 360 * i)
                ]
                if i == 0:
                    rot_phase_in_scale = current_slice_rot_phases % 360
                rot_phase_in_scale_logical = np.logical_and(
                    rot_phase_in_scale < max(current_slice_rot_phases % 360),
                    rot_phase_in_scale >= min(current_slice_rot_phases % 360),
                )
                dataframe_logical_noter = np.logical_and(
                    rot_phases < 360 * (i + 1), rot_phases > 360 * i
                )
                current_dataframe_slice[:, rot_phase_in_scale_logical] = dataframe[
                    :, dataframe_logical_noter
                ]
                current_dataframe_noter[:, rot_phase_in_scale_logical] = i + 1
                if i == 0:
                    sum_dataframe = current_dataframe_slice
                else:
                    sum_dataframe = sum_dataframe + current_dataframe_slice
            avg_dataframe = sum_dataframe / (current_dataframe_noter * 1.0)

        return avg_dataframe
