import sys

import numpy as np
import matplotlib.pyplot as plt

#sys.path.append('../../../prfi_generator/')

from prfi_gen.pulse_generator import Pulse
from prfi_gen.rfi_generator import BBRFI, NBRFI
from prfi_gen.utils import get_noised_and_normalized_array

from tqdm.notebook import trange


class DataCreator(object):
    """
    A class used to create datasets consisting of various signals and radio frequency interference (RFI).

    Attributes:
        resolution (tuple): The resolution of the dataset.
        data_structure (dict): A dictionary specifying the types and quantities of instances in the dataset.
        signal_details (dict): A dictionary containing details and parameters for each signal type.
        total_num_spects (int): The total number of spectra in the dataset.
        labels (np.ndarray): An array containing the labels for each instance in the dataset.
        data (np.ndarray): An array containing the data instances in the dataset.
        index (int): Index used for populating the dataset.
    """
    def __init__(self, resolution, data_structure, signal_detailes):
        """
        The constructor for DataCreator class.

        Args:
            resolution (tuple): Resolution of the dataset.
            data_structure (dict): Structure of the dataset.
            signal_details (dict): Details and parameters for signals in the dataset.
        """
        self.resolution = resolution
        self.data_structure = data_structure
        self.signal_detailes = signal_detailes

        self.total_num_spects = np.sum(list(data_structure.values()))

        self.lablels = np.empty((self.total_num_spects, ), dtype='U20') 
        self.data = np.empty((self.total_num_spects, *self.resolution), dtype=np.uint8)
        self.index = 0
    

    def create_dataset(self):
        """
        Method to create the dataset based on provided data structure and signal details.
        """
        for label, numbers in self.data_structure.items():
            if hasattr(self, f"fill_by_{label.lower()}"):
                getattr(self, f"fill_by_{label.lower()}")(numbers)
            else:
                print(f'Unknown label: {label}')

    def get_single_none(self):
        """
        Method to generate a single 'None' instance with specified noise and normalization.

        Returns:
            np.ndarray: A 'None' instance.
        """
        return get_noised_and_normalized_array(np.zeros(self.resolution), 3)[0].astype(np.uint8)

    def get_single_pulse(self):
        """
        Method to generate a single 'Pulse' instance with specified details and parameters.

        Returns:
            np.ndarray: A 'Pulse' instance.
        """
        pulse = Pulse(
            self.signal_detailes['Pulse']['DM'], 
            self.signal_detailes['Pulse']['f_hi'], 
            self.signal_detailes['Pulse']['f_lo'], 
            self.signal_detailes['Pulse']['n_channels'], 
            self.signal_detailes['Pulse']['t_resol']
        )
        
        pulse.generate_spectrogram(
            self.signal_detailes['Pulse']['amplitude'],
            self.signal_detailes['Pulse']['sigma'],
            self.signal_detailes['Pulse']['location']
        )
        
        pulse = pulse.add_zeros(pulse.compresed_pulse, 10, 'both')

        rand_index = np.random.randint(0, pulse.shape[1]-self.resolution[1])

        return get_noised_and_normalized_array(pulse[:,rand_index:rand_index+self.resolution[1]], np.random.uniform(*self.signal_detailes['Pulse']['noise_amplitude']))[0].astype(np.uint8)


    def get_single_bbrfi(self):
        """
        Method to generate a single 'BBRFI' instance with specified details and parameters.

        Returns:
            np.ndarray: A 'BBRFI' instance.
        """
        bbrfi = BBRFI(*self.resolution)
        rfi = bbrfi.get_rfi(1, np.random.uniform(*self.signal_detailes['BBRFI']['size_range']), np.random.randint(*self.signal_detailes['BBRFI']['location_range']))
        return get_noised_and_normalized_array(rfi, np.random.uniform(*self.signal_detailes['BBRFI']['noise_amplitude']))[0].astype(np.uint8)

    def get_single_nbrfi(self):
        """
        Method to generate a single 'NBRFI' instance with specified details and parameters.

        Returns:
            np.ndarray: A 'NBRFI' instance.
        """
        nbrfi = NBRFI(*self.resolution)
        rfi = nbrfi.get_rfi(1, np.random.uniform(*self.signal_detailes['NBRFI']['size_range']), np.random.randint(*self.signal_detailes['NBRFI']['location_range']))
        return get_noised_and_normalized_array(rfi, np.random.uniform(*self.signal_detailes['NBRFI']['noise_amplitude']))[0].astype(np.uint8)

    def fill_by_none(self, numbers):
        """
        Method to populate the dataset with 'None' instances.

        Args:
            numbers (int): The number of 'None' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='None creation', leave=True):
            self.data[self.index] = self.get_single_none()
            self.lablels[self.index] = 'None'
            self.index += 1

    def fill_by_pulse(self, numbers):
        """
        Method to populate the dataset with 'Pulse' instances.

        Args:
            numbers (int): The number of 'Pulse' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='Pulse creation', leave=True):
            self.data[self.index] = self.get_single_pulse()
            self.lablels[self.index] = 'Pulse'
            self.index += 1

    def fill_by_bbrfi(self, numbers):
        """
        Method to populate the dataset with 'BBRFI' instances.

        Args:
            numbers (int): The number of 'BBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='BBRFI creation', leave=True):
            self.data[self.index] = self.get_single_bbrfi()
            self.lablels[self.index] = 'BBRFI'
            self.index += 1

    def fill_by_nbrfi(self, numbers):
        """
        Method to populate the dataset with 'NBRFI' instances.

        Args:
            numbers (int): The number of 'NBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='NBRFI creation', leave=True):
            self.data[self.index] = self.get_single_nbrfi()
            self.lablels[self.index] = 'NBRFI'
            self.index += 1
    
    def fill_by_pulse_bbrfi(self, numbers):
        """
        Method to populate the dataset with 'Pulse+BBRFI' instances.

        Args:
            numbers (int): The number of 'Pulse+BBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='Pulse+BBRFI creation', leave=True):
            pulse_uint16 = self.get_single_pulse().astype(np.uint16)
            bbrfi_uint16 = self.get_single_bbrfi().astype(np.uint16)

            # Sum up the uint16 arrays
            pulse_bbrfi_uint16 = pulse_uint16 + bbrfi_uint16

            # Normalize to 255
            max_value = np.max(pulse_bbrfi_uint16)
            normalized_pulse_bbrfi_uint16 = np.round((pulse_bbrfi_uint16 / max_value) * 255).astype(np.uint16)
 
            self.data[self.index] = normalized_pulse_bbrfi_uint16.astype(np.uint8)
            self.lablels[self.index] = 'Pulse+BBRFI'
            self.index += 1
    
    def fill_by_pulse_nbrfi(self, numbers):
        """
        Method to populate the dataset with 'Pulse+NBRFI' instances.

        Args:
            numbers (int): The number of 'Pulse+NBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='Pulse+NBRFI creation', leave=True):
            pulse_uint16 = self.get_single_pulse().astype(np.uint16)
            nbrfi_uint16 = self.get_single_nbrfi().astype(np.uint16)

            # Sum up the uint16 arrays
            pulse_bbrfi_uint16 = pulse_uint16 + nbrfi_uint16

            # Normalize to 255
            max_value = np.max(pulse_bbrfi_uint16)
            normalized_pulse_bbrfi_uint16 = np.round((pulse_bbrfi_uint16 / max_value) * 255).astype(np.uint16)
 
            self.data[self.index] = normalized_pulse_bbrfi_uint16.astype(np.uint8)
            self.lablels[self.index] = 'Pulse+NBRFI'
            self.index += 1
    
    def fill_by_bbrfi_nbrfi(self, numbers):
        """
        Method to populate the dataset with 'NBRFI+BBRFI' instances.

        Args:
            numbers (int): The number of 'NBRFI+BBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='NBRFI+BBRFI creation', leave=True):
            nbrfi_uint16 = self.get_single_nbrfi().astype(np.uint16)
            bbrfi_uint16 = self.get_single_bbrfi().astype(np.uint16)

            # Sum up the uint16 arrays
            nbrfi_bbrfi_uint16 = nbrfi_uint16 + bbrfi_uint16

            # Normalize to 255
            max_value = np.max(nbrfi_bbrfi_uint16)
            normalized_nbrfi_bbrfi_uint16 = np.round((nbrfi_bbrfi_uint16 / max_value) * 255).astype(np.uint16)
 
            self.data[self.index] = normalized_nbrfi_bbrfi_uint16.astype(np.uint8)
            self.lablels[self.index] = 'NBRFI+BBRFI'
            self.index += 1

    def fill_by_pulse_bbrfi_nbrfi(self, numbers):
        """
        Method to populate the dataset with 'Pulse+NBRFI+BBRFI' instances.

        Args:
            numbers (int): The number of 'Pulse+NBRFI+BBRFI' instances to add to the dataset.
        """
        for _ in trange(numbers, desc='Pulse+NBRFI+BBRFI creation', leave=True):
            pulse_uint16 = self.get_single_pulse().astype(np.uint16)
            nbrfi_uint16 = self.get_single_nbrfi().astype(np.uint16)
            bbrfi_uint16 = self.get_single_bbrfi().astype(np.uint16)

            # Sum up the uint16 arrays
            pulse_nbrfi_bbrfi_uint16 = pulse_uint16 + nbrfi_uint16 + bbrfi_uint16

            # Normalize to 255
            max_value = np.max(pulse_nbrfi_bbrfi_uint16)
            normalized_pulse_nbrfi_bbrfi_uint16 = np.round((pulse_nbrfi_bbrfi_uint16 / max_value) * 255).astype(np.uint16)
 
            self.data[self.index] = normalized_pulse_nbrfi_bbrfi_uint16.astype(np.uint8)
            self.lablels[self.index] = 'Pulse+NBRFI+BBRFI'
            self.index += 1
    
    
    def suffle_data(self):
        """
        Method to shuffle the data and labels of the dataset in-place.
        """
        shuffled_indices = np.random.permutation(self.total_num_spects)

        self.data = self.data[shuffled_indices]
        self.lablels = self.lablels[shuffled_indices]

    
    @staticmethod
    def format_signal_details(signal_details):
        """
        Method to convert signal details dictionary to a formatted string.
        
        Args:
            signal_details (dict): A dictionary containing the signal details.
            
        Returns:
            str: Formatted string containing the signal details.
        """
        formatted_lines = []
        units = {'f_hi': 'GHz', 'f_lo': 'GHz', 't_resol': 'seconds', 'DM': 'pc cm^-3'}

        for key, details in signal_details.items():
            formatted_details = []

            for k, v in details.items():
                unit = units.get(k, '')

                if isinstance(v, tuple):
                    formatted_details.append(f"        {k} = {v[0]} - {v[1]} {unit}".rstrip())
                else:
                    formatted_details.append(f"        {k} = {v} {unit}".rstrip())

            formatted_section = f"{key}\n" + "\n".join(formatted_details)
            formatted_lines.append(formatted_section)

        return "\n".join(formatted_lines)

    
    def save_dataset(self, name, path_for_saving):
        """
        Method to save the created dataset along with its description.

        Args:
            name (str): Name of the dataset.
            path_for_saving (str): Path where the dataset should be saved.
        """
        newline = "\n"

        message = (
        f'Discriprion for the dataset in {name} saved\n'
        f'{path_for_saving}\n'
        f'\n'
        f'The dataset includes only synthetic data.\n'
        f'The dataset includes next classes:\n'
        f'{f"{newline}".join([f"{key} ({value} instances)" for key, value in self.data_structure.items()])}\n'
        f'\n'
        f'The initialisation parametrs of the basic classes is following:\n'
        f'{self.format_signal_details(self.signal_detailes)}\n'
        f'\n'
        f'\n'
        f'Datatype is uint8.'
        f'\n'    
        f'Totaly the dataset includes {self.total_num_spects} instances.\n'
        )
        
        with open(f'{path_for_saving}{name}_discription.txt', 'w') as file:
            file.write(message)

        np.save(f'{path_for_saving}{name}.npy', self.data)
        np.save(f'{path_for_saving}{name}_labels.npy', self.lablels)
        


    @staticmethod
    def plot_images(image_array, num, labels=None):
        """
        Method to plot images in the dataset.

        Args:
            image_array (np.ndarray): Array containing images to be plotted.
            num (int): Number of images to plot.
            labels (np.ndarray, optional): Array containing labels for the images. Defaults to None.
        """
        subplot_size = 3 

        if num == 1:
            subint = np.random.choice(image_array.shape[0], num)[0]
            plt.imshow(image_array[subint], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            if isinstance(labels, np.ndarray):
                plt.xlabel(labels[subint])
            plt.show()
        elif 2 <= num <= 3:
            fig, axs = plt.subplots(1, num, figsize=(subplot_size * num, 5))
            for i, subint in enumerate(np.random.choice(image_array.shape[0], num, replace=False)):
                axs[i].imshow(image_array[subint], cmap='gray')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                if isinstance(labels, np.ndarray):
                    axs[i].set_xlabel(labels[subint])
            plt.show()
        else:
            n_rows = int(np.ceil(np.sqrt(num)))
            n_cols = int(np.ceil(num / n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(subplot_size * n_cols, subplot_size * n_rows))
            
            for i, subint in enumerate(np.random.choice(image_array.shape[0], num, replace=False)):
                row = i // n_cols
                col = i % n_cols
                axs[row, col].imshow(image_array[subint], cmap='gray')
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                if isinstance(labels, np.ndarray):
                    axs[row, col].set_xlabel(labels[subint])
            
            for i in range(num, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axs[row, col].axis('off')  # Turn off axes for empty subplots
                
            plt.show()