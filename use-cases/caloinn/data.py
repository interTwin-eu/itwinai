import os
from typing import Optional, Tuple
import h5py
import numpy as np
import requests
from torch.utils.data import Dataset, random_split
import torch
from src.XMLHandler import XMLHandler
from itwinai.components import DataGetter, monitor_exec, DataSplitter


class CaloChallengeDownloader(DataGetter):
    def __init__(
        self,
        data_path: Optional[str] = "./calochallenge_data/",
        dataset_type: Optional[str] = "dataset_1_photons",
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        super().__init__()
        self.data_path = data_path
        self.dataset_type = dataset_type

        type_to_link_train = {
            "dataset_1_photons": [
                "https://zenodo.org/records/8099322/files/dataset_1_photons_1.hdf5"
            ],
            "dataset_1_pions": [
                "https://zenodo.org/records/8099322/files/dataset_1_pions_1.hdf5"
            ],
            "dataset_2": ["https://zenodo.org/records/6366271/files/dataset_2_1.hdf5"],
            "dataset_3": [
                "https://zenodo.org/records/6366324/files/dataset_3_1.hdf5",
                "https://zenodo.org/records/6366324/files/dataset_3_2.hdf5",
            ],
        }

        type_to_link_test = {
            "dataset_1_photons": [
                "https://zenodo.org/records/8099322/files/dataset_1_photons_2.hdf5"
            ],
            "dataset_1_pions": [
                "https://zenodo.org/records/8099322/files/dataset_1_pions_2.hdf5"
            ],
            "dataset_2": ["https://zenodo.org/records/6366271/files/dataset_2_2.hdf5"],
            "dataset_3": [
                "https://zenodo.org/records/6366324/files/dataset_3_3.hdf5",
                "https://zenodo.org/records/6366324/files/dataset_3_4.hdf5",
            ],
        }

        if dataset_type not in type_to_link_train.keys():
            print("WARNING! Dataset type is invalid " "Loading dataset 1 photon")
            self.dataset_type = "dataset_1_photons"

        self.data_train_url = type_to_link_train[self.dataset_type]
        self.data_test_url = type_to_link_test[self.dataset_type]

    @monitor_exec
    def execute(self):
        # Download data
        if not os.path.exists(self.data_path):
            if self.data_train_url is None:
                print(
                    "WARNING! Train data URL is None. "
                    "Skipping train dataset downloading"
                )
            if self.data_test_url is None:
                print(
                    "WARNING! Test data URL is None. "
                    "Skipping test dataset downloading"
                )

        def download_rename(url_list, prefix):
            for url in url_list:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    new_filename = f"{url.split('/')[-1].split('.')[0]}_{prefix}.hdf5"
                    with open(os.path.join(self.data_path, new_filename), "wb") as file:
                        # Write the content of the response to the file
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f"File downloaded and saved as: {new_filename}")
                else:
                    print(f"Failed to download file.")

        print(f"Downloading train dataset to {self.data_path}")
        download_rename(self.data_train_url, "train")
        print(f"Downloading test dataset to {self.data_path}")
        download_rename(self.data_test_url, "test")


class CalochallengeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        dataset_trainortest: str,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        dep_cut: float = 1e10,
        width_noise: float = 1e-7,
        fixed_noise: bool = False,
        noise: bool = False,
    ) -> None:
        if dataset_trainortest not in ["train", "test"]:
            print(
                "WARNING! Subset type (train ir test) is not defined" "Loading as train"
            )
            self.dataset_trainortest = "train"
        else:
            self.dataset_trainortest = dataset_trainortest

        torch.set_default_dtype(torch.float32)

        # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
        dataset_to_particle_type = {
            "dataset_1_photons": "photon",
            "dataset_1_pions": "pion",
            "dataset_2": "electron",
            "dataset_3": "electron",
        }
        if dataset_type not in dataset_to_particle_type.keys():
            print("WARNING! Dataset type is invalid " "Loading dataset 1 photon")
            self.dataset_type = "dataset_1_photons"
        else:
            self.dataset_type = dataset_type

        xml_handler = XMLHandler(
            particle_name=dataset_to_particle_type[self.dataset_type],
            filename="./calochallenge_binning/binning_" + self.dataset_type + ".xml",
        )
        self.layer_boundaries = np.unique(xml_handler.GetBinEdges())

        self.data_path = data_path

        self.noise = noise
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise

        self.data = {}

        self._load_data()
        self._preprocess_data(eps, u0up_cut, u0low_cut, dep_cut)
        if self.fixed_noise and self.noise:
            noise = self.noise_distribution.sample(self.x.shape) * self.width_noise
            self.x += noise.reshape(self.x.shape)

        if self.noise:
            self.noise_distribution = torch.distributions.Uniform(
                torch.tensor(0.0), torch.tensor(1.0)
            )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = self.x[idx]
        if not self.fixed_noise and self.noise:
            noise = self.noise_distribution.sample(data.shape) * self.width_noise
            data = (torch.tensor(data) + noise.reshape(data.shape).clone()).clone()
        cond = self.cond[idx]
        return data, cond

    def _load_data(self) -> None:
        print("Searching in :", self.data_path)
        files = [
            os.path.join(self.data_path, filename)
            for filename in os.listdir(self.data_path)
            if self.dataset_type in filename and self.dataset_trainortest in filename
        ]
        print("Found {} files. ".format(len(files)))
        if len(files) == 0:
            raise RuntimeError(f"No H5 files found at '{self.data_path}'!")

        for file in files:
            with h5py.File(file, "r") as data_file:
                if "energy" not in self.data:
                    self.data["energy"] = data_file["incident_energies"][:] / 1.0e3
                else:
                    self.data["energy"] = np.concatenate(
                        (self.data["energy"], data_file["incident_energies"][:] / 1.0e3)
                    )

                for layer_index, (layer_start, layer_end) in enumerate(
                    zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])
                ):
                    self.data[f"layer_{layer_index}"] = (
                        data_file["showers"][..., layer_start:layer_end] / 1.0e3
                    )

        print(self.data[f"layer_{layer_index}"].shape)

    def _preprocess_data(self, eps, u0up_cut, u0low_cut, dep_cut) -> None:
        """Transforms the dict 'data' into the ndarray 'x'. Furthermore, the events
        are masked and the extra dims are appended to the incident energies"""
        c, x = self.get_energy_and_sorted_layers()
        self.data = None

        # Remove all no-interaction events (only 0.7%)
        binary_mask = np.sum(x, axis=1) >= 0

        c, extra_dims = self.get_energy_dims(x, c, eps)

        binary_mask &= extra_dims[:, 0] < u0up_cut
        binary_mask &= extra_dims[:, 0] >= u0low_cut

        binary_mask &= (x < dep_cut).prod(-1) != 0

        print(f"cut on zero energy dep.: #", (np.sum(x, axis=1) >= 0).sum())
        print(f"cut on u0 upper {u0up_cut}: #", (extra_dims[:, 0] < u0up_cut).sum())
        print(f"cut on u0 lower {u0low_cut}: #", (extra_dims[:, 0] >= u0low_cut).sum())
        print(f"dep cut {dep_cut}: #", (x < dep_cut).prod(-1).sum())

        x = x[binary_mask]
        c = c[binary_mask]
        extra_dims = extra_dims[binary_mask]
        print("final shape of dataset: ", x.shape)

        x = self.normalize_layers(x, c, eps)
        x = np.concatenate((x, extra_dims), axis=1)

        self.x = x
        self.cond = c

    def get_energy_and_sorted_layers(self):
        """returns the energy and the sorted layers from the data dict"""

        # Get the incident energies
        energy = self.data["energy"]

        # Get the number of layers layers from the keys of the data array
        number_of_layers = len(self.data) - 1

        # Create a container for the layers
        layers = []

        # Append the layers such that they are sorted.
        for layer_index in range(number_of_layers):
            layer = f"layer_{layer_index}"

            layers.append(self.data[layer])

        return energy, np.concatenate(layers, axis=1)

    def get_energy_dims(self, x, c, eps):
        """Appends the extra dimensions and the layer energies to the conditions
        The layer energies will always be the last #layers entries, the extra dims will
        be the #layers entries directly after the first entry - the incident energy.
        Inbetween additional features might be appended as further conditions"""

        x = np.copy(x)
        c = np.copy(c)

        layer_energies = []

        for layer_start, layer_end in zip(
            self.layer_boundaries[:-1], self.layer_boundaries[1:]
        ):

            # Compute total energy of current layer
            layer_energy = np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True)

            # Normalize current layer
            x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (
                layer_energy + eps
            )

            # Store its energy for later
            layer_energies.append(layer_energy)

        layer_energies_np = np.array(layer_energies).T[0]

        # Compute the generalized extra dimensions
        extra_dims = [np.sum(layer_energies_np, axis=1, keepdims=True) / c]

        for layer_index in range(len(self.layer_boundaries) - 2):
            extra_dim = layer_energies_np[..., [layer_index]] / (
                np.sum(layer_energies_np[..., layer_index:], axis=1, keepdims=True)
                + eps
            )
            extra_dims.append(extra_dim)

        # Collect all the conditions
        extra_dims = np.concatenate(extra_dims, axis=1)

        return c, extra_dims

    def normalize_layers(self, x, c, eps=1.0e-10):
        """Normalizes each layer by its energy"""

        # Prevent inplace operations
        x = np.copy(x)
        c = np.copy(c)

        # Use the exact layer energies for numerical stability
        for layer_index, (layer_start, layer_end) in enumerate(
            zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])
        ):
            x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (
                np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps
            )

        return x


class CalochallengeDataSplitter(DataSplitter):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        dep_cut: float = 1e10,
        width_noise: float = 1e-7,
        fixed_noise: bool = False,
        noise: bool = False,
        train_proportion: int | float = 1.0,
        validation_proportion: int | float = 0.0,
        test_proportion: int | float = 0.0,
        rnd_seed: Optional[int] = None,
    ) -> None:
        super().__init__(train_proportion, validation_proportion, test_proportion)
        self.save_parameters(**self.locals2params(locals()))

        self.rnd_seed = rnd_seed
        if validation_proportion >= 0.0 and validation_proportion < 1:
            self.validation_proportion = validation_proportion
        else:
            print(
                "WARNING! Wrong value of validation dataset proportion"
                "Set validation dataset proportion to 0.1"
            )

        if rnd_seed:
            self.generator = torch.Generator().manual_seed(self.rnd_seed)
        else:
            self.generator = torch.Generator()

        self.train_dataset = CalochallengeDataset(
            data_path=data_path,
            dataset_type=dataset_type,
            dataset_trainortest="train",
            eps=eps,
            u0up_cut=u0up_cut,
            u0low_cut=u0low_cut,
            dep_cut=dep_cut,
            width_noise=width_noise,
            fixed_noise=fixed_noise,
            noise=noise,
        )
        self.test_dataset = CalochallengeDataset(
            data_path=data_path,
            dataset_type=dataset_type,
            dataset_trainortest="test",
            eps=eps,
            u0up_cut=u0up_cut,
            u0low_cut=u0low_cut,
            dep_cut=dep_cut,
            noise=False,
        )

    @monitor_exec
    def execute(self) -> Tuple:
        # Split train file into train and validation sets
        [train_dataset, validation_dataset] = random_split(
            self.train_dataset,
            [1 - self.validation_proportion, self.validation_proportion],
            generator=self.generator,
        )
        return train_dataset, validation_dataset, self.test_dataset
