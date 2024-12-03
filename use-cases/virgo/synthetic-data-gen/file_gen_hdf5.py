# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Unknown
# --------------------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
from time import time

import h5py
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from tqdm import tqdm


def append_to_hdf5_dataset(
    file_path: Path,
    dataset_name: str,
    array: np.ndarray,
    expected_datapoint_shape: tuple,
):
    if tuple(array.shape[1:]) != expected_datapoint_shape:
        actual_shape_str = ", ".join(str(s) for s in array.shape)
        expected_shape_str = ", ".join(str(s) for s in expected_datapoint_shape)
        raise ValueError(
            f"'array' has an incorrect shape: ({actual_shape_str}). "
            f"Should have been (x, {expected_shape_str})."
        )

    print(f"Appending to file: '{str(file_path.resolve())}'.")
    with h5py.File(file_path, "a") as f:
        dset=f[dataset_name]
        dset.resize(dset.shape[0] + array.shape[0], axis=0)
        dset[-array.shape[0] :] = array


def generate_hdf5_dataset(
    output_file: str = "virgo_data.hdf5",
    dataset_name: str = "virgo_dataset",
    num_datapoints=5,
    duration=10,
    sample_rate=500,
    num_aux_channels=10,
    num_waves_range=(10, 15),
    noise_amplitude=0.1,
    num_processes=4,
    square_size=64,
    datapoints_per_file=10,
    seed=None,
) -> None:
    """Generate a folder with num_files pickle files containing synthetic gravitational
    waves data.

    Args:
        folder_name (string): the path and name where the files will be stored
        num_files (int): Number of files which will be created.
        duration (float): Duration of the time series data in seconds (default is 6
            seconds).
        num_aux_channels (int): Number of auxiliary channels, containing the data from
            the auxiliary sensors in the detector which do not go into the strain.
        sample_rate (int): Sampling rate of the time series data in samples per second
            (default is 500 samples per second).
        num_waves_range (tuple): Range for the random number of sine waves to be
            generated for each time series. Format: (min_num_waves, max_num_waves)
            (default is (10, 15)).
        noise_amplitude (float): Amplitude of the smooth random noise added to the
            time series data (default is 0.1).
        num_processes (int): Number of cores for multiprocess (default 20)
        square_size (int): Size in pixels of qplot image (default is 500 samples per
            second).
        datapoints_per_file (int): number of independent datapoints per pickle file.
    """
    sys.path.append(str(Path.cwd().resolve()))
    from src.dataset import generate_cut_image_dataset

    if seed is None:
        # Since we can't retrieve the seed from numpy, we set it so that we can
        # store it in the dataset. More information:
        # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
        seed = np.random.randint(0, 2**32, dtype=np.uint64)
    np.random.seed(seed)

    datapoints = []

    # Creating empty HDF5 file
    datapoint_shape = (num_aux_channels + 1, square_size, square_size)
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating/overwriting file: '{output_file_path.resolve()}'.")
    with h5py.File(output_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name,
            shape=(0, *datapoint_shape),
            maxshape=(None, *datapoint_shape),
            dtype=np.float32,
        )
        dataset.attrs["Description"] = (
            "Synthetic time series data for the Virgo use case"
        )
        dataset.attrs["main_channel_idx"] = 0
        dataset.attrs["seed"] = str(seed)

    for f in tqdm(range(num_datapoints)):
        times = np.linspace(0, duration, duration * sample_rate)
        # Initialise the main data as a list of zeros
        main_data = np.zeros(len(times))
        dictionary_aux = {}
        for i in range(num_aux_channels):
            channel_name = "Aux_" + str(i)

            # Initialize an array to store the generated wave data for this row
            wave_data = np.zeros(len(times))
            # Determine the number of sine waves to generate for this column randomly
            num_waves = np.random.randint(*num_waves_range)

            # Generate each sine wave
            for _ in range(num_waves):
                # Randomly generate parameters for the sine wave (amplitude, frequency, phase)
                amplitude = np.random.uniform(0.5, 2.0)
                frequency = np.random.uniform(0.5, 5.0)
                phase = np.random.uniform(0, 2 * np.pi)

                # Generate the sine wave and add it to the wave_data
                wave_data += amplitude * np.sin(2 * np.pi * frequency * times + phase)

            # Add smooth random noise to the wave data
            smooth_noise = np.random.normal(0, noise_amplitude, len(times))
            wave_data += smooth_noise

            coeff = np.random.rand()

            main_data += coeff * wave_data

            # Create a TimeSeries object from the wave data
            ts = TimeSeries(wave_data, t0=0, dt=1 / sample_rate)

            dictionary_aux[channel_name] = [ts]

        # Creating the main timeseries
        main_data += np.random.normal(0, noise_amplitude)
        ts_main = TimeSeries(main_data, dt=1 / sample_rate)

        main_entry = {"Main": [ts_main]}

        dictionary = {**main_entry, **dictionary_aux}

        # turn dictionary into dataframe
        df_ts = pd.DataFrame(dictionary)

        datapoints.append(df_ts)
        if len(datapoints) % datapoints_per_file == 0:
            # Converting to numpy array
            df_concat = pd.concat(datapoints)
            df_concat = generate_cut_image_dataset(
                df=df_concat,
                channels=list(df_ts.columns),
                num_processes=num_processes,
                square_size=square_size,
            )
            value_array = df_concat.to_numpy()
            value_array = np.stack(
                [np.stack(row) for row in value_array], dtype=np.float32
            )

            append_to_hdf5_dataset(
                file_path=output_file_path,
                dataset_name=dataset_name,
                array=value_array,
                expected_datapoint_shape=datapoint_shape,
            )
            datapoints = []

    # Adding any remaining datapoints
    if len(datapoints) > 0:
        df_concat = pd.concat(datapoints)
        df_concat = generate_cut_image_dataset(
            df=df_concat,
            channels=list(df_concat.columns),
            num_processes=num_processes,
            square_size=square_size,
        )
        value_array = df_concat.to_numpy()
        value_array = np.stack([np.stack(row) for row in value_array], dtype=np.float32)
        append_to_hdf5_dataset(
            file_path=output_file_path,
            dataset_name=dataset_name,
            array=value_array,
            expected_datapoint_shape=datapoint_shape,
        )
        datapoints = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Virgo Dataset Generation")
    parser.add_argument(
        "--num-datapoints", type=int, help="Number of datapoints to be created."
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        help="How often to save to file while generating. Also saves when finished. ",
        default=100,
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="How many processes to use when generating data",
        default=1,
    )
    parser.add_argument(
        "--save-location",
        type=str,
        help="Location to save the resulting HDF5 file.",
        default="virgo_data.hdf5",
    )

    parser.add_argument("--seed", type=int, help="Seed for random number generator")

    args = parser.parse_args()

    start = time()
    generate_hdf5_dataset(
        output_file=args.save_location,
        num_datapoints=args.num_datapoints,
        duration=16,
        sample_rate=500,
        num_aux_channels=3,
        num_waves_range=(10, 15),
        noise_amplitude=0.5,
        datapoints_per_file=args.save_frequency,
        num_processes=args.num_processes,
        seed=args.seed,
    )
    end = time()
    total_time = end - start
    print(f"Generation took {total_time:.2f} seconds!")
