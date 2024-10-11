import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from ..src.dataset import generate_cut_image_dataset


def generate_pkl_dataset(
    folder_name='test_folder',
    num_files=5,
    duration=10,
    sample_rate=500,
    num_aux_channels=10,
    num_waves_range=(10, 15),
    noise_amplitude=0.1,
    num_processes=4,
    square_size=64,
    datapoints_per_file=10
):
    """ Generate a folder with num_files h5py files containing synthetic gravitational waves data.

    Args:
        folder_name (string): the path and name where the files will be stored
        num_files (int): Number of files which will be created.
        duration (float): Duration of the time series data in seconds (default is 6 seconds).
        num_aux_channels (int): Number of auxiliary channels, containing the data from the auxiliary sensors in the detector which do not go into the strain.
        sample_rate (int): Sampling rate of the time series data in samples per second (default is 500 samples per second).
        num_waves_range (tuple): Range for the random number of sine waves to be generated for each time series.
                                   Format: (min_num_waves, max_num_waves) (default is (10, 15)).
        noise_amplitude (float): Amplitude of the smooth random noise added to the time series data (default is 0.1).
        num_processes (int): Number of cores for multiprocess (default 20)
        square_size (int): Size in pixels of qplot image (default is 500 samples per second).

    Returns:
        A folder containing an arbitrary number of h5py files. Each file contains an arbitrary number of channels with synthetic data.
        The files are named foldername-file-number.h5
    """

    datapoints = []
    # Generate time array
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    for f in range(num_files):

        filename = 'file-'+str(f)+'.pkl'
        filepath = folder_name+'/'+filename

        times = np.linspace(0, duration, duration * sample_rate)

        # Initialise the main data as a list of zeros
        main_data = np.zeros(len(times))
        dictionary_aux = {}
        for i in range(num_aux_channels):
            channel_name = "Aux_"+str(i)

            # Initialize an array to store the generated wave data for this row
            wave_data = np.zeros(len(times))
            # Determine the number of sine waves to generate for this column randomly
            num_waves = np.random.randint(*num_waves_range)

         # Generate each sine wave
            for _ in range(num_waves):
                # Randomly generate parameters for the sine wave (amplitude, frequency, phase)
                amplitude = np.random.uniform(0.5, 2.0)
                frequency = np.random.uniform(0.5, 5.0)
                phase = np.random.uniform(0, 2*np.pi)

                # Generate the sine wave and add it to the wave_data
                wave_data += amplitude * np.sin(2 * np.pi * frequency * times + phase)

            # Add smooth random noise to the wave data
            smooth_noise = np.random.normal(0, noise_amplitude, len(times))
            wave_data += smooth_noise

            coeff = np.random.rand()

            main_data += coeff*wave_data

            # Create a TimeSeries object from the wave data
            ts = TimeSeries(wave_data, t0=0, dt=1/sample_rate)

            dictionary_aux[channel_name] = [ts]

        # Creating the main timeseries

        main_data += np.random.normal(0, noise_amplitude)
        ts_main = TimeSeries(main_data, dt=1/sample_rate)

        main_entry = {'Main': [ts_main]}

        dictionary = {**main_entry, **dictionary_aux}

        # turn dictionary into dataframe
        df_ts = pd.DataFrame(dictionary)

        # Convert timeseries dataset into q-plot
        df = generate_cut_image_dataset(df_ts, list(
            df_ts.columns), num_processes=num_processes, square_size=square_size)

        datapoints.append(df)
        if len(datapoints) % datapoints_per_file == 0:
            df_concat = pd.concat(datapoints)
            df_concat.to_pickle(filepath)
            datapoints = []

        # save dataframe to PICKLE file
        # df.to_pickle(filepath)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Virgo Dataset Generation'
    )
    parser.add_argument(
        '--target_folder_name',
        type=str,
        help='The folder to store the dataset.'
    )
    parser.add_argument(
        '--file_number',
        type=int,
        help='Number of files which will be created.'
    )

    args = parser.parse_args()  # Parse the command-line arguments

    # Creating the folders with the PKL files of timeseries data
    generate_pkl_dataset(folder_name=args.target_folder_name,
                         num_files=args.file_number,
                         duration=16,
                         sample_rate=500,
                         num_aux_channels=3,
                         num_waves_range=(10, 15),
                         noise_amplitude=0.5,
                         datapoints_per_file=500
                         )
