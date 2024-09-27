import pandas as pd
import sys
import torch
import numpy as np

from tqdm.notebook import tqdm
import h5py as h5
import multiprocessing
from tqdm import tqdm
# import time
from pathlib import Path

import gwpy
from gwpy.timeseries import TimeSeries

if torch.cuda.is_available():
    device = 'cuda'
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = 'cpu'
    

def cut_image(qplot_array,index_freq,index_time,square_size=64):
    """
    Cut qplot as square_size X square_size 2D np.array centered at peak frequency and corresponding time

    Input:
    - qplot_array (np.array) : qplot relative to the whole TimeSeries
    - index_time (int) : time index in qtransform relative to peak frequency
    - index_freq (int) : frequency index in qtransorm relative to peak frequency
    - square_size (int) : Size in pixels of qplot image (default 64)

    Return:
    - subarray (np.array) : qplot cutted as square_size X square_size np.array

    """

    center_x = index_time
    center_y = index_freq  # Replace with the actual y-coordinate


    original_width, original_height = qplot_array.shape

    # Calculate the starting and ending indices for the subarray
    start_x = max(center_x - square_size // 2, 0)
    #print(start_x)
    end_x = min(start_x + square_size, original_width)
    #print(end_x)
    start_y = max(center_y - square_size // 2, 0)
    # print(start_y)
    end_y = min(start_y + square_size, original_height)
    # print(end_y)
    # start_y=0
    # end_y=square_size


    # Adjust indices if needed to make sure the resulting subarray is (square_size X square_size)
    if end_x - start_x < square_size:
        diff_x = square_size - (end_x - start_x)
        if end_x < original_width:
            end_x += diff_x
        else:
            start_x -= diff_x
    if end_y - start_y < square_size:
        diff_y = square_size - (end_y - start_y)
        if end_y < original_height:
            end_y += diff_y
        else:
            start_y -= diff_y

    subarray = qplot_array[start_x:end_x, start_y:end_y]
    # print(subarray.shape)
    # print(type(subarray))

    return subarray



def extract_peak_frequency(hq):
    """
    Calculates peak frequency (and relative time) of a given qplot

    Input:
    -hq (gwpy.Spectrgram) : Qtransform

    Return:
    -index_time (int) : time index in qtransform relative to peak frequency
    -index_freq (int) : frequency index in qtransorm relative to peak frequency
    """

    #Calculate peak frequency, time and energy density
    index_flat = np.argmax(hq.value)

    # Convert the flattened index to 2D index
    index_time, index_freq = np.unravel_index(index_flat, hq.shape)
    peak_freq = hq.frequencies.value[index_freq]#/converting_factor_frequency
    peak_value=np.max(hq.value)
    peak_time = hq.times.value[index_time]

    return index_time,index_freq


def process_image(row,row_idx,channels,square_size):
    """
    Processes df's row to generate qplot images

    Input:
    - row (pd.Series) : row of TimeSeries Dataframe
    - row_idx (int) : index relative to row in DataFrame
    - channels (list): Name of columns of DataFrame
    - square_size (int): Size in pixels of qplot image

    Return:
    df_row (DataFrame): Row containing qplot images as 2D np.array
    """


    res_list=[]
    df_row=pd.DataFrame(columns=channels)
    for i,channel in enumerate(channels):

        qplot =row[channel].q_transform(frange=(10, 150))

        #calculate peak frequency and time indices for main channel
        if i==0:
            index_time,index_freq= extract_peak_frequency(qplot)

        # Convert the spectrogram to a NumPy array
        qplot_array = qplot.value
        qplot_array_cut= cut_image(qplot_array,index_freq,index_time,square_size)
        df_row[channel]=[qplot_array_cut]

    return df_row



def generate_cut_image_dataset(df,channels,num_processes=20,square_size=128):
    """
    Generates qplot dataset taking pandas df containing main+aux channels as input.
    The output is a df containing qtransforms (frequency range 10-150Hz) in the form of square_sizexsquare_size 2d np.array


     Parameters:
        - df (DataFrame): DataFrame containing Main and Aux channels' gwpy TimeSeries (Main channel is always first).
        - channels (list): Name of columns in the DataFrame.
        - num_processes (int): Number of cores for multiprocess (default 20)
        - square_size (int): Size in pixels of qplot image (default is 500 samples per second).

    Returns:
        - DataFrame: Pandas DataFrame containing the q_transform np.array data.
    """

    # print(channels)
    args = [(df.iloc[row],row,channels,square_size) for row in range(df.shape[0])]
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use map to pass multiple arguments to process_image
        results = list(pool.starmap(process_image, args))

    df = pd.concat(results, ignore_index=True)
    return df

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

    """
    Generate a folder with num_files h5py files containing synthetic gravitational waves data.

    Parameters:
        - folder_name (string): the path and name where the files will be stored
        - num_files (int): Number of files which will be created.
        - duration (float): Duration of the time series data in seconds (default is 6 seconds).
        - num_aux_channels (int): Number of auxiliary channels, containing the data from the auxiliary sensors in the detector which do not go into the strain.
        - sample_rate (int): Sampling rate of the time series data in samples per second (default is 500 samples per second).
        - num_waves_range (tuple): Range for the random number of sine waves to be generated for each time series.
                                   Format: (min_num_waves, max_num_waves) (default is (10, 15)).
        - noise_amplitude (float): Amplitude of the smooth random noise added to the time series data (default is 0.1).
        - num_processes (int): Number of cores for multiprocess (default 20)
        - square_size (int): Size in pixels of qplot image (default is 500 samples per second).

    Returns:
        -- A folder containing an arbitrary number of h5py files. Each file contains an arbitrary number of channels with synthetic data.
            The files are named foldername-file-number.h5
    """

    datapoints = []
    # Generate time array
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    for f in range(num_files):


        filename='file-'+str(f)+'.pkl'
        filepath=folder_name+'/'+filename

        times = np.linspace(0, duration, duration * sample_rate)

        #Initialise the main data as a list of zeros
        main_data = np.zeros(len(times))
        dictionary_aux={}
        for i in range(num_aux_channels):
            channel_name="Aux_"+str(i)

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

            coeff=np.random.rand()

            main_data+=coeff*wave_data

            # Create a TimeSeries object from the wave data
            ts = TimeSeries(wave_data, t0=0, dt=1/sample_rate)

            dictionary_aux[channel_name] = [ts]


        #Creating the main timeseries

        main_data += np.random.normal(0, noise_amplitude)
        ts_main= TimeSeries(main_data, dt=1/sample_rate)

        main_entry={'Main':[ts_main]}

        dictionary= {**main_entry, **dictionary_aux}

        #turn dictionary into dataframe
        df_ts = pd.DataFrame(dictionary)

        # Convert timeseries dataset into q-plot
        df = generate_cut_image_dataset(df_ts,list(df_ts.columns),num_processes=num_processes,square_size=square_size)
       
        datapoints.append(df)
        if len(datapoints) % datapoints_per_file == 0:
            df_concat = pd.concat(datapoints)
            df_concat.to_pickle(filepath)
            datapoints = []       

        #save dataframe to PICKLE file
        # df.to_pickle(filepath)


if __name__ == "__main__":
    desired_folder_name = sys.argv[1]
    file_number = int(sys.argv[2])
    # Creating the folders with the PKL files of timeseries data
    generate_pkl_dataset(folder_name=desired_folder_name,
                         num_files=file_number,
                         duration=16,
                         sample_rate=500,
                         num_aux_channels=3,
                         num_waves_range=(10, 15),
                         noise_amplitude=0.5,
                         datapoints_per_file=500
                         )
