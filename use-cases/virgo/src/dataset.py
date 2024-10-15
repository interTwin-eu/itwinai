"""
Utils to create a synthetic dataset to train generative models with.

The section is split in two parts:
- 1) **TimeSeries Dataset** Generation of a gwpy TimeSeries dataset
    making use of random noise and sinusoidal functions
- 2) **Q-plot Dataset** Conversion of TimeSeries dataset into 2D
    Image dataset making use of q_transform

The dataset used to train the NN with is created as a 2D images. Note that
you do not need to run the two sections each time, but can rather save
the dataset after creating it once and loading it at the beginning of
Process Data that to save time.
"""


import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


def generate_dataset_aux_channels(
        rows, columns, duration=10, sample_rate=500,
        num_waves_range=(10, 15), noise_amplitude=0.1
):
    """Generate a Pandas DataFrame with randomly generated smooth sine wave time
    series with added smooth random noise.

    Parameters:
        - rows (int): Number of rows in the DataFrame.
        - columns (int): Number of columns in the DataFrame.
        - duration (float): Duration of the time series data in seconds
            (default is 6 seconds).
        - sample_rate (int): Sampling rate of the time series data in samples
            per second (default is 500 samples per second).
        - num_waves_range (tuple): Range for the random number of sine waves
            to be generated for each time series.
            Format: (min_num_waves, max_num_waves) (default is (10, 15)).
        - noise_amplitude (float): Amplitude of the smooth random noise added
            to the time series data (default is 0.1).

    Returns:
        - DataFrame: Pandas DataFrame containing the generated time series
            data.
    """
    # Generate time array
    times = np.linspace(0, duration, duration * sample_rate)

    # Initialize an empty list to store individual DataFrames for each row
    dfs = []
    # columns_list = [f'Aux_{i+1}' for i in range(columns)]

    for index in range(rows):
        df_dict = {}
        for col in range(columns):
            # Initialize an array to store the generated wave data for this row
            wave_data = np.zeros(len(times))
            # Determine the number of sine waves to generate for this column
            # randomly
            num_waves = np.random.randint(*num_waves_range)

            # Generate each sine wave
            for _ in range(num_waves):
                # Randomly generate parameters for the sine wave (amplitude,
                # frequency, phase)
                amplitude = np.random.uniform(0.5, 2.0)
                frequency = np.random.uniform(0.5, 5.0)
                phase = np.random.uniform(0, 2*np.pi)

                # Generate the sine wave and add it to the wave_data
                wave_data += amplitude * \
                    np.sin(2 * np.pi * frequency * times + phase)

            # Add smooth random noise to the wave data
            smooth_noise = np.random.normal(0, noise_amplitude, len(times))
            wave_data += smooth_noise

            # Create a TimeSeries object from the wave data
            ts = TimeSeries(wave_data, t0=0, dt=1/sample_rate)
            df_dict[f'Aux_{col+1}'] = [ts]

        # Create a DataFrame with the TimeSeries
        df_row = pd.DataFrame(df_dict)

        # Append the DataFrame to the list
        dfs.append(df_row)

    # Concatenate the list of DataFrames into a single DataFrame along
    # rows axis
    df = pd.concat(dfs, ignore_index=True, axis=0)

    return df


def generate_dataset_main_channel(input_df, weights=None, noise_amplitude=0.1):
    """Generate a dataset where each row of a single column is a weighted linear
    combination of the entries
    in the corresponding row in the input DataFrame plus random noise.

    Parameters:
        - input_df (DataFrame): Input DataFrame generated by
            ``generate_smooth_noisy_sine_wave_series``.
        - weights (array-like): Optional weights for each entry in the row.
            If None, random weights are generated (default is None).
        - noise_amplitude (float): Amplitude of the random noise added to the
            linear combination (default is 0.1).

    Returns:
        - DataFrame: Pandas DataFrame containing the generated linear
            combination dataset.
    """
    dt = input_df.iloc[0, 0].dt
    # Initialize an empty list to store the linear combination values
    linear_combination_data = []

    # Generate random weights if not provided
    if weights is None:
        # randomly generate weights in range [0.5,1.5]
        weights = np.random.rand(len(list(input_df.columns))) + 0.5
    print(weights)
    # Iterate over rows of the input DataFrame
    for index, row in input_df.iterrows():

        # Compute the weighted linear combination of the row values
        linear_combination = np.sum(row * weights)

        # Add random noise to the linear combination
        linear_combination += np.random.normal(0, noise_amplitude)

        # Append the result to the list
        linear_combination_data.append(
            [TimeSeries(linear_combination, dt=dt, t0=0)])

    # Create a DataFrame with the linear combination data
    linear_combination_df = pd.DataFrame(
        linear_combination_data, columns=['Main'])

    return linear_combination_df


def extract_peak_frequency(hq):
    """Calculates peak frequency (and relative time) of a given qplot

    Input:
    -hq (gwpy.Spectrgram) : Qtransform

    Return:
    -index_time (int) : time index in qtransform relative to peak frequency
    -index_freq (int) : frequency index in qtransorm relative to peak frequency
    """

    # Calculate peak frequency, time and energy density
    index_flat = np.argmax(hq.value)

    # Convert the flattened index to 2D index
    index_time, index_freq = np.unravel_index(index_flat, hq.shape)
    # /converting_factor_frequency
    # peak_freq = hq.frequencies.value[index_freq]
    # peak_value = np.max(hq.value)
    # peak_time = hq.times.value[index_time]

    return index_time, index_freq


def cut_image(qplot_array, index_freq, index_time, square_size=64):
    """Cut qplot as square_size X square_size 2D np.array centered at peak
    frequency and corresponding time

    Input:
    - qplot_array (np.array) : qplot relative to the whole TimeSeries
    - index_time (int) : time index in qtransform relative to peak frequency
    - index_freq (int) : frequency index in qtransorm relative to peak
        frequency
    - square_size (int) : Size in pixels of qplot image (default 64)

    Return:
    - subarray (np.array) : qplot cutted as square_size X square_size np.array

    """

    center_x = index_time
    center_y = index_freq  # Replace with the actual y-coordinate

    original_width, original_height = qplot_array.shape

    # Calculate the starting and ending indices for the subarray
    start_x = max(center_x - square_size // 2, 0)
    # print(start_x)
    end_x = min(start_x + square_size, original_width)
    # print(end_x)
    start_y = max(center_y - square_size // 2, 0)
    # print(start_y)
    end_y = min(start_y + square_size, original_height)
    # print(end_y)
    # start_y=0
    # end_y=square_size

    # Adjust indices if needed to make sure the resulting subarray is
    # (square_size X square_size)
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


def process_image(row, row_idx, channels, square_size):
    """Processes df's row to generate qplot images

    Input:
    - row (pd.Series) : row of TimeSeries Dataframe
    - row_idx (int) : index relative to row in DataFrame
    - channels (list): Name of columns of DataFrame
    - square_size (int): Size in pixels of qplot image

    Return:
    df_row (DataFrame): Row containing qplot images as 2D np.array
    """

    # res_list = []
    df_row = pd.DataFrame(columns=channels)
    for i, channel in enumerate(channels):

        qplot = row[channel].q_transform(frange=(10, 150))

        # calculate peak frequency and time indices for strain channel
        if i == 0:
            index_time, index_freq = extract_peak_frequency(qplot)

        # Convert the spectrogram to a NumPy array
        qplot_array = qplot.value
        qplot_array_cut = cut_image(
            qplot_array, index_freq, index_time, square_size)
        df_row[channel] = [qplot_array_cut]

    return df_row


def generate_cut_image_dataset(df, channels, num_processes=20, square_size=128):
    """Generates qplot dataset taking pandas df containing main+aux channels as input.
    The output is a df containing qtransforms (frequency range 10-150Hz) in the form of square_sizexsquare_size 2d np.array

     Args:
        df (DataFrame): DataFrame containing Main and Aux channels' gwpy TimeSeries (Main channel is always first).
        channels (list): Name of columns in the DataFrame.
        num_processes (int): Number of cores for multiprocess (default 20)
        square_size (int): Size in pixels of qplot image (default is 500 samples per second).

    Returns:
        DataFrame: Pandas DataFrame containing the q_transform np.array data.
    """

    args = [(df.iloc[row], row, channels, square_size) for row in range(df.shape[0])]
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use map to pass multiple arguments to process_image
        results = list(pool.starmap(process_image, args))

    df = pd.concat(results, ignore_index=True)
    return df


def show_dataset(df, size, num_plots=10):
    """Plots qtransforms for first 4 columns in given df

    Input:
    - df (DataFrame) : DataFrame containing qtransforms in the form of 2d
        np.array
    - size (int) : square size in pixels of qplots
    - num_plots (int) : number of rows of df to make the plot for (default 10)

    Return
    - nothing, it just displays plots
    """

    ch_list = list(df.columns)
    fig, axes = plt.subplots(2*num_plots, 2, figsize=(18, 12*num_plots))
    for j in range(num_plots):

        qplt_r = np.flipud(df.iloc[j, 0].T)
        qplt_aux1 = np.flipud(df.iloc[j, 1].T)
        qplt_aux2 = np.flipud(df.iloc[j, 2].T)
        qplt_aux3 = np.flipud(df.iloc[j, 3].T)

        # Create a single subfigure with two plots
        # fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot for Real
        im_r = axes[2*j, 0].imshow(
            qplt_r, aspect='auto',
            extent=[0, size, 0, size], vmin=0, vmax=25)
        axes[2*j, 0].set_title(ch_list[0])
        axes[2*j, 0].set_xlabel('Time')
        axes[2*j, 0].set_ylabel('Frequency')
        fig.colorbar(im_r, ax=axes[2*j, 0])  # Add colorbar for Real

        # Plot for aux
        im_g = axes[2*j, 1].imshow(
            qplt_aux1, aspect='auto',
            extent=[0, size, 0, size], vmin=0, vmax=25)
        axes[2*j, 1].set_title(ch_list[1])
        axes[2*j, 1].set_xlabel('Time')
        axes[2*j, 1].set_ylabel('Frequency')
        fig.colorbar(im_g, ax=axes[2*j, 1])  # Add colorbar for Generated
        #
        im_g = axes[2*j+1, 0].imshow(
            qplt_aux2, aspect='auto',
            extent=[0, size, 0, size], vmin=0, vmax=25)
        axes[2*j+1, 0].set_title(ch_list[2])
        axes[2*j+1, 0].set_xlabel('Time')
        axes[2*j+1, 0].set_ylabel('Frequency')
        fig.colorbar(im_g, ax=axes[2*j+1, 0])  # Add colorbar for Generated
        #
        im_g = axes[2*j+1, 1].imshow(
            qplt_aux3, aspect='auto',
            extent=[0, size, 0, size], vmin=0, vmax=25)
        axes[2*j+1, 1].set_title(ch_list[3])
        axes[2*j+1, 1].set_xlabel('Time')
        axes[2*j+1, 1].set_ylabel('Frequency')
        fig.colorbar(im_g, ax=axes[2*j+1, 1])  # Add colorbar for Generated

        # Get the bounding boxes of the axes including text decorations
        r = fig.canvas.get_renderer()

        def get_bbox(ax): return ax.get_tightbbox(
            r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, axes.flat)),
                          mtrans.Bbox).reshape(axes.shape)

        # Get the minimum and maximum extent, get the coordinate half-way
        # between those
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))
                        ).reshape(axes.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))
                        ).reshape(axes.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        for y in ys[1::2]:
            line = plt.Line2D(
                [0, 1], [y, y], transform=fig.transFigure, color="black")
            fig.add_artist(line)

    # plt.savefig('very high loss qplots.png')
    plt.show()


def normalize_(data, chan=4):
    """Normalizes the qplot data to the range [0,1] for NN convergence purposes

    Input:
    - data (torch.Tensor) : dataset of qtransforms
    - chan (int) : number of channels in dataset (default 4)

    Return:
    - data (torch.tensor) : normalized dataset
    """
    # Compute the maximum value for each channel across all 900 tensors
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(0)[0].max(
        1)[0]
    print("Maximum values for each channel across all tensors:",
          max_vals, max_vals.shape)
    # Divide each element by the maximum value of its channel
    data /= max_vals.view(1, chan, 1, 1)
    return data
