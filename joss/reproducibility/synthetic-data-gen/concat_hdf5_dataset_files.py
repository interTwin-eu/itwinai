# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
from pathlib import Path

import h5py
import numpy as np


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

    # print(f"Appending to file: '{str(file_path.resolve())}'.")
    with h5py.File(file_path, "a") as f:
        dset = f[dataset_name]
        dset.resize(dset.shape[0] + array.shape[0], axis=0)
        dset[-array.shape[0] :] = array


def main():
    parser = argparse.ArgumentParser(description="Virgo Dataset Generation")
    parser.add_argument(
        "--dir", type=str, help="Directory containing the HDF5 files to concatenate"
    )
    parser.add_argument(
        "--save-location",
        type=str,
        help="Location to save the resulting HDF5 file.",
        default="total_virgo_data.hdf5",
    )
    args = parser.parse_args()
    dir = Path(args.dir)
    save_location = Path(args.save_location)
    num_aux_channels = 3
    square_size = 64
    dataset_name = "virgo_dataset"

    # Creating empty HDF5 file
    datapoint_shape = (num_aux_channels + 1, square_size, square_size)
    save_location.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating/overwriting file: '{save_location.resolve()}'.")
    with h5py.File(save_location, "w") as f:
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

    # NOTE: This will not necessarily iterate in same order as the suffices of the
    # file names
    files = [
        entry for entry in dir.iterdir()
        if (entry.suffix == ".hdf5" and entry.stem != "virgo_data")
    ]
    for entry in files:
        with h5py.File(entry, "r") as f:
            data = f[dataset_name][:]
        print(f"Adding {len(data)} rows from entry: {entry}")

        append_to_hdf5_dataset(
            file_path=save_location,
            dataset_name=dataset_name,
            array=data,
            expected_datapoint_shape=datapoint_shape,
        )


if __name__ == "__main__":
    main()
