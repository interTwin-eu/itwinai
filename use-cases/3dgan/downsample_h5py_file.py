# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Downsample H5 files to a more manageable size."""

import h5py

IN_FILENAME = "large_file.h5"
OUT_FILENAME = "sample.h5"
MAXITEMS = 100

with h5py.File(IN_FILENAME, "r") as input_file:
    with h5py.File(OUT_FILENAME, "w") as outfile:
        for key in input_file.keys():
            print(input_file[key])
            shape = list(input_file[key].shape)
            shape[0] = MAXITEMS
            outfile.create_dataset_like(name=key, other=input_file[key], shape=tuple(shape))
            print(outfile[key])
            outfile[key][...] = input_file[key][:MAXITEMS]

        print("verify similarities")
        print(input_file["energy"][:10])
        print(outfile["energy"][:10])
