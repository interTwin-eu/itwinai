# Interact with Rucio dataset files

The following script assumes that all the files within a "Rucio Dataset" (or `DIDs` - see below) are present in a RSE (Rucio Storage Element), and that this RSE is accessible locally.
 - `DIDs` (or Data Identifiers - see Rucio [documentation](https://rucio.github.io/documentation/started/concepts/file_dataset_container/)) are composed of a scope plus a dataset name in the `SCOPE:Name` format.
 - If the files are not present in the RSE, replicate the dataset on the desired RSE before running the script.

Run the following bash script

```bash
> ./rucio_dataset_files.sh <SCOPE:DataSet> <output_filename> <output_dirname>
```
where
 - `SCOPE:Name` is the Rucio DID. You can list all the scopes with the command `rucio list-scopes`, and the dataset name with `rucio list-did <SCOPE>:` (note the colon).
 - `output_filename` is the output file that contains the "filepath" of all the files in the dataset.
 - `output_dirname` is the output directory with all the dataset files (in the form of symbolic links), to avoid duplication of files on disk. It also prevents users to search within the disk, which could get complicated depending on the storage kind and model.

```bash
# Example
> ./rucio_dataset_files.sh calorimeter:training_data_hdf5 calorimeter_files.txt calorimeter_symlink_dir

# And check the output file and the directory with the symlinks
> cat calorimeter_files.txt
> ls -l calorimeter_symlink_dir
```
