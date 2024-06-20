# Interact with Rucio dataset files

The following script assumes that all the files within a DID are present in a RSE (Rucio Storage Element), and that this RSE is accessible locally.
 - DIDs are composed of a scope plus a dataset name in the `SCOPE:DataSet` format.
 - If the files are not present in the RSE, replicate the dataset on the desired RSE before running the script.

Run the following bash script

```bash
> ./rucio_dataset_files.sh <SCOPE:DataSet> <output_file> <output_symlink_dir>

# Example
> ./rucio_dataset_files.sh calorimeter:training_data_hdf5 calorimeter_files.txt calorimeter_symlink_dir
> cat calorimeter_files.txt
> ls -l calorimeter_symlink_dir
```
