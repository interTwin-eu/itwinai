# xtclim
## ML-based extreme events detection and characterization (CERFACS)

The code is adapted from CERFACS' [repository](https://github.com/cerfacs-globc/xtclim/tree/master).
The implementation of a pipeline with itwinai framework is shown below. 

## Method 
Convolutional Variational AutoEncoder.

## Input
"3D daily images", daily screenshots of Europe for three climate variables (maximum temperature, precipitation, wind).

## Output 
Error between original and reconstructed image: postprocessed for analysis in the `scenario_season_comparison.ipynb` file.

## Idea 
The more unusual an image (anomaly), the higher error.

## Information on files

In the preprocessing folder, the `preprocess_functions_2d_ssp.py` class loads NetCDF files from a `data` folder, which has to be specified in `dataset_root` in the config file `pipeline.yaml` (please change the location). The data can be found [here](https://b2drop.eudat.eu/s/rtAadDNYDWBkxjJ). The given class normalizes and adjusts the data for the network. The function `preprocess_2d_seasons.py` splits the data into seasonal files. Preprocessed data is stored in the `input` folder.

The file `train.py` trains the network. Caution: It will overwrite the weights of the network already saved in outputs (unless you change the path name `outputs/cvae_model_3d.pth` in the script).

The `anomaly.py` file evaluates the network on the available datasets - train, test, and projection.

## How to launch pipeline

The config file `pipeline.yaml` contains all the steps to execute the workflow. You can launch it from the root of the repository with:

```bash
python train.py -p pipeline.yaml

```

## TODOs
Integration of post-processing step + distributed strategies
