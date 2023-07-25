# Machine Learning Tropical Cyclones Detection

## Credit
- Gabriele Accarino
- Davide Donno
- Francesco Immorlano
- Donatello Elia
- Giovanni Aloisio

## Overview
The repository provides a Machine Learning (ML) library to setup training and validation of a Tropical Cyclones (TCs) Detection model. ERA5 reanalysis and the International Best Track Archive for Climate Stewardship (IBTrACS) data are used as input and the target, respectively. Input-Output data pairs are provided as TFRecords.

Input drivers:
- 10m wind gust since previous post-processing [ms**{-1}]
- mean sea level pressure [Pa]
- temperature at 300 mb [K]
- temperature at 500 mb [K]

Target:
- TC center row-column coordinates within the 40 x 40 pixels patch 

## Code Structure

The _trainval.py_ script allows running both training and validation on input data. The _data_ folder should be located at the same level of _lib_ and _scripts_ folders. 

Here is an example of training with a batch size of 512 over 3 epochs.

```bash
cd scripts
python -u trainval.py --batch_size 512 --epochs 3
```
The _trainval.py_ script takes advantage of the Command Line Interface (CLI) to get additional arguments that are useful for both training and validation of the model.

A complete list of the available arguments is provided in the following:

The CLI arguments taken in input by trainval.py script are:
- -bs, --batch_size : Global batch size of data.
- -e, --epochs : Number of epochs through which the model must be trained.
- -rn, --run_name [Optional | Default: 'debug'] : Name to be assigned to the trained model. 
- -tm, --trained_model [Optional | Default: None]: The filepath to a trained model to be loaded (ONLY if we want to continue a training).
- -ks, --kernel_size [Optional | Default: None] : Kernel size (only for Model V5 architecture). Possible values: 3,5,7,8,11,13. 
- -s, --shuffle [Optional | Default: 'False'] : Whether to shuffle dataset TFRecords filenames.
- -a, --augmentation [Optional | Default: None] : Whether or not to perform data augmentation.
- -c, --cores [Optional | Default: None] : Number of cores (for multicore CPUs. NOT designed for GPUs).
- -sb, --shuffle_buffer [Optional | Default: None] :  Number of consecutive samples to be shuffled.
- -lr, --learning_rate [Optional | Default: 0.0001] : Learning rate at which the model is trained.
- -ts, --target_scale [Optional | Default: 'False'] : Whether or not to scale the target.
- -l, --loss [Optional | Defualt: 'mae'] : Loss function to be applied. Possible values: mae, mse.
- -n, --network [Optional | Default: 'vgg_v1'] : Neural network used to train the model. Possible values: vgg_v1, vgg_v2, vgg_v3, model_v5.
- -ac, --activation [Optional | Default: 'linear'] : Last layer activation function.
- -at, --aug_type [Optional | Default: 'only_tcs'] : Type of augmentation. Possible values : only_tcs, all_patches.
- -pt, --patch_type [Optional | Default: 'nearest'] : Type of patches used during training. Possible values: alladjacent, nearest.
- -lc, --label_no_cyclone [Optional | Default: None] : The coordinate value assigned to indicate cyclone absence.
- -rg, --regularization_strength [Optional | Default: 'none'] : Regularization strength. Possible values : weak, medium, strong, very_strong, none.

## Python3 Environment 
The code has been tested on Python 3.8.16 with the following dependencies:
- keras=2.12.0
- numpy=1.23.5
- pandas=1.5.3
- psutil=5.9.5
- scikit-learn=1.2.2
- tensorflow-macos=2.12.0
