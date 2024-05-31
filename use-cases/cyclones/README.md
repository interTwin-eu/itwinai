# Tropical cyclone detection

The code is adapted from the CMCC use case's
[repository](https://github.com/CMCC-Foundation/ml-tropical-cyclones-detection).

## Setup env

```bash
# After activating the environment
pip install -r requirements.txt
```

## Dataset

If the automatic download from python does not work, try from the command line from
within the virtual environment:

```bash
gdown https://drive.google.com/drive/folders/1TnmujO4T-8_j4bCxqNe5HEw9njJIIBQD -O data/tmp_data/trainval --folder
```

For more info visit the [gdown](https://github.com/wkentaro/gdown) repository.

## Training

Launch training:

```bash
# # ONLY IF tensorflow>=2.16
# export TF_USE_LEGACY_KERAS=1

source ../../.venv-tf/bin/activate
python train.py -p pipeline.yaml 
```

On JSC, the dataset is pre-downloaded and you can use the following command:

```bash
# # ONLY IF tensorflow>=2.16
# export TF_USE_LEGACY_KERAS=1

source ../../envAItf_hdfml/bin/activate
python train.py -p pipeline.yaml --data_path /p/project/intertwin/smalldata/cmcc

# Launch a job with SLURM
sbatch startscript.sh
```
