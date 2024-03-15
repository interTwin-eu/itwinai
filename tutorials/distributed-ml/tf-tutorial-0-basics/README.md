# Tutorial: distributed strategies for Tensorflow

In this tutorial we show how to use Tensorflow `MultiWorkerMirroredStrategy`. Note that the environment is tested on the HDFML system at JSC. For other systems, the module versions might need change accordingly. Other strategies will be updated here.

First, from the root of this repo, build the environment containing
Tensorflow. You can *try* with:

```bash
# Creates a Python venv called envAItf_hdfml
make tf-gpu-jsc
```
Please note this is still to be added to the make file in the root. Contact RS in the meantime for environment source file.

If you want to distribute the code in `train.py`, run from terminal:

```bash
sbatch tfmirrored_slurm.sh
```