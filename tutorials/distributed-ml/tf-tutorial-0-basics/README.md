# Tutorial: distributed strategies for Tensorflow

In this tutorial we show how to use Tensorflow `MultiWorkerMirroredStrategy`.
Note that the environment is tested on the HDFML system at JSC.
For other systems, the module versions might need change accordingly.

Importantly, it should be kept it mind that under distributed training with Tensorflow,
the batch size should be scaled with the number of workers in this way:

```bash
# scale batch size with number of workers (or replicas)
batch_size = args.batch_size * num_replicas
```

The other point to consider is the specification of epochs in Tensorflow. 
An example code snipped for training an ML model in Tensorflow is shown below.

```bash
# trains the model
model.fit(dist_train, 
          epochs=args.epochs,
          steps_per_epoch=len(x_train)//batch_size)
```

Here `steps_per_epoch` specifies the number of iterations of `train_dataset`
over the specified number of `epochs`, where `train_dataset`
is the TF dataset class that is employed for training the model.
If `train_dataset` is too small, then the amount of data could be too less
for the specified number of `epochs`. This can be mitigated with the `repeat()` 
option, which ensures that there is sufficient data for training.

```bash
train_dataset = train_dataset.batch(batch_size).repeat()
```

After consideration of these aspects, from the root of this repository, 
build the environment containing Tensorflow. You can *try* with:

```bash
# Creates a Python venv called envAItf_hdfml
make tf-gpu-jsc
```

If you want to distribute the code in `train.py`, run from terminal:

```bash
sbatch tfmirrored_slurm.sh
```
