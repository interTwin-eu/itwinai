# Run all versions of distributed ML for MNIST
echo "Torch DDP training: $(sbatch ddp_slurm.sh)"
echo "DeepSpeed training: $(sbatch deepspeed_slurm.sh)"
echo "Horovod training: $(sbatch hvd_slurm.sh)"