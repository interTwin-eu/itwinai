def remove_indentation_from_multiline_string(multiline_string: str) -> str:
    """Removes the indentation from the start of each line in a multi-line string. The
    main purpose of this function is allowing you to define multi-line strings that
    don't touch the left margin of the editor, thus increasing readability.
    """
    return "\n".join([line.lstrip() for line in multiline_string.split("\n")])


def get_srun_command(
    distributed_strategy: str,
    training_command: str,
    torch_log_dir: str = "logs_torchrun",
) -> str:
    if distributed_strategy in ["ddp", "deepspeed"]:
        main_command = rf"""
        srun --cpu-bind=none --ntasks-per-node=1 \
        bash -c "torchrun \
        --log_dir='{torch_log_dir}' \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
        --rdzv_backend=c10d \
        --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
        {training_command}
    """
    elif distributed_strategy == "horovod":
        main_command = rf"""
	    export CUDA_VISIBLE_DEVICES="0,1,2,3"
        srun --cpu-bind=none \
            --ntasks-per-node=$SLURM_GPUS_PER_NODE \
            --cpus-per-task=$SLURM_CPUS_PER_GPU \
            --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
            {training_command}
    """
    else:
        raise ValueError(
            f"Given distributed strategy: {distributed_strategy}, is invalid."
        )

    main_command = main_command.strip()
    return remove_indentation_from_multiline_string(main_command)


def slurm_template(
    job_name: str,
    account: str,
    time: str,
    partition: str,
    std_out: str,
    err_out: str,
    num_nodes: int,
    num_tasks_per_node: int,
    gpus_per_node: int,
    cpus_per_gpu: int,
    setup_command: str,
    main_command: str,
) -> str:
    template = f"""
        #!/bin/bash
        
        # Job configuration
        #SBATCH --job-name={job_name}    
        #SBATCH --account={account}       
        #SBATCH --partition={partition}
        #SBATCH --time={time}
        
        #SBATCH --output={std_out}
        #SBATCH --error={err_out}
        
        # Resources allocation
        #SBATCH --nodes={num_nodes}
        #SBATCH --ntasks-per-node={num_tasks_per_node}
        #SBATCH --gpus-per-node={gpus_per_node}
        #SBATCH --cpus-per-gpu={cpus_per_gpu}
        #SBATCH --exclusive
        
        {setup_command}
        
        {main_command}"""
    # Removing indentation
    main_command = main_command.strip()
    return remove_indentation_from_multiline_string(template)


def main():
    # SLURM specific settings
    job_name = "my_test_job"
    account = "intertwin"
    time = "01:00:00"
    partition = "develbooster"
    std_out = "job.out"
    err_out = "job.err"
    num_nodes = 1
    num_tasks_per_node = 1
    gpus_per_node = 4
    cpus_per_gpu = 4

    # Setting up the environment
    setup_command = """
        ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py
        source .venv/bin/activate
    """
    setup_command = setup_command.strip()
    setup_command = remove_indentation_from_multiline_string(setup_command)

    # Main training command, e.g. srun
    torch_log_dir = "logs_torchrun"
    main_command = get_srun_command(
        "ddp", "python main.py", torch_log_dir=torch_log_dir
    )
    template = slurm_template(
        job_name=job_name,
        account=account,
        time=time,
        partition=partition,
        std_out=std_out,
        err_out=err_out,
        num_nodes=num_nodes,
        num_tasks_per_node=num_tasks_per_node,
        gpus_per_node=gpus_per_node,
        cpus_per_gpu=cpus_per_gpu,
        setup_command=setup_command,
        main_command=main_command,
    )
    print(template)


if __name__ == "__main__":
    main()
