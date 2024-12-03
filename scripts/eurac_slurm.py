# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from create_slurm import (
    get_srun_command,
    remove_indentation_from_multiline_string,
    slurm_template,
)


def get_eurac_setup_command(venv: str, cpus_per_gpu: int):
    if cpus_per_gpu > 0:
        omp_num_threads = "$SLURM_CPUS_PER_GPU"
    else:
        omp_num_threads = "1"

    hpc_modules = [
        "Stages/2024",
        "GCC",
        "OpenMPI",
        "CUDA/12",
        "MPI-settings/CUDA",
        "Python/3.11.3",
        "HDF5",
        "PnetCDF",
        "libaio",
        "mpi4py",
    ]

    setup_command = rf"""
        ml {' '.join(hpc_modules)}
        source {venv}/bin/activate
        export OMP_NUM_THREADS={omp_num_threads}
    """

    setup_command = setup_command.strip()
    return remove_indentation_from_multiline_string(setup_command)


def get_eurac_training_command(
    config_file: str, pipe_key: str, distributed_strategy: str
):
    training_command = rf"""
    $(which itwinai) exec-pipeline \
        --config {config_file} \
        --pipe-key {pipe_key} \
        -o strategy={distributed_strategy}
    """
    training_command = training_command.strip()
    return remove_indentation_from_multiline_string(training_command)


def main():
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

    distributed_strategy = "ddp"
    venv = ".venv"

    config_file = "config.yaml"
    pipe_key = "rnn_training_pipeline"

    dist_strats = ["ddp", "deepspeed", "horovod"]

    for distributed_strategy in dist_strats:
        training_command = get_eurac_training_command(
            config_file=config_file,
            pipe_key=pipe_key,
            distributed_strategy=distributed_strategy,
        )
        main_command = get_srun_command(
            distributed_strategy=distributed_strategy, training_command=training_command
        )
        setup_command = get_eurac_setup_command(venv=venv, cpus_per_gpu=cpus_per_gpu)
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
        print("#" * 20)
        print(template)
        print("#" * 20)


if __name__ == "__main__":
    main()
