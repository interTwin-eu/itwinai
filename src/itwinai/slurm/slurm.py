# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from itwinai.slurm.slurm_script_builder import (
    SlurmScriptBuilder,
    SlurmScriptConfiguration,
)
from itwinai.slurm.utils import get_slurm_job_parser


def main():
    parser = get_slurm_job_parser()
    args = parser.parse_args()

    slurm_script_configuration = SlurmScriptConfiguration(
        job_name=args.job_name,
        account=args.account,
        time=args.time,
        partition=args.partition,
        std_out=args.std_out,
        err_out=args.err_out,
        num_nodes=args.num_nodes,
        num_tasks_per_node=args.num_tasks_per_node,
        gpus_per_node=args.gpus_per_node,
        cpus_per_gpu=args.cpus_per_gpu,
    )

    submit_slurm_job = not args.no_submit_slurm_job
    retain_file = not args.no_retain_file

    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_script_configuration,
        distributed_strategy=args.dist_strat,
        debug=args.debug,
        training_command=args.training_cmd,
    )

    slurm_script_builder.process_slurm_script(
        retain_file=retain_file, submit_slurm_job=submit_slurm_job
    )


if __name__ == "__main__":
    main()
