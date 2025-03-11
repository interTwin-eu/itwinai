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
    remove_indentation_from_multiline_string,
)
from itwinai.slurm.utils import get_slurm_job_parser


class VirgoSlurmScriptBuilder(SlurmScriptBuilder):
    def __init__(
        self,
        slurm_script_configuration: SlurmScriptConfiguration,
        distributed_strategy: str,
        training_command: None | str = None,
        python_venv: str = ".venv",
        debug: bool = False,
        config_name: str = "config",
        config_path: str = ".",
        pipe_key: str = "rnn_training_pipeline",
    ):
        super().__init__(
            slurm_script_configuration=slurm_script_configuration,
            distributed_strategy=distributed_strategy,
            training_command=training_command,
            python_venv=python_venv,
            debug=debug,
        )
        self.config_path = config_path
        self.config_name = config_name
        self.pipe_key = pipe_key

    def get_training_command(self):
        if self.training_command is not None:
            return self.training_command

        training_command = rf"""
        $(which itwinai) exec-pipeline \
            --config-path {self.config_path} \
            --config-name {self.config_name} \
            +pipe-key={self.pipe_key} \
            strategy={self.distributed_strategy} \
            checkpoint_path=checkpoints_ddp/epoch_{{}}.pth
        """
        training_command = training_command.strip()
        return remove_indentation_from_multiline_string(training_command)


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

    script_builder = VirgoSlurmScriptBuilder(
        slurm_script_configuration=slurm_script_configuration,
        distributed_strategy=args.dist_strat,
        training_command=args.training_cmd,
        python_venv=args.python_venv,
        debug=args.debug,
        pipe_key=args.pipe_key,
        config_name=args.config_name,
        config_path=args.config_path,
    )

    submit_job = not args.no_submit_job
    retain_file = not args.no_retain_file

    mode = args.mode
    if mode == "single":
        script_builder.process_slurm_script(
            submit_slurm_job=submit_job, retain_file=retain_file
        )
    elif mode == "runall":
        script_builder.run_slurm_script_all_strategies(
            submit_slurm_job=submit_job, retain_file=retain_file
        )
    elif mode == "scaling-test":
        script_builder.run_scaling_test(
            submit_slurm_job=submit_job,
            retain_file=retain_file,
            num_nodes_list=args.scalability_nodes,
        )
    else:
        # This shouldn't really ever happen, but checking just in case
        raise ValueError(
            f"Mode can only be 'single', 'runall' or 'scaling-test', but was '{mode}'."
        )


if __name__ == "__main__":
    main()
