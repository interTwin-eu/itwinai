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
    remove_indentation_from_multiline_string,
)
from itwinai.slurm.utils import get_slurm_script_parser


class EuracSlurmScriptBuilder(SlurmScriptBuilder):

    def __init__(
        self,
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
        distributed_strategy: str,
        python_venv: str = ".venv",
        debug: bool = False,
        config_file: str = "config.yaml",
        pipe_key: str = "rnn_training_pipeline",
    ):
        super().__init__(
            job_name,
            account,
            time,
            partition,
            std_out,
            err_out,
            num_nodes,
            num_tasks_per_node,
            gpus_per_node,
            cpus_per_gpu,
            distributed_strategy,
            python_venv,
            debug,
        )
        self.config_file = config_file
        self.pipe_key = pipe_key

    def get_training_command(self):
        training_command = rf"""
        $(which itwinai) exec-pipeline \
            --config {self.config_file} \
            --pipe-key {self.pipe_key} \
            -o strategy={self.distributed_strategy}
        """
        training_command = training_command.strip()
        return remove_indentation_from_multiline_string(training_command)


def main():
    # TODO: Update parser to not contain EURAC stuff and then change it here instead
    parser = get_slurm_script_parser()
    default_config_file = "config.yaml"
    default_pipe_key = "rnn_training_pipeline"
    parser.add_argument(
        "--config-file",
        type=str,
        default=default_config_file,
        help="Which config file to use for training.",
    )
    parser.add_argument(
        "--pipe-key",
        type=str,
        default=default_pipe_key,
        help="Which pipe key to use for running the pipeline.",
    )
    args = parser.parse_args()

    script_builder = EuracSlurmScriptBuilder(
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
        distributed_strategy=args.dist_strat,
        debug=args.debug,
        pipe_key=args.pipe_key,
        config_file=args.config_file,
    )

    mode = args.mode
    if mode == "single":
        script_builder.process_slurm_script()
    elif mode == "runall":
        script_builder.run_slurm_script_all_strategies()
    elif mode == "scaling-test":
        script_builder.run_scaling_test()
    else:
        # This shouldn't really ever happen, but checking just in case
        raise ValueError(
            f"Mode can only be 'single', 'runall' or 'scaling-test', but was '{mode}'."
        )


if __name__ == "__main__":
    main()
