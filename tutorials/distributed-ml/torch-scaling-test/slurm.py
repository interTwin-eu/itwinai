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


class TutorialSlurmScriptBuilder(SlurmScriptBuilder):
    def __init__(
        self,
        slurm_script_configuration: SlurmScriptConfiguration,
        distributed_strategy: str,
        training_command: str | None = None,
        python_venv: str = ".venv",
        debug: bool = False,
        use_itwinai_trainer: bool = False,
    ):
        super().__init__(
            slurm_script_configuration=slurm_script_configuration,
            distributed_strategy=distributed_strategy,
            training_command=training_command,
            python_venv=python_venv,
            debug=debug,
        )
        self.use_itwinai_trainer = use_itwinai_trainer

    def generate_identifier(self) -> str:
        if self.use_itwinai_trainer:
            prepend = "itwinai-"
        else:
            prepend = "baseline-"
        return prepend + super().generate_identifier()

    def get_training_command(self):
        if self.training_command is not None:
            return self.training_command

        if self.use_itwinai_trainer:
            training_command = (
                "itwinai_trainer.py -c config/base.yaml "
                f"-c config/{self.distributed_strategy}.yaml -s {self.distributed_strategy}"
            )
        else:
            training_command = (
                f"{self.distributed_strategy}_trainer.py -c config/base.yaml "
                f"-c config/{self.distributed_strategy}.yaml"
            )

        if self.distributed_strategy == "horovod":
            training_command = "python " + training_command
        return training_command


def main():
    parser = get_slurm_job_parser()
    # Customizing the parser to this specific use case before retrieving the args
    parser.add_argument(
        "--itwinai-trainer",
        action="store_true",
        help="Whether to use the itwinai trainer or not.",
    )
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
        cpus_per_task=args.cpus_per_task,
    )

    save_script = not args.no_save_script
    submit_job = not args.no_submit_job

    # Setting the training command for the single run
    if args.itwinai_trainer:
        training_command = (
            f"itwinai_trainer.py -c config/base.yaml "
            f"-c config/{args.dist_strat}.yaml -s {args.dist_strat}"
        )
    else:
        training_command = (
            f"{args.dist_strat}_trainer.py -c config/base.yaml "
            f"-c config/{args.dist_strat}.yaml"
        )
    if args.dist_strat == "horovod":
        training_command = "python " + training_command

    # Building the script
    script_builder = TutorialSlurmScriptBuilder(
        slurm_script_configuration=slurm_script_configuration,
        distributed_strategy=args.dist_strat,
        debug=args.debug,
        training_command=training_command,
        python_venv=args.python_venv,
        use_itwinai_trainer=args.itwinai_trainer,
    )

    # Processing the script depending on the given mode
    mode = args.mode
    if mode == "single":
        script_builder.process_slurm_script(
            save_script=save_script, submit_slurm_job=submit_job
        )
    elif mode == "runall":
        # Running all strategies with and without the itwinai trainer
        script_builder.training_command = None
        script_builder.use_itwinai_trainer = False
        script_builder.run_slurm_script_all_strategies(
            save_script=save_script, submit_slurm_job=submit_job
        )

        # We do this twice as there are two types of strategies
        script_builder.use_itwinai_trainer = True
        script_builder.run_slurm_script_all_strategies(
            save_script=save_script, submit_slurm_job=submit_job
        )
    elif mode == "scaling-test":
        # Running the scaling test with and without the itwinai trainer
        script_builder.training_command = None
        script_builder.use_itwinai_trainer = False
        script_builder.run_scaling_test(
            save_script=save_script,
            submit_slurm_job=submit_job,
            num_nodes_list=args.scalability_nodes,
        )

        script_builder.use_itwinai_trainer = True
        script_builder.run_scaling_test(
            save_script=save_script,
            submit_slurm_job=submit_job,
            num_nodes_list=args.scalability_nodes,
        )
    else:
        # This shouldn't really ever happen, but checking just in case
        raise ValueError(
            f"Mode can only be 'single', 'runall' or 'scaling-test', but was '{mode}'."
        )


if __name__ == "__main__":
    main()
