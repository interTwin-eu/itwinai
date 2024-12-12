# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import List
from pathlib import Path
from itwinai.slurm.slurm_script_builder import (
    SlurmScriptBuilder,
    remove_indentation_from_multiline_string,
)
from itwinai.slurm.utils import get_slurm_script_parser


class TutorialSlurmScriptBuilder(SlurmScriptBuilder):

    def __init__(
        self,
        account: str,
        time: str,
        partition: str,
        num_nodes: int,
        num_tasks_per_node: int,
        gpus_per_node: int,
        cpus_per_gpu: int,
        distributed_strategy: str,
        job_name: str | None = None,
        std_out: str | None = None,
        err_out: str | None = None,
        python_venv: str = ".venv",
        debug: bool = False,
        training_command: str | None = None,
    ):
        super().__init__(
            account=account,
            time=time,
            partition=partition,
            num_nodes=num_nodes,
            num_tasks_per_node=num_tasks_per_node,
            gpus_per_node=gpus_per_node,
            cpus_per_gpu=cpus_per_gpu,
            job_name=job_name,
            std_out=std_out,
            err_out=err_out,
            python_venv=python_venv,
            distributed_strategy=distributed_strategy,
            debug=debug,
        )
        self.training_command = training_command

    def get_training_command(self):
        if self.training_command is None:
            raise ValueError("self.training_command cannot be None!")
        training_command = self.training_command.strip()
        return remove_indentation_from_multiline_string(training_command)

    def run_slurm_script_all_strategies(
        self,
        setup_command: str | None = None,
        main_command: str | None = None,
        file_folder: Path = Path("slurm_scripts"),
        retain_file: bool = True,
        run_script: bool = True,
        strategies: List[str] = ["ddp", "horovod", "deepspeed"],
    ):
        strategies = ["ddp", "deepspeed", "horovod"]
        trainer_commands = {
            "itwinai": "itwinai_trainer.py -c config/base.yaml -c config/{0}.yaml -s {0}",
            "baseline": "{0}_trainer.py -c config/base.yaml -c config/{0}.yaml",
        }

        for trainer_type, command_template in trainer_commands.items():
            for strategy in strategies:
                # Insert the strategy into the command template
                self.training_command = command_template.format(strategy)
                self.distributed_strategy = strategy

                file_name = (
                    f"{trainer_type}-{self.distributed_strategy}"
                    f"-{self.num_nodes}x{self.gpus_per_node}.sh"
                )
                file_path = file_folder / file_name
                self.process_slurm_script(
                    setup_command=None,
                    main_command=None,
                    retain_file=retain_file,
                    run_script=run_script,
                    file_path=file_path,
                )


def main():
    parser = get_slurm_script_parser()
    # Customizing the parser to this specific use case before retrieving the args
    parser.add_argument(
        "--itwinai-trainer",
        action="store_true",
        help="Whether to use the itwinai trainer or not.",
    )
    args = parser.parse_args()

    retain_file = not args.no_retain_file
    run_script = not args.no_run_script

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

    script_builder = TutorialSlurmScriptBuilder(
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
        training_command=training_command,
    )

    mode = args.mode
    if mode == "single":
        script_builder.process_slurm_script(retain_file=retain_file, run_script=run_script)
    elif mode == "runall":
        script_builder.run_slurm_script_all_strategies(retain_file=retain_file, run_script=run_script)
    elif mode == "scaling-test":
        script_builder.run_scaling_test(retain_file=retain_file, run_script=run_script)
    else:
        # This shouldn't really ever happen, but checking just in case
        raise ValueError(
            f"Mode can only be 'single', 'runall' or 'scaling-test', but was '{mode}'."
        )


if __name__ == "__main__":
    main()
