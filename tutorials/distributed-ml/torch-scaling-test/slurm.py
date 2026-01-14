# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


from argparse import ArgumentParser

from itwinai.slurm.configuration import MLSlurmBuilderConfig
from itwinai.slurm.script_builder import generate_default_slurm_script


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--distributed-strategy", choices=["ddp", "horovod", "deepspeed"], default="ddp"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "runall", "scaling-test"],
        default="single",
    )
    parser.add_argument(
        "--scalability-nodes",
        default="1,2,4,8",
        help="Comma-separated node counts for the scalability test.",
    )
    parser.add_argument("--python-venv", default=".venv")
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--save-script", action="store_true")
    parser.add_argument("--submit-job", action="store_true")
    parser.add_argument(
        "--itwinai-trainer",
        action="store_true",
        help="Whether to use the itwinai trainer or not.",
    )
    args = parser.parse_args()

    if args.itwinai_trainer:
        training_command = (
            f"itwinai_trainer.py -c config/base.yaml "
            f"-c config/{args.distributed_strategy}.yaml -s {args.distributed_strategy}"
        )
    else:
        training_command = (
            f"{args.distributed_strategy}_trainer.py -c config/base.yaml "
            f"-c config/{args.distributed_strategy}.yaml"
        )
    if args.distributed_strategy == "horovod":
        training_command = "python " + training_command

    config = MLSlurmBuilderConfig(
        distributed_strategy=args.distributed_strategy,
        mode=args.mode,
        scalability_nodes=args.scalability_nodes,
        python_venv=args.python_venv,
        training_cmd=training_command,
        save_script=args.save_script,
        submit_job=args.submit_job,
        use_ray=args.use_ray,
    )
    generate_default_slurm_script(config)


if __name__ == "__main__":
    main()
