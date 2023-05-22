"""
Entry point of any workflow.
This script gathers the info from the workflow definition file
and runs a workflow step by step, passing the info loaded from
the workflow definition file.

Example:

>>> python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

"""

import os
import subprocess
import argparse
from typing import Dict
import yaml
from omegaconf import OmegaConf


def load_yaml(path: str) -> Dict:
    """Load YAML file as dict.

    Args:
        path (str): path to YAML file.

    Raises:
        exc: yaml.YAMLError for loading/parsing errors.

    Returns:
        Dict: nested dict representation of parsed YAML file.
    """
    with open(path, "r", encoding="utf-8") as yaml_file:
        try:
            loaded_config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return loaded_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simple DT workflow.')
    parser.add_argument(
        '-f', '--workflow-file',
        type=str,
        help='Path to file describing DT a workflow.',
        required=True
    )
    parser.add_argument(
        '--cwl',
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    # 1. Parse workflow file

    # workflow definition file
    workflow = load_yaml(args.workflow_file)

    deps = []
    for dependency in workflow['conf-dependencies']:
        deps.append(load_yaml(dependency))

    workflow = OmegaConf.merge(workflow, *deps)

    # 2. Deploy steps (i.e., create conda envs), if not already there

    for step in workflow.get('steps'):
        step_name, step_data = next(iter(step.items()))
        if not os.path.exists(step_data['env']['prefix']):
            print(f'Deploying "{step_name}" step...')
            # Install python env from conda env definition file
            subprocess.run(
                (f"mamba env create -p {step_data['env']['prefix']} "
                 f"--file {step_data['env']['file']}").split(),
                check=True
            )
            # Install local python project from source, if present
            if step_data['env'].get('source') is not None:
                subprocess.run(
                    (f"conda run -p {step_data['env']['prefix']} "
                     "python -m pip install --no-deps "
                     f"-e {step_data['env']['source']}").split(),
                    check=True
                )

    # 3. Run the by invoking the CWL execution tool'
        # invoke workflow with CWL
    if args.cwl:
        print('Invoked workflow with CWL.')
        raise NotImplementedError('CWL workflow definition need to be updated')
        # print(
        #     (f"cwltool {workflow.get('workflowFileCWL')} "
        #      f"{args.workflow_file}")
        # )
        # subprocess.run(
        #     (f"cwltool {workflow.get('workflowFileCWL')} "
        #      f"{args.workflow_file}"),
        #     shell=True,
        #     check=True,
        # )

    # invoke workflow step-by-step with 'conda run ...'
    else:
        for step in workflow.get('steps'):
            step_name, step_data = next(iter(step.items()))
            # Args
            args_str = ''
            if step_data['args'] is not None:
                args_str = " ".join([
                    f"--{k} {v}" for k, v in step_data['args'].items()
                ])

            print(f'Running "{step_name}" step...')
            print(
                f"conda run -p {step_data['env']['prefix']} "
                f"{step_data['command']} {args_str} \n\n"
            )
            subprocess.run(
                (f"conda run -p {step_data['env']['prefix']} "
                 f"{step_data['command']} {args_str} ").split(),
                check=True,
            )
