"""A generated module for Itwinai functions

This module has been generated via dagger init and serves as a reference to
basic module structure as you get started with Dagger.

Two functions have been pre-created. You can modify, delete, or add to them,
as needed. They demonstrate usage of arguments and return types using simple
echo and grep commands. The functions can be called from the dagger CLI or
from one of the SDKs.

The first line in this comment block is a short description line and the
rest is a long description with more detail on the module's purpose or usage,
if appropriate. All modules should have a short description.
"""

# NOTE: it's recommended to move your code into other files in this package
# and keep __init__.py for imports only, according to Python's convention.
# The only requirement is that Dagger needs to be able to import a package
# called "main" (i.e., src/main/).
#
# For example, to import from src/main/main.py:
# >>> from .main import Itwinai as Itwinai

import dataclasses
import datetime
import random
from string import Template
from typing import Annotated, Optional, Self

import dagger
from dagger import BuildArg, Doc, dag, function, object_type

from .literals import MLFramework, Stage


def get_codename(release_info: str) -> str:
    """
    Extracts the codename (VERSION_CODENAME or os_version) from release information.

    Args:
        release_info (str): The string containing the output of /etc/*-release.

    Returns:
        str: The extracted codename (e.g., "jammy" or "bookworm").
    """
    # Create a dictionary from the release info
    release_dict = {}
    for line in release_info.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release_dict[key.strip()] = value.strip().strip('"')

    # Attempt to extract the codename
    return release_dict.get("VERSION_CODENAME", release_dict.get("os_version", "Unknown"))


@object_type
class Itwinai:
    unique_id: Annotated[
        Optional[str],
        Doc("Unique ID to identify the image from tag. Could be the git commit hash of HEAD"),
    ] = None
    container: Optional[dagger.Container] = dataclasses.field(default=None, init=False)
    full_name: Optional[str] = dataclasses.field(default=None, init=False)

    # Note: since build_container returns self, when executing only it through dagger call
    # (e.g., dagger call build-container [args]), dagger will actually execute all the
    # methods in this class in order

    @function
    async def build_container(
        self,
        context: Annotated[
            dagger.Directory,
            Doc("location of source directory"),
        ],
        dockerfile: Annotated[
            str,
            Doc("location of Dockerfile"),
        ],
        build_args: Annotated[
            Optional[str],
            Doc("Comma-separated build args"),
        ] = None,
    ) -> Self:
        """Build itwinai container image from existing Dockerfile"""
        # context = (
        #     dag.container()
        #     .with_directory("/src", context)
        #     .with_workdir("/src")
        #     .with_file("/src/additional_requirements.txt", additional_requirements)
        #     .directory("/src")
        # )
        if build_args:
            build_args = [
                BuildArg(name=arg_couple.split("=")[0], value=arg_couple.split("=")[1])
                for arg_couple in build_args.split(",")
            ]

        self.container = (
            dag.container()
            .build(
                context=context,
                dockerfile=dockerfile,
                build_args=build_args,
            )
            .with_label(
                name="org.opencontainers.image.created",
                value=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
        )

        # Get itwinai version
        itwinai_version = (
            await self.container.with_exec(
                [
                    "bash",
                    "-c",
                    'pip list | grep -w "itwinai" | head -n 1 | tr -s " " | cut -d " " -f2 ',
                ]
            ).stdout()
        ).strip()

        self.container = self.container.with_label(
            name="org.opencontainers.image.version",
            value=itwinai_version,
        )

        return self

    @function
    def terminal(self) -> dagger.Container:
        return self.container.terminal()

    @function
    async def test_local(self) -> str:
        """Test itwinai container image with pytest on non-HPC environments."""
        test_cmd = ["pytest", "-v", "-m", "not hpc and not functional", "tests"]
        return await self.container.with_exec(test_cmd).stdout()

    @function
    async def publish(
        self,
        registry: Annotated[
            str,
            Doc("The registry URL where the container will be published"),
        ] = "ghcr.io/intertwin-eu",
        name: Annotated[
            str,
            Doc("The name of the container image"),
        ] = "itwinai-dev",
        tag: Annotated[
            Optional[str],
            Doc(
                "Optional tag for the container image; defaults to random int if not provided"
            ),
        ] = None,
    ) -> str:
        """Push container to registry"""
        from datetime import datetime

        tag = tag or self.unique_id or random.randrange(10**8)
        self.full_name = f"{registry}/{name}:{tag}"
        return await (
            self.container.with_label(
                name="org.opencontainers.image.ref.name",
                value=self.full_name,
            )
            # Invalidate cache to ensure that the container is always pushed
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .publish(self.full_name)
        )

    @function
    async def test_hpc(
        self,
        kubeconfig: Annotated[
            dagger.Secret, Doc("Kubeconfig for k8s cluster with interLink's VK")
        ],
    ) -> str:
        from .k8s import create_pod_manifest, submit_job

        # Create pod manifest
        gpus_per_node = 4
        jobscript = "/app/tests/torch/slurm.vega.sh"
        pre_exec_cmd = (
            "export CONTAINER_PATH=itwinai_dist_test.sif "
            f"&& singularity pull --force $CONTAINER_PATH docker://{self.full_name} "
            f"&& singularity exec $CONTAINER_PATH cat {jobscript} > slurm.vega.sh "
            # Activate env on Vega
            "&& . /etc/bashrc "
            "&& source /ceph/hpc/software/cvmfs_env.sh "
            "&& module use /usr/share/Modules/modulefiles "
            # Env variables necessary for tests
            "&& export MNIST_PATH=/ceph/hpc/data/st2301-itwin-users/mbunino/mnist "
            "&& export NO_COLOR=1 "
            "&& export DIST_MODE=ddp "
            "&& export RUN_NAME=ddp-itwinai "
            "&& export COMMAND='pytest -v -m torch_dist /app/tests' "
            # Quick fix
            f"&& export SLURM_GPUS_PER_NODE={gpus_per_node} "
            # Launch code in SLURM job
            "&& source slurm.vega.sh "
        )
        annotations = {
            "slurm-job.vk.io/flags": (
                f"-p gpu --gres=gpu:{gpus_per_node} "
                "--ntasks-per-node=1 --nodes=1 --time=00:10:00"
            ),
            "slurm-job.vk.io/pre-exec": (
                "trap 'export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1' ERR && "
                f"(set -e && {pre_exec_cmd}) ; trap - ERR"
            ),
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = ["sleep 10 && exit $PRE_EXEC_RETURN_CODE"]
        pod_manifest = create_pod_manifest(
            annotations=annotations,
            image_path=image_path,
            cmd_args=cmd_args,
            name="ci-test-itwinai-hpc",
        )

        # Submit pod
        kubeconfig_str = await kubeconfig.plaintext()
        status = submit_job(
            kubeconfig_str=kubeconfig_str, pod_manifest=pod_manifest, verbose=False
        )

        if status not in ["Succeeded", "Completed"]:
            raise RuntimeError(f"Pod did not complete successfully! Status: {status}")

        return f"Pod finished with status: {status}"

    @function
    async def test_n_publish(
        self,
        kubeconfig: Annotated[
            dagger.Secret, Doc("Kubeconfig for k8s cluster with interLink's VK")
        ],
        stage: Annotated[
            Stage,
            Doc("Whether to push the final image to the production image name or not"),
        ] = Stage.DEV,
        tag_template: Annotated[
            Optional[str],
            Doc(
                "Custom image tag pattern. Example: "
                "'${itwinai_version}-torch${torch_version}-${os_version}'"
            ),
        ] = None,
        framework: Annotated[
            MLFramework, Doc("ML framework in container")
        ] = MLFramework.TORCH,
    ) -> None:
        """Pipeline to test container and push it, including both local
        tests and tests on HPC via interLink.

        Args:
            production (bool, optional): whether to push the final image
                to the production image name or not. Defaults to False.
        """

        if stage == stage.DEV:
            image = "itwinai-dev"
        elif stage == stage.PRODUCTION:
            image = "itwinai"
        elif stage == stage.CVMFS:
            image = "itwinai-cvmfs"
        else:
            raise ValueError(f"Unrecognized stage '{stage}'")

        # Test locally
        await self.test_local()

        # Publish to registry with random hash
        await self.publish()

        # Test on HPC with
        await self.test_hpc(kubeconfig=kubeconfig)

        # Publish to registry with final hash
        itwinai_version = (
            await self.container.with_exec(
                [
                    "bash",
                    "-c",
                    'pip list | grep -w "itwinai" | head -n 1 | tr -s " " | cut -d " " -f2 ',
                ]
            ).stdout()
        ).strip()
        os_info = (
            await self.container.with_exec(
                [
                    "bash",
                    "-c",
                    "cat /etc/*-release",
                ]
            ).stdout()
        ).strip()
        os_version = get_codename(os_info)

        if framework == MLFramework.TORCH:
            tag_template = (
                tag_template or "${itwinai_version}-torch${framework_version}-${os_version}"
            )
            framework_version = (
                await self.container.with_exec(
                    [
                        "bash",
                        "-c",
                        (
                            'pip list | grep -w "torch[^-]" | head -n 1 | tr -s " " '
                            '| cut -d " " -f2 | cut -f1,2 -d .'
                        ),
                    ]
                ).stdout()
            ).strip()
        elif framework == MLFramework.TENSORFLOW:
            tag_template = (
                tag_template or "${itwinai_version}-tf${framework_version}-${os_version}"
            )
            framework_version = (
                await self.container.with_exec(
                    [
                        "bash",
                        "-c",
                        (
                            # TODO: check this command
                            'pip list | grep -w "tensorflow[^-]" | head -n 1 | tr -s " " '
                            '| cut -d " " -f2 | cut -f1,2 -d .'
                        ),
                    ]
                ).stdout()
            ).strip()

        tag = Template(tag_template).substitute(
            itwinai_version=itwinai_version,
            framework_version=framework_version,
            os_version=os_version,
        )
        await self.publish(name=image, tag=tag)
