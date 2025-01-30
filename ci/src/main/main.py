# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import dataclasses
import datetime
import random
from string import Template
from typing import Annotated, Self

import dagger
import yaml
from dagger import BuildArg, Doc, Ignore, dag, function, object_type

from .hpc import Singularity
from .interlink import InterLink
from .k8s import create_pod_manifest
from .literals import MLFramework, Stage
from .utils import get_codename

ignore_list = [
    "**",
    "!src/itwinai/**",
    "!tests/**",
    "!env-files/**",
    "**/__pycache__",
    "!pyproject.toml",
]


@object_type
class Itwinai:
    """CI/CD for container images of itwinai."""

    name: Annotated[
        str | None,
        Doc(
            "Unique name to identify the image from tag. Could be the git commit hash of HEAD"
        ),
    ] = None
    _container: Annotated[dagger.Container | None, Doc("Container instance")] = (
        dataclasses.field(default=None, init=False)
    )
    full_name: Annotated[
        str | None,
        Doc("Full image name. Example: ghcr.io/intertwin-eu/itwinai-dev:0.2.3-torch2.4-jammy"),
    ] = dataclasses.field(default=None, init=False)
    _unique_id: str | None = dataclasses.field(default=None, init=False)
    sif: Annotated[dagger.File | None, Doc("SIF file")] = dataclasses.field(
        default=None, init=False
    )

    _logs: list[str] = dataclasses.field(default_factory=list, init=False)

    @property
    def unique_id(self) -> str:
        """Unique ID of the container. Falls back to random integer."""
        if self._unique_id is None:
            self._unique_id = self.name or str(random.randrange(10**8))
        return self._unique_id

    @function
    def logs(self) -> str:
        """Print the logs generted so far. Useful to terminate a chain of steps
        returning Self, preventing lazy execution of it.
        """
        if not len(self._logs):
            return "There are no logs to show"
        return "\n\n".join(self._logs)

    @function
    async def build_container(
        self,
        context: Annotated[
            dagger.Directory,
            Ignore(ignore_list),
            Doc("location of source directory"),
        ],
        dockerfile: Annotated[
            str,
            Doc("location of Dockerfile"),
        ],
        build_args: Annotated[
            list[str] | None,
            Doc("Comma-separated build args"),
        ] = None,
    ) -> Self:
        """Build itwinai container image from existing Dockerfile"""
        if build_args:
            build_args = [
                BuildArg(name=arg_couple.split("=")[0], value=arg_couple.split("=")[1])
                for arg_couple in build_args
            ]

        self._container = (
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
            await self._container.with_exec(
                [
                    "bash",
                    "-c",
                    'pip list | grep -w "itwinai" | head -n 1 | tr -s " " | cut -d " " -f2 ',
                ]
            ).stdout()
        ).strip()
        self._logs.append(f"INFO: Built container for itwinai v{itwinai_version}")

        self._container = self._container.with_label(
            name="org.opencontainers.image.version",
            value=itwinai_version,
        )

        return self

    @function
    async def debug_ignore(
        self,
        source: Annotated[
            dagger.Directory,
            Ignore(ignore_list),
        ],
    ) -> list[str]:
        """List all files after filtering by Dagger ignore."""
        return await source.glob(pattern="**/*")

    @function
    def container(self) -> dagger.Container:
        """Get the container previously built.
        Raises error if the container does not exist.
        """
        if not self._container:
            raise RuntimeError(
                "Container not found! You need to chain this function "
                "to another that created an itwinai container"
            )
        return self._container

    @function
    def with_container(
        self,
        address: Annotated[
            str,
            Doc("Container image to use as itwinai container"),
        ],
    ) -> Self:
        """Returns itself with an eisting itwinai container.
        Useful to skip the container build if a container already exists in some registry.
        """
        self._logs.append(f"INFO: Loading itwinai container from {address}")
        self._container = dag.container().from_(address=address)
        return self

    @function
    async def test_local(self) -> Self:
        """Test itwinai container image with pytest on non-HPC environments."""
        tests_result = await (
            self.container()
            .with_exec(
                [
                    "pytest",
                    "-v",
                    "-m",
                    "not hpc and not functional and not tensorflow",
                    "/app/tests",
                ]
            )
            .stdout()
        )
        self._logs.append("INFO: Results of local tests:")
        self._logs.append(tests_result)
        return self

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
            str | None,
            Doc(
                "Optional tag for the container image; defaults to random int if not provided"
            ),
        ] = None,
    ) -> Self:
        """Push container to registry"""

        tag = tag or self.unique_id
        self.full_name = f"{registry}/{name}:{tag}"
        outcome = await (
            self.container()
            .with_label(
                name="org.opencontainers.image.ref.name",
                value=self.full_name,
            )
            # Invalidate cache to ensure that the container is always pushed
            .with_env_variable(
                "CACHE", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            )
            .publish(self.full_name)
        )
        self._logs.append(f"INFO: results of publishing to {self.full_name}")
        self._logs.append(outcome)
        return self

    @function
    def interlink(
        self,
        values: Annotated[dagger.Secret, Doc("interLink installer configuration")],
        name: Annotated[
            str, Doc("Name of the k3s cluster in which the interLink VK is deployed")
        ] = "interlink-cluster",
        wait: Annotated[
            int,
            Doc(
                "Sleep time (seconds) needed to wait for the VK to appear in the k3s cluster."
            ),
        ] = 60,
        vk_name: Annotated[
            str | None,
            Doc("Name of the interLink VK. Automatically detected from values if not given"),
        ] = None,
        kubernetes: Annotated[
            dagger.Service | None,
            Doc(
                "Endpoint to exsisting k3s service to attach to. "
                "Example (from the CLI): tcp://localhost:1997"
            ),
        ] = None,
    ) -> InterLink:
        """Get interLink service."""
        return InterLink(
            values=values, name=name, wait=wait, vk_name=vk_name, kubernetes=kubernetes
        )

    @function
    async def test_hpc(
        self,
        values: Annotated[dagger.Secret, Doc("interLink installer configuration")],
        name: Annotated[
            str, Doc("Name of the k3s cluster in which the interLink VK is deployed")
        ] = "interlink-cluster",
        wait: Annotated[
            int,
            Doc(
                "Sleep time (seconds) needed to wait for the VK to appear in the k3s cluster."
            ),
        ] = 60,
        kubernetes: Annotated[
            dagger.Service | None,
            Doc(
                "Endpoint to exsisting k3s service to attach to. "
                "Example (from the CLI): tcp://localhost:1997"
            ),
        ] = None,
        image: Annotated[
            str | None,
            Doc(
                "Container image to test on HPC. If given, it will override any itwinai "
                "container previously created by other functions in the chain"
            ),
        ] = None,
    ) -> Self:
        """Test container on remote HPC using interLink"""

        image = image or self.full_name

        if not image:
            raise RuntimeError(
                "Undefined container image. Either chain this function to "
                "another that will create and publish a container, or give an image "
                "as argument."
            )
        self._logs.append(f"INFO: testing on hpc image {image}")

        # Create pod manifest
        gpus_per_node = 1
        cpus_per_gpu = 4
        num_nodes = 2
        jobscript = "/app/tests/torch/slurm.vega.sh"
        pre_exec_cmd = (
            "export CONTAINER_PATH=itwinai_dist_test.sif "
            f"&& singularity pull --force $CONTAINER_PATH docker://{image} "
            f"&& singularity exec $CONTAINER_PATH cat {jobscript} > slurm.vega.sh "
            # Activate env on Vega
            "&& . /etc/bashrc "
            "&& source /ceph/hpc/software/cvmfs_env.sh "
            "&& module use /usr/share/Modules/modulefiles "
            # Quick fix
            # f"&& export SLURM_GPUS_PER_NODE={gpus_per_node} "
            # f"&& export SLURM_CPUS_PER_GPU={cpus_per_gpu} "
            # Env variables necessary for tests
            "&& export MNIST_PATH=/ceph/hpc/data/st2301-itwin-users/mbunino/mnist "
            "&& export NO_COLOR=1 "
            # Launch code in SLURM job
            # DDP
            "&& export DIST_MODE=ddp "
            "&& export RUN_NAME=ddp-itwinai "
            "&& export COMMAND='pytest -v -m torch_dist /app/tests' "
            "&& source slurm.vega.sh "
            # DeepSpeed
            "&& export DIST_MODE=deepspeed "
            "&& export RUN_NAME=ds-itwinai "
            "&& export COMMAND='pytest -v -m deepspeed_dist /app/tests' "
            "&& source slurm.vega.sh "
            # Horovod
            "&& export DIST_MODE=horovod "
            "&& export RUN_NAME=horovod-itwinai "
            "&& export COMMAND='pytest -v -m horovod_dist /app/tests' "
            "&& source slurm.vega.sh "
        )
        # Request memory and CPUs
        resources = {
            "limits": {"cpu": 48, "memory": "150Gi"},
            "requests": {"cpu": cpus_per_gpu * gpus_per_node, "memory": "20Gi"},
        }
        hpc_partition = "gpu"
        annotations = {
            "slurm-job.vk.io/flags": (
                # --cpus-per-gpu fails on Vega through interLink
                f"-p {hpc_partition} --gres=gpu:{gpus_per_node} "
                f"--gpus-per-node={gpus_per_node} "
                f"--ntasks-per-node=1 --nodes={num_nodes} "
                f"--cpus-per-task={cpus_per_gpu * gpus_per_node} "
                "--time=00:30:00"
            ),
            "slurm-job.vk.io/pre-exec": (
                "trap 'export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1' ERR && "
                f"(set -e && {pre_exec_cmd}) ; trap - ERR"
            ),
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = ["sleep 10 && exit $PRE_EXEC_RETURN_CODE"]
        pod_name = f"ci-test-itwinai-hpc-{self.unique_id}"

        # Launch interLink service
        interlink_svc = self.interlink(
            values=values, name=name, wait=wait, kubernetes=kubernetes
        )
        async with interlink_svc.start_serving():
            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=cmd_args,
                name=pod_name,
                resources=resources,
                target_node=interlink_svc.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)
            self._logs.append(f"INFO: submitting the following pod:\n{pod_manifest_str}")

            # self._logs.append("DEBUG: interlink cluster status")
            # self._logs.append(await interlink_svc.status())

            self._logs.append(await interlink_svc.submit_pod(pod_manifest_str))
            status, logs = await interlink_svc.wait_pod(
                name=pod_name, timeout=3000, poll_interval=30
            )

        if status not in ["Succeeded", "Completed"]:
            message = (
                f"Pod did not complete successfully! Status: {status}\n"
                f"{"#"*100}\nJOB LOGS:\n\n{logs}"
            )
            print(message)
            raise RuntimeError(message)

        self._logs.append(f"INFO: pod finished with status: '{status}'")
        self._logs.append(f"INFO: pod finished with logs: \n{logs}")

        return self

    @function
    async def test_n_publish(
        self,
        values: Annotated[dagger.Secret, Doc("interLink installer configuration")],
        stage: Annotated[
            Stage,
            Doc("Whether to push the final image to the production image name or not"),
        ] = Stage.DEV,
        tag_template: Annotated[
            str | None,
            Doc(
                "Custom image tag pattern. Example: "
                "'${itwinai_version}-torch${torch_version}-${os_version}'"
            ),
        ] = None,
        framework: Annotated[
            MLFramework, Doc("ML framework in container")
        ] = MLFramework.TORCH,
        skip_hpc: Annotated[bool, Doc("Skip tests on remote HPC")] = False,
        kubernetes: Annotated[
            dagger.Service | None,
            Doc(
                "Endpoint to exsisting k3s service to attach to. "
                "Example (from the CLI): tcp://localhost:1997"
            ),
        ] = None,
        interlink_cluster_name: Annotated[
            str, Doc("Name of the k3s cluster in which the interLink VK is deployed")
        ] = "interlink-cluster",
    ) -> str:
        """End-to-end pipeline to test a container and push it, including both local
        tests and tests on HPC via interLink.
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

        if not skip_hpc:
            # Publish to registry with random hash
            await self.publish()
            # Test on HPC with
            await self.test_hpc(
                values=values, kubernetes=kubernetes, name=interlink_cluster_name
            )
        else:
            self._logs.append("INFO: skipping tests on HPC")

        # Publish to registry with final hash
        itwinai_version = (
            await self.container()
            .with_exec(
                [
                    "bash",
                    "-c",
                    'pip list | grep -w "itwinai" | head -n 1 | tr -s " " | cut -d " " -f2 ',
                ]
            )
            .stdout()
        ).strip()
        os_info = (
            await self.container()
            .with_exec(
                [
                    "bash",
                    "-c",
                    "cat /etc/*-release",
                ]
            )
            .stdout()
        ).strip()
        os_version = get_codename(os_info)

        if framework == MLFramework.TORCH:
            tag_template = (
                tag_template or "${itwinai_version}-torch${framework_version}-${os_version}"
            )
            framework_version = (
                await self.container()
                .with_exec(
                    [
                        "bash",
                        "-c",
                        (
                            'pip list | grep -w "torch[^-]" | head -n 1 | tr -s " " '
                            '| cut -d " " -f2 | cut -f1,2 -d .'
                        ),
                    ]
                )
                .stdout()
            ).strip()
        elif framework == MLFramework.TENSORFLOW:
            tag_template = (
                tag_template or "${itwinai_version}-tf${framework_version}-${os_version}"
            )
            framework_version = (
                await self.container()
                .with_exec(
                    [
                        "bash",
                        "-c",
                        (
                            # TODO: check this command
                            'pip list | grep -w "tensorflow[^-]" | head -n 1 | tr -s " " '
                            '| cut -d " " -f2 | cut -f1,2 -d .'
                        ),
                    ]
                )
                .stdout()
            ).strip()
        # In GH actions ${...} is interpolated, so we replace $ with @
        tag = Template(tag_template.replace("@", "$")).substitute(
            itwinai_version=itwinai_version,
            framework_version=framework_version,
            os_version=os_version,
        )
        self._logs.append(f"INFO: Preparing final image name {image}:{tag}")
        await self.publish(name=image, tag=tag)

        return self.logs()

    @function
    def singularity(
        self,
        base_image: Annotated[
            str, Doc("Base Singularity image")
        ] = "quay.io/singularity/docker2singularity",
    ) -> Singularity:
        return Singularity(base_image=base_image)

    @function
    def to_singularity(
        self,
        base_image: Annotated[
            str, Doc("Base Singularity image")
        ] = "quay.io/singularity/docker2singularity",
    ) -> dagger.Container:
        """Convert itwinai container to a Singularity image and return a container with the
        produced image in it.
        """
        singularity = self.singularity(base_image=base_image)
        return singularity.container().with_file(
            "container.sif", singularity.export(self.container())
        )

    @function
    async def publish_singularity(
        self,
        password: Annotated[dagger.Secret, Doc("Password for registry")],
        username: Annotated[dagger.Secret, Doc("Username for registry")],
        uri: Annotated[
            str,
            Doc("Target URI for the image"),
        ],
    ) -> str:
        return await self.singularity().publish(
            container=self.container(), password=password, username=username, uri=uri
        )
