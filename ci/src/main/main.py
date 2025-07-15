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
import uuid
from string import Template
from typing import Annotated, Dict, Self

import dagger
import yaml
from dagger import BuildArg, Doc, Ignore, dag, function, object_type

from .hpc import Singularity
from .interlink import InterLink
from .k8s import create_pod_manifest
from .literals import MLFramework
from .utils import get_codename

ignore_list = [
    "**",
    "!src/itwinai/**",
    "!tests/**",
    "!env-files/**",
    "!use-cases/**",
    "**/__pycache__",
    "!pyproject.toml",
]


@object_type
class Itwinai:
    """CI/CD for container images of itwinai."""

    docker_registry: str
    singularity_registry: str
    image: str
    tag: str | None
    container: dagger.Container | None = dagger.field(default=None)
    nickname: str

    _unique_id: str | None = dataclasses.field(default=None, init=False)
    _logs: list[str] = dataclasses.field(default_factory=list, init=False)
    _all_containers: Dict[str, dagger.Container] = dataclasses.field(
        default_factory=dict, init=False
    )

    @classmethod
    def create(
        cls,
        docker_registry: Annotated[
            str,
            Doc("The Docker registry base URL where the container will be published"),
        ] = "ghcr.io/intertwin-eu",
        singularity_registry: Annotated[
            str,
            Doc(
                "Harbor registry namespace (i.e., 'registry/project') where to publish the "
                "Singularity images"
            ),
        ] = "registry.egi.eu/dev.intertwin.eu",
        image: Annotated[
            str,
            Doc("The name of the container image"),
        ] = "itwinai-dev",
        nickname: Annotated[
            str,
            Doc(
                "A simple name to indicate the flavor of the image. Used to generate the "
                "corresponding 'latest' tag"
            ),
        ] = "torch",
        tag: Annotated[
            str | None,
            Doc("Tag for the container image. Defaults to random uuid if not provided"),
        ] = None,
        container: Annotated[
            dagger.Container | None,
            Doc("Optional container image to use as itwinai container"),
        ] = None,
    ):
        tag = tag or str(uuid.uuid4())
        return cls(
            singularity_registry=singularity_registry,
            docker_registry=docker_registry,
            image=image,
            tag=tag,
            container=container,
            nickname=nickname,
        )

    @function
    def logs(self) -> str:
        """Print the logs generted so far. Useful to terminate a chain of steps
        returning Self, preventing lazy execution of it.
        """
        return "\n\n".join(self._logs) or "There are no logs to show"

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
        build_arm: Annotated[
            bool,
            Doc("Whether to build for ARM"),
        ] = False,
    ) -> Self:
        """Build itwinai container image from existing Dockerfile"""
        if build_args:
            build_args = [
                BuildArg(name=arg_couple.split("=")[0], value=arg_couple.split("=")[1])
                for arg_couple in build_args
            ]
        if build_arm:
            # Build also for ARM
            print("INFO: Building container for linux/arm64 platform")
            arm_container = dag.container(platform=dagger.Platform("linux/arm64")).build(
                context=context,
                dockerfile=dockerfile,
                build_args=build_args,
            )
            self._all_containers["linux/arm64"] = arm_container
            # Just a trick to force build now
            await arm_container.with_exec(["ls", "-la"]).stdout()

        print("INFO: Building container for linux/amd64 platform")
        self.container = (
            dag.container(platform=dagger.Platform("linux/amd64"))
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
        self._all_containers["linux/amd64"] = self.container

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
        self._logs.append(f"INFO: Built container for itwinai v{itwinai_version}")

        self.container = self.container.with_label(
            name="org.opencontainers.image.version",
            value=itwinai_version,
        )

        return self

    @function
    async def test_local(
        self, cmd: Annotated[list[str] | None, Doc("Command to run tests")] = None
    ) -> Self:
        """Test itwinai container image with pytest on non-HPC environments."""

        cmd = cmd or [
            "pytest",
            "-v",
            "--disable-warnings",
            # tell pytest-xdist to parallelize tests over logical cores
            "-n",
            "logical",
            "-m",
            "not hpc and not functional and not tensorflow",
            "/app/tests",
        ]

        for platform_name, container in self._all_containers.items():
            print(f"INFO: Testing container for {platform_name} platform")
            self._logs.append(
                f"INFO: running pytest for 'local' tests on container for {platform_name}:"
            )
            tests_result = await container.with_exec(cmd).stdout()
            self._logs.append(tests_result)
        return self

    @function
    async def publish(
        self,
        uri: Annotated[
            str | None,
            Doc("Optional target URI for the image"),
        ] = None,
    ) -> Self:
        """Push container to registry. Multi-arch support."""

        uri = uri or f"{self.docker_registry}/{self.image}:{self.tag}"

        all_containers = []
        for platform_name, container in self._all_containers.items():
            print(f"INFO: Preparing to publish container for {platform_name} platform")
            all_containers.append(
                container.with_label(
                    name="org.opencontainers.image.ref.name",
                    value=uri,
                )
                # Invalidate cache to ensure that the container is always pushed
                .with_env_variable(
                    "CACHE", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                )
            )
        outcome = await dag.container().publish(address=uri, platform_variants=all_containers)
        self._logs.append(f"INFO: publishing Docker image to {uri}")
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
                "container previously created by other functions in the chain. Example: "
                "docker://ghcr.io/intertwin-eu/itwinai:latest"
            ),
        ] = None,
    ) -> Self:
        """Test container on remote HPC using interLink. Note that the container image must
        already exist in some publicly accessible containers registry.
        """

        image = image or f"docker://{self.docker_registry}/{self.image}:{self.tag}"
        self._logs.append(f"INFO: testing on HPC image {image}")

        # Create pod manifest
        gpus_per_node = 1
        cpus_per_gpu = 4
        num_nodes = 2
        jobscript = "/app/tests/torch/slurm.vega.sh"
        pre_exec_cmd = (
            "export CONTAINER_PATH=itwinai_dist_test.sif "
            f"&& singularity pull --force $CONTAINER_PATH {image} "
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
            "&& export SHARED_FS_PATH=/ceph/hpc/data/st2301-itwin-users/tmp-mbunino2 "
            # Launch code in SLURM job
            # Ray
            "&& export DIST_MODE=ray "
            "&& export RUN_NAME=ray-itwinai "
            "&& export COMMAND='pytest -v -m ray_dist /app/tests' "
            "&& source slurm.vega.sh "
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
            # Ray: must be at the bottom! srun are blocking forever and won't release
            # the resources, this no more srun cannot be invoked after a Ray cluster
            # is launched.
            "&& export DIST_MODE=ray "
            "&& export RUN_NAME=ray-itwinai "
            "&& export COMMAND='pytest -v -m ray_dist /app/tests' "
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
                "--time=01:00:00 "
            ),
            "slurm-job.vk.io/pre-exec": (
                "trap 'export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1' ERR && "
                f"(set -e && {pre_exec_cmd}) ; trap - ERR"
            ),
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = ["sleep 10 && exit $PRE_EXEC_RETURN_CODE"]
        pod_name = f"ci-test-itwinai-hpc-{self.image}-{self.tag}"

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
                name=pod_name, timeout=10000, poll_interval=600
            )

        if status not in ["Succeeded", "Completed"]:
            message = (
                f"Pod did not complete successfully! Status: {status}\n"
                f"{'#' * 100}\nJOB LOGS:\n\n{logs}"
            )
            print(message)
            raise RuntimeError(message)

        self._logs.append(f"INFO: pod finished with status: '{status}'")
        self._logs.append(f"INFO: pod finished with logs: \n{logs}")

        return self

    @function
    async def dev_pipeline(
        self,
        tag_template: Annotated[
            str,
            Doc("Custom image tag pattern."),
        ] = "${itwinai_version}-torch${framework_version}-${os_version}",
        framework: Annotated[
            MLFramework, Doc("ML framework in container")
        ] = MLFramework.TORCH,
        skip_singularity: Annotated[bool, Doc("Avoid publishing a Singularity image")] = False,
        password: Annotated[
            dagger.Secret | None, Doc("Password for Singularity registry")
        ] = None,
        username: Annotated[
            dagger.Secret | None, Doc("Username for Singularity registry")
        ] = None,
    ) -> str:
        """CI pipeline for pre-release containers. Tests are only local."""

        if not skip_singularity:
            assert username is not None, "Missing username for Singularity registry"
            assert password is not None, "Missing password for Singularity registry"

        # Test locally
        await self.test_local()

        # Publish to Docker registry with "final" tag
        tag = await self._evaluate_tag_template(tag_template=tag_template, framework=framework)
        await self.publish(uri=f"{self.docker_registry}/{self.image}:{tag}")

        if not skip_singularity:
            # Publish to Singularity registry
            await self.publish_singularity(
                uri=f"oras://{self.singularity_registry}/{self.image}:{tag}",
                username=username,
                password=password,
            )

        return self.logs()

    @function
    async def release_pipeline(
        self,
        values: Annotated[dagger.Secret, Doc("interLink installer configuration")],
        tag_template: Annotated[
            str | None,
            Doc(
                "Custom image tag pattern. Example: "
                "'${itwinai_version}-torch${framework_version}-${os_version}'"
            ),
        ] = None,
        framework: Annotated[
            MLFramework, Doc("ML framework in container")
        ] = MLFramework.TORCH,
        skip_hpc: Annotated[bool, Doc("Skip tests on remote HPC")] = False,
        skip_singularity: Annotated[bool, Doc("Avoid publishing a Singularity image")] = False,
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
        password: Annotated[
            dagger.Secret | None, Doc("Password for Singularity registry")
        ] = None,
        username: Annotated[
            dagger.Secret | None, Doc("Username for Singularity registry")
        ] = None,
    ) -> str:
        """CI pipeline for release containers. Test on HPC and generate Singularity images."""

        if not skip_singularity:
            assert username is not None, "Missing username for Singularity registry"
            assert password is not None, "Missing password for Singularity registry"

        # Test locally
        await self.test_local()

        if not skip_hpc:
            if skip_singularity:
                # The Docker image will be automatically converted to Singularity on HPC
                # before running the tests
                uri = f"{self.docker_registry}/itwinai-dev:{self.tag}"
                await self.publish(uri=uri)
                # Append Docker prefix to tell singularity where to pull the image from
                uri = f"docker://{uri}"
            else:
                # Publish to registry with random hash
                uri = f"oras://{self.singularity_registry}/itwinai-dev:{self.tag}"
                await self.publish_singularity(uri=uri, username=username, password=password)

            # Test on HPC with
            await self.test_hpc(
                values=values, kubernetes=kubernetes, name=interlink_cluster_name, image=uri
            )

        else:
            self._logs.append("INFO: skipping tests on HPC")

        # Publish to registry with "final" and latest tag
        final_tag = await self._evaluate_tag_template(
            tag_template=tag_template, framework=framework
        )
        for tag in [final_tag, f"{self.nickname}-latest"]:
            # Publish to Docker registry
            await self.publish(uri=f"{self.docker_registry}/{self.image}:{tag}")
            if not skip_singularity:
                # Publish to Singularity registry
                await self.publish_singularity(
                    uri=f"oras://{self.singularity_registry}/{self.image}:{tag}",
                    username=username,
                    password=password,
                )

        return self.logs()

    async def _evaluate_tag_template(self, tag_template: str, framework: MLFramework) -> str:
        """Interpolate the fields in the tag template with values computed from the
        container.
        """
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

        # In GH actions ${...} is interpolated, so we replace $ with @
        return Template(tag_template.replace("@", "$")).substitute(
            itwinai_version=itwinai_version,
            framework_version=framework_version,
            os_version=os_version,
        )

    @function
    def singularity(
        self,
        base_image: Annotated[
            str, Doc("Base Singularity image")
        ] = "quay.io/singularity/docker2singularity",
        docker: Annotated[
            dagger.Container | None,
            Doc(
                "Docker container to convert to Singularity. "
                "If given it overrides the current itwinai container."
            ),
        ] = None,
    ) -> Singularity:
        """Access Singularity module."""
        return Singularity(base_image=base_image, docker=docker or self.container)

    @function
    async def publish_singularity(
        self,
        password: Annotated[dagger.Secret, Doc("Password for Singularity registry")],
        username: Annotated[dagger.Secret, Doc("Username for Singularity registry")],
        uri: Annotated[
            str | None,
            Doc("Optional target URI for the image"),
        ] = None,
    ) -> Self:
        """Convert itwinai container to Singularity and push it to some registry."""
        uri = uri or f"oras://{self.singularity_registry}/{self.image}:{self.tag}"
        self._logs.append(f"INFO: the Singularity image will be published to: {uri}")
        stdout = await self.singularity(docker=self.container).publish(
            password=password, username=username, uri=uri
        )
        self._logs.append(stdout)
        return self
