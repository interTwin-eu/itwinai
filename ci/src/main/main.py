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

    docker_registry: str
    singularity_registry: str
    name: str
    tag: str | None
    container: dagger.Container | None = dagger.field(default=None)

    _unique_id: str | None = dataclasses.field(default=None, init=False)
    _logs: list[str] = dataclasses.field(default_factory=list, init=False)

    @classmethod
    def create(
        cls,
        docker_registry: Annotated[
            str,
            Doc("The Docker registry base URL where the container will be published"),
        ] = "ghcr.io/intertwin-eu",
        singularity_registry: Annotated[
            str,
            Doc("The Docker registry base URL where the container will be published"),
        ] = "registry.egi.eu/dev.intertwin.eu",
        name: Annotated[
            str,
            Doc("The name of the container image"),
        ] = "itwinai-dev",
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
            name=name,
            tag=tag,
            container=container,
        )

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
        self._logs.append(f"INFO: Built container for itwinai v{itwinai_version}")

        self.container = self.container.with_label(
            name="org.opencontainers.image.version",
            value=itwinai_version,
        )

        return self

    @function
    async def test_local(self) -> Self:
        """Test itwinai container image with pytest on non-HPC environments."""
        tests_result = await self.container.with_exec(
            [
                "pytest",
                "-v",
                "-m",
                "not hpc and not functional and not tensorflow",
                "/app/tests",
            ]
        ).stdout()
        self._logs.append("INFO: Results of local tests:")
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
        """Push container to registry"""

        uri = uri or f"{self.docker_registry}/{self.name}:{self.tag}"
        outcome = await (
            self.container.with_label(
                name="org.opencontainers.image.ref.name",
                value=uri,
            )
            # Invalidate cache to ensure that the container is always pushed
            .with_env_variable(
                "CACHE", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            )
            .publish(address=uri)
        )
        self._logs.append(f"INFO: results of publishing to {uri}")
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

        image = image or f"docker://{self.docker_registry}/{self.name}:{self.tag}"

        if not image:
            raise RuntimeError(
                "Undefined container image. Either chain this function to "
                "another that will create and publish a container, or give an image "
                "as argument."
            )
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
                "--time=00:30:00 --exclude gn42"
            ),
            "slurm-job.vk.io/pre-exec": (
                "trap 'export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1' ERR && "
                f"(set -e && {pre_exec_cmd}) ; trap - ERR"
            ),
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = ["sleep 10 && exit $PRE_EXEC_RETURN_CODE"]
        pod_name = f"ci-test-itwinai-hpc-{self.name}-{self.tag}"

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
                "'${itwinai_version}-torch${framework_version}-${os_version}'"
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
        password: Annotated[
            dagger.Secret | None, Doc("Password for Singularity registry")
        ] = None,
        username: Annotated[
            dagger.Secret | None, Doc("Username for Singularity registry")
        ] = None,
    ) -> str:
        """DEPRECATED. End-to-end pipeline to test a container and push it, including both local
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
            # # Publish to registry with random hash
            # await self.publish()
            # # Test on HPC with
            # await self.test_hpc(
            #     values=values, kubernetes=kubernetes, name=interlink_cluster_name
            # )

            # Using singularity...

            assert username is not None, "Missing username for Singularity registry"
            assert password is not None, "Missing password for Singularity registry"
            # Publish to registry with random hash
            uri = f"oras://{self.singularity_registry}/{self.name}:{self.tag}"
            await self.publish_singularity(uri=uri, username=username, password=password)

            # Test on HPC with
            await self.test_hpc(
                values=values, kubernetes=kubernetes, name=interlink_cluster_name, image=uri
            )

            # Publish to registry with final URI
            uri = f"oras://{self.singularity_registry}/{self.name}:{self.tag}"
            await self.publish_singularity(uri=uri, username=username, password=password)
        else:
            self._logs.append("INFO: skipping tests on HPC")

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
        # In GH actions ${...} is interpolated, so we replace $ with @
        tag = Template(tag_template.replace("@", "$")).substitute(
            itwinai_version=itwinai_version,
            framework_version=framework_version,
            os_version=os_version,
        )
        self._logs.append(f"INFO: Preparing final image name {image}:{tag}")
        self.image = image
        self.tag = tag
        await self.publish()

        return self.logs()

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
    ) -> str:
        """CI pipeline for pre-release containers. Tests are only local."""

        # Test locally
        await self.test_local()

        # Publish to Docker registry with "final" tag
        tag = await self._evaluate_tag_template(tag_template=tag_template, framework=framework)
        await self.publish(uri=f"{self.docker_registry}/{self.name}:{tag}")

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

        # Test locally
        await self.test_local()

        if not skip_hpc:
            assert username is not None, "Missing username for Singularity registry"
            assert password is not None, "Missing password for Singularity registry"
            # Publish to registry with random hash
            uri = f"oras://{self.singularity_registry}/{self.name}:{self.tag}"
            await self.publish_singularity(uri=uri, username=username, password=password)

            # Test on HPC with
            await self.test_hpc(
                values=values, kubernetes=kubernetes, name=interlink_cluster_name, image=uri
            )

        else:
            self._logs.append("INFO: skipping tests on HPC")

        # Publish to registry with "final" tag
        final_tag = self._evaluate_tag_template(tag_template=tag_template, framework=framework)
        for tag in [final_tag, "latest"]:
            # Publish to Docker registry
            self.publish(uri=f"{self.docker_registry}/{self.name}:{tag}")
            if not skip_hpc:
                # Publish to Singularity registry
                await self.publish_singularity(
                    uri=f"oras://{self.singularity_registry}/{self.name}:{tag}",
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
            dagger.Container, Doc("Base Singularity image")
        ] = dag.container().from_("quay.io/singularity/docker2singularity"),
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
        uri = uri or f"oras://{self.singularity_registry}/{self.name}:{self.tag}"
        self._logs.append(f"INFO: the Singularity image will be published at: {uri}")
        stdout = await self.singularity(docker=self.container).publish(
            password=password, username=username, uri=uri
        )
        self._logs.append(stdout)
        return self
