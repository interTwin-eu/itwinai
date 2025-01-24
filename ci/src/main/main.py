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
from typing import Annotated, Optional, Self
import yaml

import dagger
from dagger import BuildArg, Doc, dag, function, object_type

from .literals import MLFramework, Stage
from .interlink import InterLinkService
from .k8s import K8sClient, create_pod_manifest


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
    name: Annotated[
        Optional[str],
        Doc(
            "Unique name to identify the image from tag. Could be the git commit hash of HEAD"
        ),
    ] = None
    container: Annotated[Optional[dagger.Container], Doc("Container instance")] = (
        dataclasses.field(default=None, init=False)
    )
    full_name: Annotated[
        Optional[str],
        Doc("Full image name. Example: ghcr.io/intertwin-eu/itwinai-dev:0.2.3-torch2.4-jammy"),
    ] = dataclasses.field(default=None, init=False)
    _unique_id: Optional[str] = dataclasses.field(default=None, init=False)
    sif: Annotated[Optional[dagger.File], Doc("SIF file")] = dataclasses.field(
        default=None, init=False
    )

    # Note: since build_container returns self, when executing only it through dagger call
    # (e.g., dagger call build-container [args]), dagger will actually execute all the
    # methods in this class in order (?)

    @property
    def unique_id(self) -> str:
        """Unique ID of the container. Falls back to random integer."""
        if self._unique_id is None:
            self._unique_id = self.name or str(random.randrange(10**8))
        return self._unique_id

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
    async def interlink_service(
        self, values: dagger.File, name: str = "unk", wait: int = 60
    ) -> dagger.Service:
        """Create and return an interLink service (k3s service). Can be used in the CLI
        with "<service> up". Example:

        >>> dagger call interlink-service --values tmp.yaml up
        """
        interlink_svc = InterLinkService(values=values, name=name, wait=wait)
        svc = await interlink_svc.start()
        # print(await interlink_svc.kubeconfig.contents())
        return svc

    @function
    async def interlink_client(
        self, values: dagger.File, name: str = "unk", wait: int = 60
    ) -> dagger.Container:
        """Create an interLink service (k3s service) and return the container of the k8s
        client, which can be chanined in the CLI, e.g., with a terminal. Example:

        >>> dagger call interlink-client --values tmp.yaml teminal
        """
        interlink_svc = InterLinkService(values=values, name=name, wait=wait)
        await interlink_svc.start()
        k8s_client = K8sClient(kubeconfig=interlink_svc.kubeconfig)
        return k8s_client.container()

    @function
    async def pipeline(self, values: dagger.File, name: str = "unk") -> str:
        interlink_svc = InterLinkService(values=values, name=name, wait=60)
        async with interlink_svc.start_serving():
            k8s_client = K8sClient(kubeconfig=interlink_svc.kubeconfig)
            # await k8s_client.container().terminal()

            pod_str = """apiVersion: v1
kind: Pod
metadata:
  name: short-lived-pod
spec:
  containers:
  - name: short-lived-container
    image: busybox
    command: ["sleep", "10", "&&", "exit", "2"]
  restartPolicy: Never
"""
            stdout = []
            stdout.append(await k8s_client.status())
            stdout.append(await k8s_client.submit_pod(pod_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod("short-lived-pod", timeout=1)
            stdout.extend([status, logs])

        return "\n\n".join(stdout)

        # interlink_srv: dagger.Service = dag.interlink(name).interlink_cluster(values)
        # await interlink_srv.start()

        # kubeconfig: dagger.File = dag.interlink(name).cluster_config(local=False)

        # client = (
        #     dag.container()
        #     .from_("alpine/helm")
        #     .with_exec(["apk", "add", "kubectl"])
        #     .with_mounted_file("/.kube/config", kubeconfig)
        #     .with_env_variable("KUBECONFIG", "/.kube/config")
        # )

        # pods = await client.with_exec("kubectl get pods".split()).stdout()
        # nodes = await client.with_exec("kubectl get nodes".split()).stdout()
        # return f"Nodes:\n{nodes}\n\nPods:\n{pods}"

        # Pass kubeconfig to alpine/kubectl

        # Close service
        # interlink_srv.stop()

    @function
    async def interlink_status(self, values: dagger.File) -> str:
        """Get status of K3S cluster with interLink deployment."""
        return await self.interlink(values).status()

    @function
    def interlink_container(self, values: dagger.File) -> dagger.Container:
        """Return interlink container -- good for chaining in the cli (e.g., open terminal)"""
        return self.interlink(values).container()

    @function
    async def il_test(self, values: dagger.File, name: str = "unk") -> str:
        """Test container on remote HPC using interLink"""
        # Create pod manifest
        gpus_per_node = 1
        cpus_per_gpu = 1

        # Request memory and CPUs
        resources = {
            "limits": {"cpu": 48, "memory": "150Gi"},
            "requests": {"cpu": cpus_per_gpu * gpus_per_node, "memory": "20Gi"},
        }
        annotations = {
            "slurm-job.vk.io/flags": (
                # --cpus-per-gpu fails on Vega through interLink
                f"-p gpu --gres=gpu:{gpus_per_node} --gpus-per-node={gpus_per_node} "
                "--ntasks-per-node=1 --nodes=1 "
                f"--cpus-per-task={cpus_per_gpu * gpus_per_node} "
                "--time=00:30:00"
            )
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = ["sleep 10 && ls -la"]
        pod_name = f"ci-test-itwinai-hpc-{self.unique_id}"

        # Launch interLink service
        interlink_svc = InterLinkService(values=values, name=name, wait=60)
        async with interlink_svc.start_serving():
            k8s_client = K8sClient(kubeconfig=interlink_svc.kubeconfig)
            # await k8s_client.container().terminal()

            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=cmd_args,
                name=pod_name,
                resources=resources,
                target_node=interlink_svc.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)

            stdout = []
            stdout.append(await k8s_client.status())
            stdout.append(await k8s_client.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod(name=pod_name, timeout=3000)
            stdout.extend([status, logs])

        return "\n\n".join(stdout)

    @function
    async def il(self, values: dagger.File) -> str:
        pod_str = """apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: my-app
spec:
  containers:
  - name: my-container
    image: nginx:1.21
    ports:
    - containerPort: 80"""

        await self.interlink(values).submit_pod(pod_str)
        return await self.interlink(values).get_pod_status("my-pod")

    # @function
    # def interlink_cluster(self, values: dagger.File) -> dagger.Container:
    #     if not self._interlink:
    #         self._interlink = dag.interlink().interlink_cluster(values)
    #     return self._interlink

    @function
    def terminal(self) -> dagger.Container:
        """Open terminal into container"""
        return self.container.terminal()

    @function
    async def test_local(self) -> str:
        """Test itwinai container image with pytest on non-HPC environments."""
        test_cmd = [
            "pytest",
            "-v",
            "-m",
            "not hpc and not functional and not tensorflow",
            "/app/tests",
        ]
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

        tag = tag or self.unique_id
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
    async def test_hpc(self, values: dagger.File, name: str = "unk") -> str:
        """Test container on remote HPC using interLink"""

        # Create pod manifest
        gpus_per_node = 4
        cpus_per_gpu = 4
        jobscript = "/app/tests/torch/slurm.vega.sh"
        pre_exec_cmd = (
            "export CONTAINER_PATH=itwinai_dist_test.sif "
            f"&& singularity pull --force $CONTAINER_PATH docker://{self.full_name} "
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
        annotations = {
            "slurm-job.vk.io/flags": (
                # --cpus-per-gpu fails on Vega through interLink
                f"-p gpu --gres=gpu:{gpus_per_node} --gpus-per-node={gpus_per_node} "
                "--ntasks-per-node=1 --nodes=1 "
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
        interlink_svc = InterLinkService(values=values, name=name, wait=60)
        async with interlink_svc.start_serving():
            k8s_client = K8sClient(kubeconfig=interlink_svc.kubeconfig)
            # await k8s_client.container().terminal()

            pod_manifest = create_pod_manifest(
                annotations=annotations,
                image_path=image_path,
                cmd_args=cmd_args,
                name=pod_name,
                resources=resources,
                target_node=interlink_svc.vk_name,
            )
            pod_manifest_str = yaml.dump(pod_manifest)

            stdout = []
            stdout.append(await k8s_client.submit_pod(pod_manifest_str))
            # await k8s_client.container().terminal()
            status, logs = await k8s_client.wait_pod(name=pod_name, timeout=3000)
            stdout.extend([status, logs])

        if status not in ["Succeeded", "Completed"]:
            raise RuntimeError(f"Pod did not complete successfully! Status: {status}\n{stdout}")

        return f"Pod finished with status: {status}\n{stdout}"

        # Submit pod

        # kubeconfig_str = await kubeconfig.plaintext()
        # status = submit_job(
        #     kubeconfig_str=kubeconfig_str, pod_manifest=pod_manifest, verbose=False
        # )

        # if status not in ["Succeeded", "Completed"]:
        #     raise RuntimeError(f"Pod did not complete successfully! Status: {status}")

        # return f"Pod finished with status: {status}"

        # import yaml

        # pod_manifest_str = yaml.dump(pod_manifest)
        # self.submit_pod_interlink(pod_str=pod_manifest_str, values=...)
        # # Wait for completion...?

    @function
    async def test_n_publish(
        self,
        values: dagger.File,
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
        skip_hpc: Annotated[bool, Doc("Skip tests on remote HPC")] = False,
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

        if not skip_hpc:
            # Publish to registry with random hash
            await self.publish()
            # Test on HPC with
            await self.test_hpc(values=values)

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
        await self.publish(name=image, tag=tag)

    @function
    async def singularity(self, src_container: str) -> dagger.File:
        """Convert Docker to Singulartiy/Apptainer"""
        # https://hpc-docs.cubi.bihealth.org/how-to/software/apptainer/#option-2-converting-docker-images
        singularity_conv = (
            dag.container()
            .from_("quay.io/singularity/docker2singularity")
            # This part is only if you want to use the builtin conversion script which depends
            # on the host Docker runtime
            # .with_unix_socket(path="/var/run/docker.sock", source=socket)
            # # insecure_root_capabilities=True in with_exec is equivalent to --privileged
            # .with_exec(
            #     [
            #         "bash",
            #         "-c",
            #         f"/docker2singularity.sh --name /output/container.sif {container}",
            #     ],
            #     insecure_root_capabilities=True,
            # )
            .with_exec(
                [
                    "bash",
                    "-c",
                    f"singularity pull container.sif docker://{src_container}",
                ]
            )
            .terminal()
        )
        # Calling export on self.sif from here will not work because export must be called
        # from the CLI
        # .export(path=(Path(output_dir) / "container.sif").resolve().as_posix())
        self.sif = singularity_conv.file("container.sif")
        return self.sif
