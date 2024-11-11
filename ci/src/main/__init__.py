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

from typing import Annotated, Optional, Self
import dataclasses
import random
import os 

import dagger
from dagger import dag, function, object_type, Doc


@object_type
class Itwinai:

    container: Optional[dagger.Container] = dataclasses.field(default=None, init=False)
    full_name: Optional[str] = dataclasses.field(default=None, init=False)
    
    # Note: since build_container returns self, when executing only it through dagger call
    # (e.g., dagger call build-container [args]), dagger will actually execute all the
    # methods in this class in order

    @function
    def build_container(
        self,
        context: Annotated[
            dagger.Directory,
            Doc("location of source directory"),
        ],
        dockerfile: Annotated[
            str,
            Doc("location of Dockerfile"),
        ],
        # TODO: pass build args as well
    ) -> Self:
        """Build itwinai container image from existing Dockerfile"""
        self.container = (
            dag.container()
            .build(context=context, dockerfile=dockerfile)
        )
        return self
    
    @function
    def terminal(self)->dagger.Container:
        return self.container.terminal()
        

    @function
    async def test_local(self) -> str:
        """Test itwinai container image with pytest on non-HPC environments."""
        test_cmd = [
            "pytest",
            "-v",
            "-m",
            "not hpc and not functional",
            "tests"
        ]
        return await (
            self.container
            .with_exec(test_cmd)
            .stdout()
        )

    @function
    async def publish(
        self,
        registry: str = "ghcr.io/intertwin-eu",
        name: str = "itwinai-dev",
        tag: Optional[str] = None
        # TODO: use annotated for args
    ) -> str:
        """Push container to registry"""
        from datetime import datetime
        tag = tag if tag else random.randrange(10 ** 8)
        self.full_name = f"{registry}/{name}:{tag}"
        return await (
            self.container
            # Invalidate cache to ensure that the container is always pushed
            .with_env_variable("CACHE", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            .publish(self.full_name)
        )
    
    @function
    async def test_hpc(
        self,
        kubeconfig_str: Annotated[
            dagger.Secret, 
            Doc("Kubeconfig for k8s cluster with interLink's VK")
        ]
    )->str:
        from .k8s import create_pod_manifest, submit_job
            
        # created pod manifest
        # pre_exec_cmd = (
        #     "ls /pippo"
        # )
        gpus_per_node = 4 
        pre_exec_cmd = (
            "export CONTAINER_PATH=itwinai_dist_test.sif "
            # # Some garbage to try to get away from tests...
            # "&& cd /tmp && mkdir -p 1234567-itwinai && cd 1234567-itwinai "
            
            f"&& singularity pull --force $CONTAINER_PATH docker://{self.full_name} "
            "&& singularity exec $CONTAINER_PATH cat /app/tests/torch/slurm.vega.sh > slurm.vega.sh "
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
            "slurm-job.vk.io/flags": f"-p gpu --gres=gpu:{gpus_per_node} --ntasks-per-node=1 --nodes=1 --time=00:10:00",
            "slurm-job.vk.io/pre-exec": f" {pre_exec_cmd} || export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1"
        }
        image_path = "/ceph/hpc/data/st2301-itwin-users/cern/hello-world-image.sif"
        cmd_args = [
            "sleep 10 && exit $PRE_EXEC_RETURN_CODE"
            ]
        pod_manifest = create_pod_manifest(
            annotations=annotations,
            image_path=image_path,
            cmd_args=cmd_args
        )
        
        # submit pod
        kubeconfig_str=await kubeconfig_str.plaintext()
        status = submit_job(
            kubeconfig_str=kubeconfig_str,
            pod_manifest=pod_manifest,
            verbose=False
        )

        if status not in ["Succeeded", "Completed"]:
            raise RuntimeError(f"Pod did not complete successfully! Status: {status}")
        
        return f"Pod finished with status: {status}"

    @function
    async def test_n_publish(
        self,
        kubeconfig_str: Annotated[
            dagger.Secret, 
            Doc("Kubeconfig for k8s cluster with interLink's VK")
        ],
        production: bool = False
        )->None:
        # TODO: use annotated for args
        # TODO: adapt to support also TF
        """Pipeline to test container and push it, including both local
        tests and tests on HPC via interLink.

        Args:
            production (bool, optional): whether to push the final image
                to the production image name or not. Defaults to False.
        """
        
        # Test locally
        await self.test_local()
        
        # Publish to registry with random hash
        await self.publish()
        
        # Test on HPC with 
        # await self.test_hpc(kubeconfig_str=kubeconfig_str)
        
        # Publish to registry with final hash
        itwinai_version = (await (
            self.container
            .with_exec([
                "bash", "-c", 'pip list | grep -w "itwinai" | head -n 1 | tr -s " " | cut -d " " -f2 '
                ])
            .stdout()
        )).strip()
        torch_version = (await (
            self.container
            .with_exec([
                "bash", "-c", 'pip list | grep -w "torch[^-]" | head -n 1 | tr -s " " | cut -d " " -f2 | cut -f1,2 -d .'
                ])
            .stdout()
        )).strip()
        ubuntu_codename = (await (
            self.container
            .with_exec([
                "bash", "-c", 'cat /etc/*-release | grep -w DISTRIB_CODENAME | head -n 1 | cut -d = -f2'
                ])
            .stdout()
        )).strip()
        tag = f"{itwinai_version}-torch{torch_version}-{ubuntu_codename}"
        image = "itwinai" if production else "itwinai-dev"
        await self.publish(name=image, tag=tag)
        
        
        