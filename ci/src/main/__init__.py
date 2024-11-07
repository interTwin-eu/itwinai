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
    ) -> Self:
        """Build itwinai container image from existing Dockerfile"""
        self.container = (
            dag.container()
            .build(context=context, dockerfile=dockerfile)
        )
        return self

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
    ) -> str:
        """Push container to registry"""
        tag = tag if tag else random.randrange(10 ** 8)

        # Test locally
        await self.test_local()
        self.full_name = f"{registry}/{name}:{tag}"
        return await self.container.publish(self.full_name)
    
    @function
    def test_remote(self)->str:
    # def test_remote(self, kubeconfig_str: str)->str:
        from .k8s import create_pod_manifest, submit_job
    
        # created pod manifest
        annotations = {
            "slurm-job.vk.io/flags": "-p gpu --gres=gpu:1 --ntasks-per-node=1 --nodes=1 --time=00:10:00",
            "slurm-job.vk.io/pre-exec": "ls /pippo || export SINGULARITYENV_PRE_EXEC_RETURN_CODE=1"
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
        status = submit_job(
            kubeconfig_str=kubeconfig_str,
            pod_manifest=pod_manifest,
            verbose=True
        )

        if status not in ["Succeeded", "Completed"]:
            raise RuntimeError("Pod failed!")
        
        return f"Pod finished with status: {status}"
        
    
