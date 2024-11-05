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

import dagger
from dagger import dag, function, object_type, Doc


@object_type
class Itwinai:

    torch_container: Optional[dagger.Container] = dataclasses.field(default=None, init=False)

    @function
    def build_torch(
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
        """Build itwinai torch container image from existing Dockerfile"""
        self.torch_container = (
            dag.container()
            .build(context=context, dockerfile=dockerfile)
        )
        return self

    @function
    async def test_torch(self) -> str:
        """Test itwinai torch container image with pytest on non-HPC environments."""
        test_cmd = [
            "pytest",
            "-v",
            "-m",
            "not hpc",
            "tests"
        ]
        return await (
            self.torch_container
            .with_exec(test_cmd)
            .stdout()
        )
