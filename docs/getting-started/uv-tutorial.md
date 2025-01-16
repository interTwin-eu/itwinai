# Tutorial for using the uv package manager

[uv](https://docs.astral.sh/uv/) is a Python package manager meant to act as a drop-in
replacement for `pip` (and many more tools). In this project, we use it to manage our
packages, similar to how `poetry` works. This is done using a lockfile called
`uv.lock`.

## uv as a drop-in replacement for pip

`uv` is a lot faster than `pip`, so we recommend installing packages from `PyPI`
with `uv pip install <package>` instead of `pip install <package>`. You don't need to
change anything in your project to use this feature, as it works as a drop-in
replacement to `pip`.

## uv as a project-wide package manager

If you wish to use the `uv sync` and/or `uv lock` commands, which is how you use `uv`
to manage all your project packages, then note that these commands will only work
with the directory called `.venv` in the project directory. Sometimes, this can be a
bit annoying, especially with an existing venv, so we recommend using a
[symlink](https://en.wikipedia.org/wiki/Symbolic_link). If you need to have multiple
venvs that you want to switch between, you can update the symlink to whichever of them
you want to use at the moment. For SLURM scripts, you can hardcode them if need be.

### Symlinking .venv

To create a symlink between your venv and the `.venv` directory, you can use the
following command:

```bash
ln -s <path/to/your_venv> <path/to/.venv>
```

As an example, if I am in the `itwinai/` folder and my venv is called `envAI_juwels`,
then the following will create the wanted symlink:

```bash
ln -s envAI_juwels .venv
```

### Installing from uv.lock

> [!Warning]
> If `uv` creates your venv for you, the venv will not contain `pip`. However, you need
> to have `pip` installed to be able to run the installation scripts for `Horovod` and
> `DeepSpeed`, so we have included `pip` in the dependencies in `pyproject.toml`.

To install from the `uv.lock` file into the `.venv` venv, you can do the following:

```bash
uv sync
```

If the `uv.lock` file has optional dependencies (e.g. `macos` or `torch`), then these
can be added with the `--extra` flag as follows:

```bash
uv sync --extra torch --extra macos
```

These will usually correspond to the optional dependencies in the `pyproject.toml`. In
particular, if you are a developer you would use one of the following two commands. If
you are on HPC with cuda, you would use:

```bash
uv sync --no-cache --extra dev --extra nvidia --extra torch --extra tf 
```

If you are developing on your local computer with macOS, then you would use:

```bash
uv sync --extra torch --extra tf --extra dev --extra macos
```

### Updating the uv.lock file

To update the project's `uv.lock` file with the dependencies of the project, you can
use the command:

```bash
uv lock
```

This will create a `uv.lock` file if it doesn't already exist, using the dependencies
from the `pyproject.toml`.

## Adding new packages to the project

To add a new package to the project (i.e. to the `pyproject.toml` file) with `uv`, you
can use the following command:

```bash
uv add <package>
```

> [!Warning]
> This will add the package to your `.venv` venv, so make sure to have symlinked to
> this directory if you haven't already.
