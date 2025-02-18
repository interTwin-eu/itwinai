Tutorial for Using the uv Package Manager
=========================================

`uv <https://docs.astral.sh/uv/>`_ is a Python package manager meant to act as a drop-in
replacement for ``pip`` (and many more tools). In this project, we use it to manage our
packages, similar to how ``poetry`` works. This is done using a lockfile called
``uv.lock``.

The reasons for choosing ``uv`` over something like ``poetry`` are multiple:

* ``uv`` tries to replace more than just ``poetry``, but also ``pyenv``, ``pip``, 
  ``virtualenv`` and more. Thus, we can use a single, unified tool for most of our
  Python needs, similar to ``cargo`` in Rust. 

* ``uv`` is much faster than other package managers, providing a speed boost of 10-100
  times compared to ``pip``. 

uv as a Drop-In Replacement for pip
-----------------------------------

``uv`` is a lot faster than ``pip``, so we recommend installing packages from ``PyPI``
with ``uv pip install <package>`` instead of ``pip install <package>``. Since ``uv``
works as a drop-in replacement for ``pip``, you can use this feature to speedup any
installation without changing the setup of your project.

uv as a Project-Wide Package Manager
------------------------------------

If you wish to use the ``uv sync`` and/or ``uv lock`` commands, which is how you use ``uv``
to manage all your project packages, then note that these commands will only work
with the directory called ``.venv`` in the project directory.	This can occasionally be
inconvenient, especially with an existing virtual environment, so we recommend using a 
`symlink <https://en.wikipedia.org/wiki/Symbolic_link>`_.	If you need to switch between
multiple virtual environments, you can update the symlink to point to the desired one.

Using Symlinks to Manage Multiple venvs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a symlink between your venv and the ``.venv`` directory, you can use the
following command:

.. code-block:: bash

    ln -s <path/to/your_venv> <path/to/.venv>

As an example, if I am in the ``itwinai/`` folder and my venv is called ``envAI_juwels``,
the following will create the desired symlink:

.. code-block:: bash

    ln -s envAI_juwels .venv

Installing from the uv.lock File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   If ``uv`` creates your venv for you, the venv will not have ``pip`` installed.
   However, ``pip``is required to be able to run the installation scripts for
   ``Horovod`` and ``DeepSpeed``, so we have included it as a dependency in the
   ``pyproject.toml``.

To install from the ``uv.lock`` file into the ``.venv`` venv, you can do the following:

.. code-block:: bash

    uv sync

Optional dependencies listed in the ``uv.lock`` file (e.g., ``torch``, ``tf``, and dev
in itwinai) can be included with the --extra flag as shown below:

.. code-block:: bash

    uv sync --extra torch --extra dev

These "extras" correspond to the optional dependencies in the ``pyproject.toml``. In
particular, if you are a developer you would use one of the following two commands. If
you are on HPC with CUDA, you would use:

.. code-block:: bash

    uv sync --no-cache --extra dev --extra torch --extra tf 

If you are developing on your local computer, then you would use:

.. code-block:: bash

    uv sync --extra torch --extra tf --extra dev

Updating the uv.lock File
~~~~~~~~~~~~~~~~~~~~~~~~~~

To update the project's ``uv.lock`` file with the dependencies of the project, you can
use the command:

.. code-block:: bash

    uv lock

This will create a ``uv.lock`` file if it doesn't already exist, using the dependencies
from the ``pyproject.toml``.

Adding New Packages to the Project
----------------------------------

To add a new package to the project (i.e. to the ``pyproject.toml`` file) with ``uv``, you
can use the following command:

.. code-block:: bash

    uv add <package>

if you want to add a package to a specific optional dependency, then you can add the
``--optional <name-of-extra>`` flag:

.. code-block:: bash

    uv add <package> --optional <extra>

As an example, if I wanted to add ``numpy`` as an optional dependency under the
dependency group ``torch``, then I would do the following:

.. code-block:: bash

    uv add numpy --optional torch

This has the advantage that ``uv`` will run its dependency solver, meaning it will find
suitable version constraints that fit with the other packages in the ``pyproject.toml``. 

.. warning::

   This will add the package to your ``.venv`` venv, so make sure to have symlinked to
   this directory if you haven't already.
