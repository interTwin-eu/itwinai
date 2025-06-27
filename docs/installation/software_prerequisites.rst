Setting up the system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First of all, before installing itwinai and its Python dependencies let's make sure that the
system dependencies such as CUDA drivers, compilers, and MPI libraries, are correctly set up.

Supported OSs are Linux and macOS.

.. warning::

   On high-performance computing (HPC) systems, **you must load the appropriate modules
   before creating or activating your Python virtual environment** to ensure compatibility with
   system libraries. 

.. include:: ./hpc_modules.rst


Creating a Python Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The suggested way of managing Python dependencies, including itwinai, is through Python virtual
environments. Creating a virtual environment is allows to isolate dependencies and prevent
conflicts with other Python projects.

Beware that some HPC centers advise against using Python virtual environments as they create a
large amount of files, which can clog some distributed filesystems. In such situation, you
should prefer using containers.

To manage python virtual environments we use UV, which can be installed from
`this page <https://docs.astral.sh/uv/getting-started/installation/>`_. Learn more on UV
package manager from our `UV tutorial <uv_tutorial.rst>`_

If you don't already have a virtual environment, you can create one with the following
command:

.. code-block:: bash

   # Remember to load the software modules first (see section above)!
   
   uv venv

   # Alternatively to the command above, if you just want to use plain pip instead of UV
   python -m venv .venv

Notice that a new directory called ``.venv`` is created to contain your virtual
environment. Now, you can start your virtual environment with the following command: 

.. code-block:: bash 

   # Remember to load the software modules first (see section above)!

   source .venv/bin/activate
