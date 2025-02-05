Creating a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While not mandatory, creating a virtual environment is highly recommended to isolate
dependencies and prevent conflicts with other Python projects.

.. warning::

    On high-performance computing (HPC) systems, you must load the appropriate modules
    before activating your virtual environment to ensure compatibility with system
    libraries. See the dropdown below for detailed instructions:

    .. dropdown:: HPC Systems

       .. include:: ./hpc_modules.rst


If you don't already have a virtual environment, you can create one with the following
command:

.. code-block:: bash 

   python -m venv <name-of-venv>

Remember to replace ``<name-of-venv>`` with the name you want for your virtual
environment. Now, you can start your virtual environment with the following command: 

.. code-block:: bash 

   source <name-of-venv>/bin/activate
