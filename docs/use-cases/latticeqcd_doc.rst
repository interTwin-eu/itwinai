Normalizing flow for generating lattice field configurations (Lattice QCD, ETHZ/CSIC)
=====================================================================================

The code is adapted from `this notebook <https://github.com/jkomijani/normflow_>`_ from the Lattice QCD use-case.

More information on the use-case is available in the published deliverables,
`D4.2 <https://zenodo.org/records/10417138>`_,
`D7.2 <https://zenodo.org/records/10417161>`_ and `D7.4 <https://zenodo.org/records/10224277>`_.

Environment setup
-----------------
To set up the environment, please first refer to the `itwinai installation instructions
<https://itwinai.readthedocs.io/latest/installation/developer_installation.html>`_.
For ``itwinai``, using the ``uv`` project management tool is recommended.
A `comprehensive tutorial <https://itwinai.readthedocs.io/latest/installation/uv_tutorial.html>`_
is provided in the itwinai documentation.

With ``uv``, the ``normflow`` repository (located under ``use-cases/lattice-qcd``) can be installed
using the ``--extra lattice-qcd`` flag. For example, to install the normflow environment in itwinai with
Torch support in development mode, run:

.. code-block:: bash

   uv sync --no-cache --extra dev --extra torch --extra lattice-qcd

This will install all required dependencies for both ``itwinai`` and ``normflow``, including
Torch and optional developer tools.

About the use-case and integration
----------------------------------
.. include:: ../../use-cases/lattice-qcd/README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- sphinx-start -->
   :end-before: <!-- sphinx-end -->
