Normalizing flow for generating lattice field configurations (Lattice QCD, ETHZ/CSIC)
=====================================================================================

The code is adapted from `this notebook <https://github.com/jkomijani/normflow_>`_ from the Lattice QCD use-case.

More information on the use-case is available in the published deliverables,
`D4.2 <https://zenodo.org/records/10417138>`_,
`D7.2 <https://zenodo.org/records/10417161>`_ and `D7.4 <https://zenodo.org/records/10224277>`_.

Environment setup
-----------------
In order to setup the environment, please first look at the `itwinai installation instructions
<https://itwinai.readthedocs.io/latest/installation/developer_installation.html>`_. For `itwinai`,
using ``uv`` project management is recommended. A `comprehensive tutorial <https://itwinai.readthedocs.io/latest/installation/uv_tutorial.html>`_ on this is provided in the
itwinai documentation. With ``uv``, the ``normflow`` repository can be installed with the ``--extra``
flag. For instance, for installing normflow environment in itwinai with torch support in development
mode, the command is:
```
uv sync --no-cache  --extra dev --extra torch --extra lattice-qcd
```
This installs all itwinai and normflow dependencies with torch.

About the use-case and integration
----------------------------------
.. include:: ../../use-cases/lattice-qcd/README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- sphinx-start -->
   :end-before: <!-- sphinx-end -->
