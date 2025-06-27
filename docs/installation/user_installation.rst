User Installation (for Non-Developers)
======================================

This guide provides step-by-step instructions for installing the ``itwinai`` library for
users.

.. The explanation for creating a venv is the same for developers and users
.. include:: ./software_prerequisites.rst


Installing the ``itwinai`` Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can choose if you want to install ``itwinai`` with support for either PyTorch or
TensorFlow by using extras:

.. tab-set:: 

    .. tab-item:: PyTorch

        To install ``itwinai`` with PyTorch without GPU acceleration, you can use the
        following command:

        .. code-block:: bash
            
            uv pip install "itwinai[torch]"

        To enable GPU acceleration, you can use the following command:

        .. code-block:: bash

            uv pip install ".[torch]" \
                --extra-index-url https://download.pytorch.org/whl/cu121


    .. tab-item:: TensorFlow

        To install ``itwinai`` with TensorFlow without GPU acceleration, you can use the
        following command:

        .. code-block:: bash
            
            uv pip install "itwinai[tf]"

        To enable GPU acceleration, you can use the following command:

        .. code-block:: bash

            uv pip install ".[tf-cuda]"

.. Explanation for installing horovod, DS, and other packages that need to be installed AFTER itwinai
.. include:: ./post_itwinai_installation.rst
