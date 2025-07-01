Introduction
============

Each use case comes with their own tutorial on how to run it. Before running them,
however, you should set up a Python virtual environment.

After installing and activating the virtual environment, you will want to install the
use-case specific dependencies, if applicable. This can be done by first ``cd``-ing
into the use-case directory and then installing the requirements, as follows

.. code-block:: bash

   cd use-cases/<name-of-use-case>
   pip install -r requirements.txt


Alternatively, you can use the use-case Docker image, if available. After setting
everything up, you can now run the use case as specified in the use case's tutorial.
