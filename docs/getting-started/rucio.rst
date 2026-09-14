.. _itwinai_rucio_usage:

Using RUCIO with itwinai
========================

itwinai provides pre-built container images to facilitate the deployment and scaling of machine
learning applications. If you need to use `RUCIO <https://rucio.cern.ch/>`_ for your use case,
you can use the images built from Dockerfiles ``env-files/torch/slim.Dockerfile``, or 
``env-files/torch/jupyter/slim.Dockerfile`` with JupyterLab support.

These images provide the complete itwinai functionality in addition to RUCIO and associated 
download capabilites. 

RUCIO short guide
-----------------

When using the image mentioned above, the following steps are necessary before using RUCIO.

1. **Create a RUCIO account**: contact the relevant RUCIO administrator in your project to set
   up your RUCIO account.

2. **Create a** ``rucio.cfg`` **file**: it must contain at least the following fields (see
   ``env-files/torch/rucio.cfg`` for a template, and replace ``account`` with your own RUCIO
   account name):

   .. code-block:: ini

       [client]
       rucio_host = https://ri-scale-server.rucioit.cern.ch
       auth_host = https://ri-scale-server.rucioit.cern.ch
       auth_type = oidc
       account = your-account-name
       oidc_scope = openid profile eduperson_entitlement offline_access
       oidc_issuer = https://aai-dev.egi.eu/auth/realms/egi

3. **Follow the OIDC authentication flow**: the first RUCIO command prints a URL that you have
   to open in your browser to complete the authentication:

   .. code-block:: bash

       rucio whoami

4. **Download the necessary datasets**:

   .. code-block:: bash

       rucio download <scope>:<name>
