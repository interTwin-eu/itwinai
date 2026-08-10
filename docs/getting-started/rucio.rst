Using RUCIO with itwinai 
==============================

itwinai provides pre-built container images to facilitate the deployment and scaling of machine
learning applications. If you need to use RUCIO (``https://rucio.cern.ch/``) for your use-case,
you can take advantage of rucio.Dockerfile to build your container. Alternatively, you can use
an already deployed package on GHCR: 

- **RUCIO + itwinai image** ``https://github.com/users/okrochak/packages/container/package/hypermeteo-downscaling-plugin``

These images have the complete itwinai functionality in addition to RUCIO. 

RUCIO short guide
--------------------------

When using the image mentioned above, the next steps are necessary before using RUCIO. 

- **1. RUCIO account creation**: contact the relevant RUCIO administrator in your project to set 
  up your RUCIO account. 
- **2. Create a `rucio.cfg` file**: A rucio.cfg file is necessary with the following fields
  [client]
  rucio_host = https://ri-scale-server.rucioit.cern.ch
  auth_host = https://ri-scale-server.rucioit.cern.ch
  auth_type = oidc
  account = your-account-name
  oidc_scope = openid profile eduperson_entitlement offline_access
  oidc_issuer = https://aai-dev.egi.eu/auth/realms/egi
- **Follow OIDC authentication**: /opt/conda/envs/rucio/bin/rucio whoami
- **Download the necessary datasets**: /opt/conda/envs/rucio/bin/rucio download <scope>:<name>