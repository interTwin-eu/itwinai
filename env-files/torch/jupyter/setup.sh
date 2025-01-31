#!/bin/bash
set -e
python /opt/setup-rucio-jupyterlab/configure.py

# Creation of the rucio.cfg file
mkdir -p /certs /tmp;
echo -n $RUCIO_ACCESS_TOKEN > /tmp/rucio_oauth.token;
# mkdir -p /opt/rucio/etc;
# echo "[client]" >> /opt/rucio/etc/rucio.cfg;
# echo "rucio_host = https://rucio-intertwin-testbed.desy.de" >> /opt/rucio/etc/rucio.cfg;
# echo "auth_host = https://rucio-intertwin-testbed-auth.desy.de" >> /opt/rucio/etc/rucio.cfg;
# #echo "ca_cert = /certs/rucio_ca.pem" >> /opt/rucio/etc/rucio.cfg;
# echo "ca_cert = /opt/conda/lib/python3.9/site-packages/certifi/cacert.pem" >> /opt/rucio/etc/rucio.cfg;
# echo "account = $JUPYTERHUB_USER" >> /opt/rucio/etc/rucio.cfg;
# echo "auth_type = oidc" >> /opt/rucio/etc/rucio.cfg;
# echo "oidc_audience = rucio-testbed" >> /opt/rucio/etc/rucio.cfg;
# echo "oidc_polling = true" >> /opt/rucio/etc/rucio.cfg;
# echo "oidc_scope = openid profile offline_access eduperson_entitlement" >> /opt/rucio/etc/rucio.cfg;
# echo "auth_token_file_path = /tmp/rucio_oauth.token" >> /opt/rucio/etc/rucio.cfg;

exec "$@"