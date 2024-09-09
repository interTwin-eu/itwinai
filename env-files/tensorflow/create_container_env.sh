#!/bin/bash

# Install dependencies in container, assuming that the container image
# is from NGC and tensorflow is already installed

pip install --no-cache-dir --upgrade pip

# WHEN USING TF >= 2.16:
# install legacy version of keras (2.16)
# Since TF 2.16, keras updated to 3.3,
# which leads to an error when more than 1 node is used
# https://keras.io/getting_started/
pip install --no-cache-dir tf_keras==2.16.* || exit 1

# Install Pov4ML
pip install "prov4ml[linux]@git+https://github.com/matbun/ProvML@main" || exit 1

# itwinai
pip --no-cache-dir install . || exit 1
