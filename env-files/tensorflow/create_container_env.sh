#!/bin/bash

# Install dependencies in container, assuming that the container image
# is from NGC and tensorflow is already installed

pip install --upgrade pip

# itwinai
pip3 install -e .[dev]