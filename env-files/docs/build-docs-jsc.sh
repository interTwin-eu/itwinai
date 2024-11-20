#!/bin/bash

# Build the documentation locally and serve it on localhost on JSC systems

ml --force purge
ml Stages/2023  GCCcore/.11.3.0 Python/3.10.4 Pandoc/2.19.2

source .venv-docs/bin/activate
cd docs
make clean && make html && python -m http.server -d _build/html