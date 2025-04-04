# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Dagger module for itwinai CI.

This module provides logic to build containers, run tests with pytest, and more.

Since itwinai is designed for HPC deployment, the containers need to be tested on relevant
computing environments with hardware (e.g., GPUs) and software (e.g. CUDA) not accessible
in standard GitHub actions VMs. Through an in-pipeline deployment of interLink, we can
offload some tests to run on HPC.

By deploying interLink within the CI pipeline, some tests can be offloaded to run on HPC.

Additionally, since HPC systems prefer Singularity/Apptainer images over Docker, this
module enables the conversion and publication of Docker containers as SIF files.

Two CI pipelines are provided: a development pipeline, which is simpler and does not
run tests on HPC, and a release pipeline, where containers undergo thorough testing on
HPC, are converted to Singularity, and are pushed to both Docker and Singularity
container registries.
"""

from .main import Itwinai as Itwinai

__all__ = ["Itwinai"]
