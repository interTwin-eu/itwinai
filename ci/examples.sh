#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Example of running dagger pipelines -- this script is mostly a scratchpad

# Build and run local tests (no HPC required)
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-local
# Build container with additional requirements
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="REQUIREMENTS=env-files/torch/requirements/cmcc-requirements.txt" \
    test-local

# Build and publish
dagger call --name="$(git rev-parse --verify HEAD)" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish

# Pipeline method: build, test local, push, test remote, and push (publish)
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="nvcr.io/nvidia/pytorch:24.05-py3"
export BASE_IMG_DIGEST="$(docker pull $BASE_IMG_NAME > /dev/null 2>&1 && docker inspect $BASE_IMG_NAME --format='{{index .RepoDigests 0}}' | awk -F'@' '{print $2}')"
dagger call --name="${COMMIT_HASH}-torch" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='${itwinai_version}-torch${framework_version}-${os_version}'

# Open teminal in newly created container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    terminal


############## SLIM ###############
# Build container
dagger call --name="$(git rev-parse --verify HEAD)"  \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    test-local

# Test on HPC and publish
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(docker pull $BASE_IMG_NAME > /dev/null 2>&1 && docker inspect $BASE_IMG_NAME --format='{{index .RepoDigests 0}}' | awk -F'@' '{print $2}')"
dagger call --name="${COMMIT_HASH}-torch-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='${itwinai_version}-slim-torch${framework_version}-${os_version}'


# Convert to singularity
dagger call --name="${COMMIT_HASH}-torch-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    singularity --src-container "python:3.12" \
    export --path my_container.sif