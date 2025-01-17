#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# shellcheck disable=all

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
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
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
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
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

############## JUPYTER (SLIM) ###############

# Build and test locally
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="jupyter/scipy-notebook:python-3.10.11"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call --name="${COMMIT_HASH}-torch-jlab-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-local

# Build and publish
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="jupyter/scipy-notebook:python-3.10.11"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call --name="${COMMIT_HASH}-torch-jlab-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='jlab-slim-${itwinai_version}-torch${framework_version}-${os_version}'

############## JUPYTER ###############

# Build and test locally
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="jupyter/scipy-notebook:python-3.10.11"
export BASE_IMG_DIGEST=$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')
dagger call --name="${COMMIT_HASH}-torch-jlab" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-local