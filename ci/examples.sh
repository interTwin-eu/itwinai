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

############## EXAMPLE OF WORKFLOWS ###############

# IMPORTANT: when chaining steps that retrun Self, the chain is not really executed because
# of lazy initialization of the steps. A workaround is appending the `logs` fuction at the
# end of the pipeline, which prints the logs and "forces" the pipeline to be executed
# step-by-step.

# Build a container and open a terminal in it
dagger call \
    build-container ... \
    container \
    terminal

# Build a container, test it "locally" (non-HPC), and see the logs
dagger call \
    build-container ... \
    test-local \
    logs

# Build a container, publish it to some containers registry, and see the logs
dagger call \
    build-container ... \
    publish ... \
    logs

# Build a container, test it, publish it to some containers registry, and see the logs
dagger call \
    build-container ... \
    test-local \
    publish ... \
    logs

# Build a container, test it, publish it to some containers registry, test it on HPC,
# publish it again with another name... and see the whole logs
dagger call \
    build-container ... \
    test-local \
    publish ... \
    test-hpc ... \
    publish ... \
    logs

# Or more simply, run existing end-to-end pipeline
dagger call \
    build-container ... \
    release-pipeline ...

# You can also do all the above, but starting from an exisiting itwinai container from
# some registry. This way you don't have to build it from scratch.
# You can reuse all the functions shown above.

# Open a teminal in an existing itwinai container image
dagger call \
        --container ghcr.io/intertwin-eu/itiwnai:latest \
    container \
    terminal

# Test on HPC an existing itwinai container image
dagger call \
        --container ghcr.io/intertwin-eu/itiwnai:latest \
    test-hpc ... \
    logs

# The Singularity type allows to convert a container to a SIF

# Get terminal in container with only singularity inside
dagger call singularity client terminal

# Convert Docker container to SIF and export it to the local filesystem
dagger call singularity --container ghcr.io/intertwin-eu/itiwnai:latest \
    convert export --path my_container.sif

# Convert and publish the resulting SIF to some registry
dagger call singularity --container ghcr.io/intertwin-eu/itiwnai:latest \
    publish --username ... --password ... --uri oras://registry.cern.ch/itwinai/dev/busybox:latest

# Remove a specific tag from a registry
dagger call singularity \
    remove --username ... \
            --password ... \
            --registry ... \
            --project ... \
            --name ... \
            --tag ...


############## TORCH ###############

# Build and run local tests (no HPC required)
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-local \
    logs
# Build container with additional requirements
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="REQUIREMENTS=env-files/torch/requirements/cmcc-requirements.txt" \
    test-local \
    logs

# Build and publish
dagger call --name="$(git rev-parse --verify HEAD)" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish \
    logs

# Pipeline method: build, test local, push, test remote, and push (publish)
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="nvcr.io/nvidia/pytorch:24.05-py3"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
export KUBERNETES="--kubernetes tcp://localhost:6443" # Set this to empty string to avoid using k8s endpoint
dagger call \
        --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline --values=file:tmp.yaml --framework=TORCH  \
        --tag-template='${itwinai_version}-torch${framework_version}-${os_version}' \
        --password env:SING_PWD --username env:SING_USER


# Development pipeline
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="nvcr.io/nvidia/pytorch:24.05-py3"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    dev-pipeline

# Release pipeline
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="nvcr.io/nvidia/pytorch:24.05-py3"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
export KUBERNETES="--kubernetes tcp://localhost:6443" # Set this to empty string to avoid using k8s endpoint
dagger call \
        --tag "${COMMIT_HASH}-torch" \
        --nickname torch \
        --image itwinai \
        --docker-registry ghcr.io/intertwin-eu \
        --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline --values=file:tmp.yaml --framework=TORCH  \
        --tag-template='${itwinai_version}-torch${framework_version}-${os_version}' \
        --password env:SING_PWD --username env:SING_USER $KUBERNETES

# Open teminal in newly created container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    container \
    terminal


############## TORCH SLIM ###############
# Build container
dagger call --name="$(git rev-parse --verify HEAD)"  \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile --build-arm \
    test-local \
    logs


# Release pipeline
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call \
    --nickname torch-slim \
    --image itwinai-dev \
    --docker-registry ghcr.io/intertwin-eu \
    --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container \
    --context .. \
    --dockerfile ../env-files/torch/slim.Dockerfile \
    --build-args ="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline \
    --values file:tmp.yaml \
    --framework TORCH \
    --tag-template '${itwinai_version}-slim-torch${framework_version}-${os_version}' \
    --password env:SING_PWD \
    --username env:SING_USER \
    --kubernetes tcp://localhost:6443

# Dev pipeline
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call \
    --nickname torch-slim \
    --image itwinai-dev \
    --docker-registry ghcr.io/intertwin-eu \
    --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container \
    --context .. \
    --dockerfile ../env-files/torch/slim.Dockerfile \
    --build-args ="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    dev-pipeline \
    --framework TORCH \
    --tag-template '${itwinai_version}-slim-torch${framework_version}-${os_version}' \
    --password env:SING_PWD \
    --username env:SING_USER

# Test on HPC and publish
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
export KUBERNETES="--kubernetes tcp://localhost:6443" # Set this to empty string to avoid using k8s endpoint
dagger call --name="${COMMIT_HASH}-torch-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline --values=file:tmp.yaml --framework=TORCH $KUBERNETES \
    --tag-template='${itwinai_version}-slim-torch${framework_version}-${os_version}'


# Convert to singularity
dagger call --name="${COMMIT_HASH}-torch-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    singularity --src-container "python:3.12" \
    export --path my_container.sif


#######################################
#######################################


# Release pipeline (but for testing purposes)
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call \
    --nickname torch-slim \
    --image itwinai-dev \
    --tag torch-slim-${COMMIT_HASH} \
    --docker-registry ghcr.io/intertwin-eu \
    --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container \
    --context .. \
    --dockerfile ../env-files/torch/slim.Dockerfile \
    --build-args ="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline \
    --values file:tmp.yaml \
    --framework TORCH \
    --tag-template '${itwinai_version}-slim-torch${framework_version}-${os_version}' \
    --skip-singularity \
    --kubernetes tcp://localhost:6443


#######################################
#######################################



############## TORCH SKINNY ###############

export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call \
    --nickname torch-skinny \
    --image itwinai-dev \
    --docker-registry ghcr.io/intertwin-eu \
    --singularity-registry registry.egi.eu/dev.intertwin.eu \
    build-container \
    --context .. \
    --dockerfile ../env-files/torch/skinny.Dockerfile \
    --build-args ="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline \
    --values file:tmp.yaml \
    --framework TORCH \
    --tag-template '${itwinai_version}-slim-torch${framework_version}-${os_version}' \
    --password env:SING_PWD \
    --username env:SING_USER \
    --skip-singularity


# CI pytest
dagger call \
    build-container \
    --context .. \
    --dockerfile ../env-files/torch/skinny.Dockerfile \
    test-local \
    --cmd "pytest,-v,-n,logical,/app/tests/,-m,not hpc and not tensorflow" \
    logs


############## JUPYTER (SLIM) ###############

# Build and test locally
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.12-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
dagger call --tag="${COMMIT_HASH}-torch-jlab-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-local \
    logs

# Build and publish
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.12-slim"
export BASE_IMG_DIGEST="$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')"
export KUBERNETES="--kubernetes tcp://localhost:6443" # Set this to empty string to avoid using k8s endpoint
dagger call --tag="${COMMIT_HASH}-torch-jlab-slim" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    release-pipeline --values=file:tmp.yaml --framework=TORCH $KUBERNETES \
    --tag-template='jlab-slim-${itwinai_version}-torch${framework_version}-${os_version}'

############## JUPYTER ###############

# Build and test locally
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="jupyter/scipy-notebook:python-3.10.11"
export BASE_IMG_DIGEST=$(echo "$BASE_IMG_NAME" | cut -d ':' -f 1)@$(docker buildx imagetools inspect $BASE_IMG_NAME | grep "Digest:" | head -n 1 | awk '{print $2}')
dagger call --tag="${COMMIT_HASH}-torch-jlab" \
    build-container --context=.. --dockerfile=../env-files/torch/jupyter/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-local \
    logs


############## interLink ###############

# Start service in terminal
dagger call interlink --values=file:tmp.yaml interlink-cluster up

# Access the k8s cluster with interLink VK from terminal
dagger call interlink --values=file:tmp.yaml --kubernetes tcp://localhost:6443 client terminal

# Test interlink offloading mechanism with some toy pods
dagger call interlink --values=file:tmp.yaml --kubernetes tcp://localhost:6443 test-offloading