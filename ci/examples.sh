#!/bin/bash
# Example of running dagger pipelines

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
dagger call --unique-id="$(git rev-parse --verify HEAD)" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish

# Pipeline method: build, test local, push, test remote, and push (publish)
export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="nvcr.io/nvidia/pytorch:24.05-py3"
export BASE_IMG_DIGEST="$(docker pull $BASE_IMG_NAME > /dev/null 2>&1 && docker inspect $BASE_IMG_NAME --format='{{index .RepoDigests 0}}' | awk -F'@' '{print $2}')"
dagger call --unique-id="${COMMIT_HASH}_torch" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='${itwinai_version}-torch${framework_version}-${os_version}'

# Open teminal in newly created container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    terminal


############## SLIM ###############
# Build container with additional requirements
dagger call --unique-id="$(git rev-parse --verify HEAD)"  \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    test-local

export COMMIT_HASH=$(git rev-parse --verify HEAD)
export BASE_IMG_NAME="python:3.10-slim"
export BASE_IMG_DIGEST="$(docker pull $BASE_IMG_NAME > /dev/null 2>&1 && docker inspect $BASE_IMG_NAME --format='{{index .RepoDigests 0}}' | awk -F'@' '{print $2}')"
dagger call --unique-id="${COMMIT_HASH}_torch_slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
        --build-args="COMMIT_HASH=$COMMIT_HASH,BASE_IMG_NAME=$BASE_IMG_NAME,BASE_IMG_DIGEST=$BASE_IMG_DIGEST" \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='${itwinai_version}-slim-torch${framework_version}-${os_version}'