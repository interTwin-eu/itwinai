#!/bin/bash
# Example of running dagger pipelines

# Build and run local tests (no HPC required)
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-local
# Build container with additional requirements
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile --additional-requirements env-files/torch/requirements/cmcc-requirements.txt \
    test-local

# Build and publish
dagger call --commit-id="$(git rev-parse --verify HEAD)" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish

# Pipeline method: build, test local, push, test remote, and push (publish)
dagger call --commit-id="$(git rev-parse --verify HEAD)" \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH

# Open teminal in newly created container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    terminal


############## SLIM ###############
# Build container with additional requirements
dagger call --commit-id="$(git rev-parse --verify HEAD)"  \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    test-local

dagger call --commit-id="$(git rev-parse --verify HEAD)_slim" \
    build-container --context=.. --dockerfile=../env-files/torch/slim.Dockerfile \
    test-n-publish --kubeconfig=env:KUBECONFIG_STR --stage=DEV --framework=TORCH \
    --tag-template='${itwinai_version}-slim-torch${framework_version}-${ubuntu_codename}'