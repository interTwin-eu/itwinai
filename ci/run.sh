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
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish

# Pipeline method: build, test local, push, test remote, and push (publish)
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-n-publish --kubeconfig-str=env:KUBECONFIG_STR

# Open teminal in newly created container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    terminal