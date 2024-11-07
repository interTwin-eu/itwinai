# Example of running dagger pipeline of build and local test for torch container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-local

# Build, test local, and publish
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish

# Build, test local, publish, and test remote
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-remote --kubeconfig-str "$KUBECONFIG_STR"