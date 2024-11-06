# Example of running dagger pipeline of build and local test for torch container
dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-local

dagger call \
    build-container --context=.. --dockerfile=../env-files/torch/Dockerfile \
    publish