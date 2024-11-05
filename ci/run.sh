# Example of running dagger pipeline of build and local test for torch container
dagger call \
    build-torch --context=.. --dockerfile=../env-files/torch/Dockerfile \
    test-torch