# Launch some workflows

# MNIST training
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

# MNIST inference
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml
