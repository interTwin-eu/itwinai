workflow-runner-cern: environment-cern.yml
	mamba env create -p ./.venv --file environment-cern.yml

# Development environment
dev-env: dev-env.yml
	mamba env create -p ./.venv-dev --file dev-env.yml
	conda run -p ./.venv-dev python -m pip install --no-deps -e ./ai

# Create pytorch env under ./ai/ folder
ai-env: ai/pytorch-env.yml
	mamba env create -p ./ai/.venv-pytorch --file ai/pytorch-env.yml
	conda run -p ./ai/.venv-pytorch python -m pip install --no-deps -e ./ai