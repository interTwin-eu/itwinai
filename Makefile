# "Custom" workflow runner environment
workflow-runner-cern: environment-cern.yml
	micromamba env create -p ./.venv --file environment-cern.yml -y

# Development environment
dev-env: dev-env.yml ai/setup.py
	micromamba env create -p ./.venv-dev --file dev-env.yml -y
	micromamba run -p ./.venv-dev python -m pip install -e ./ai

# Create pytorch env under ./ai/ folder
ai-env: ai/env-files/pytorch-lock.yml ai/setup.py
	micromamba env create -p ./ai/.venv-pytorch --file ai/env-files/pytorch-lock.yml -y
	micromamba run -p ./ai/.venv-pytorch python -m pip install -e ./ai

lock-ai: ai/env-files/pytorch-env.yml ai/env-files/pytorch-env-gpu.yml
	@echo "NOTE: Run this command from *whitin* ./.venv where conda-lock is available!"
	# Lock for pytorch CPU
	conda-lock lock -f ai/env-files/pytorch-env.yml --lockfile ai/env-files/pytorch-lock.yml
	# Lock for pytorch GPU
	conda-lock lock -f ai/env-files/pytorch-env-gpu.yml --lockfile ai/env-files/pytorch-gpu-lock.yml
