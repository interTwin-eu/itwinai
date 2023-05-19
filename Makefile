workflow-runner:
	mamba env create -p ./.venv --file environment-cern.yml

dev-env:
	mamba env create -p ./.venv-dev --file dev-env.yml
	conda run -p ./.venv-dev python -m pip install --no-deps -e ./ai
