# AI workflows (T6.5)

## Development env setup

Requirements:

- Mamba / conda

Installation:

```bash
# Optional
cd ./ai
# Inside ./ai folder
mamba env create -p ./.venv --file ./dev-env.yml
# Install this package
conda run -p ./.venv python -m pip install --no-deps -e .

conda activate ./.venv
```
