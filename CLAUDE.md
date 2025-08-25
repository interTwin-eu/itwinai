# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About itwinai

itwinai is a Python toolkit designed for AI and machine learning workflows in digital twin applications.
It's developed by CERN in collaboration with Forschungszentrum Jülich (FZJ) and provides distributed
training, hyperparameter optimization on HPC systems, and integrated ML logging capabilities.

## Development Commands

### Installation and Environment Setup

- **Developer installation**: `uv pip install -e ".[dev,torch]"` (editable install with dev dependencies and PyTorch support)
- **PyTorch environment**: `make torch-env` (GPU support) or `make torch-env-cpu` (CPU only)
- **TensorFlow environment**: `make tensorflow-env` (GPU support) or `make tensorflow-env-cpu` (CPU only) - rarely used
- **Documentation environment**: `make docs-env-jsc` (JSC-specific)

### Testing

- **Run all tests**: `make test` or `.venv/bin/pytest -v tests/`
- **Local tests (skip HPC)**: `make test-local` or `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/pytest -v tests/ -m "not hpc"`
- **JSC-specific tests**: `make test-jsc`
- **Functional tests**: `pytest -v tests/ -m "functional"`

### Code Quality

- **Linting**: `ruff check src/` (configured in pyproject.toml)
- **Formatting**: `ruff format src/`
- **Type checking**: Uses pyright (configured in pyproject.toml)

### CLI Usage

- **Main CLI entry point**: `itwinai --help`
- **Execute pipeline**: `itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline`
- **Generate flamegraph**: `itwinai generate-flamegraph --file profiling_data.out`
- **SLURM script generation**: `itwinai generate-slurm -c slurm_config.yaml`

## Architecture Overview

### Core Components (`src/itwinai/`)

**Pipeline System**: The framework is built around a modular component system where ML workflows are
composed of `BaseComponent` objects arranged in `Pipeline` sequences.

- `components.py`: Base classes for all workflow components (`BaseComponent`, `Trainer`, etc.)
- `pipeline.py`: `Pipeline` class that executes components sequentially, passing outputs to inputs
- `parser.py`: Configuration parsing for YAML-defined pipelines

**Distributed Training**: Multi-framework support for distributed ML training.

- `distributed.py`: Environment detection and distributed setup utilities
- `torch/trainer.py`: PyTorch trainer with DDP, DeepSpeed, Horovod support
- `torch/distributed.py`: PyTorch-specific distributed strategies
- `tensorflow/trainer.py`: TensorFlow distributed training support

**HPC Integration**: Native SLURM support for running on supercomputing clusters.

- `slurm/`: SLURM job script generation and submission
- `slurm/slurm_script_builder.py`: Automated SLURM script creation
- `slurm/slurm_script_configuration.py`: Configuration management for different HPC systems

**Logging and Monitoring**: Comprehensive experiment tracking and profiling.

- `loggers.py`: MLFlow, WandB, console logging implementations
- `torch/monitoring/`: GPU utilization and performance monitoring
- `torch/profiling/`: PyTorch profiler integration and py-spy aggregation

### Configuration System

The framework uses **Hydra + OmegaConf** for configuration management:

- YAML configs define entire pipelines using `_target_` for class instantiation
- Dynamic parameter overrides via CLI: `--override param.value=new_value`
- Environment-specific resolvers: `${itwinai.cwd:}` for current working directory
- SLURM configs separate from training configs for HPC deployment

### Key Patterns

**Component-Based Architecture**: All workflow steps inherit from `BaseComponent` and implement an
`execute()` method. Components can be chained in pipelines or used standalone.

**Multi-Framework Support**: The same pipeline definition works with PyTorch or TensorFlow by
swapping trainer components.

**HPC-First Design**: Built for supercomputing environments with automatic SLURM integration,
multi-node scaling, and HPC-specific optimizations.

**CLI Architecture Pattern**: The CLI uses a hybrid approach mixing Typer, ArgumentParser, OmegaConf, and Hydra:

- **Typer function signatures**: Define arguments purely for generating nice help text (`itwinai command --help`)
- **Internal ArgumentParser**: Performs the actual argument parsing within command functions
- **Rationale**: This allows leveraging Typer's excellent help generation while maintaining the flexibility of ArgumentParser for complex parsing logic that integrates with OmegaConf/Hydra

Example: The `run` command defines `submit_job` and `save_script` in both the Typer signature (for help text) and creates an internal ArgumentParser that actually processes these flags. The parsed values override configuration file settings.

## Working with Use Cases

The `use-cases/` directory contains complete examples:

- `mnist/torch/`: PyTorch MNIST training with distributed support
- `3dgan/`: Generative adversarial network for 3D data
- `virgo/`: Gravitational wave detection
- `eurac/`: Weather forecasting models
- `lattice-qcd/`: Quantum chromodynamics simulations

Each use case includes:

- `config.yaml`: Pipeline configuration
- `slurm_config.yaml`: HPC job configuration  
- `trainer.py`: Custom trainer implementation
- `dataloader.py`: Data loading logic

## Plugin System

itwinai supports plugins for extending functionality:

- Plugins installed via `uv pip install git+<plugin-repo>`
- Declared in configs: `plugins: [git+https://github.com/user/itwinai-plugin]`
- Plugins can add new components, trainers, or data loaders

## Testing Strategy

Tests are organized by component type:

- `tests/components/`: Core pipeline and component tests
- `tests/torch/`: PyTorch-specific tests with distributed markers
- `tests/use-cases/`: Integration tests for complete workflows
- Markers for different test types: `integration`, `hpc`, `torch_dist`, `functional`

## Environment Variables

- `ITWINAI_LOG_LEVEL`: Control logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `TORCH_ENV`: Name of PyTorch environment for testing
- `TF_ENV`: Name of TensorFlow environment for testing

## Development Workflow

1. Install in editable mode: `uv pip install -e ".[dev,torch]"`
2. Create virtual environments for frameworks: `make torch-env` and/or `make tensorflow-env`
3. Run tests locally: `make test-local`
4. Use `ruff` for code formatting and linting
5. Test with real use cases in `use-cases/` directory
6. For HPC development, test on JSC systems using `make test-jsc`
