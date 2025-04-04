# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import importlib
from typing import List

from .exceptions import SanityCheckError

core_modules = [
    "itwinai",
    "itwinai.cli",
    "itwinai.components",
    "itwinai.distributed",
    "itwinai.loggers",
    "itwinai.parser",
    "itwinai.pipeline",
    "itwinai.serialization",
    "itwinai.type",
    "itwinai.utils",
    "itwinai.tests",
    "itwinai.tests.dummy_components",
    "itwinai.tests.exceptions",
    "itwinai.tests.sanity_check",
    "itwinai.plugins",
    "itwinai.scalability_report",
    "itwinai.scalability_report.data",
    "itwinai.scalability_report.utils",
    "itwinai.scalability_report.reports",
    "itwinai.scalability_report.plot",
]

torch_modules = [
    "itwinai.torch",
    "itwinai.torch.data",
    "itwinai.torch.gan",
    "itwinai.torch.models",
    "itwinai.torch.models.mnist",
    "itwinai.torch.config",
    "itwinai.torch.distributed",
    "itwinai.torch.inference",
    "itwinai.torch.mlflow",
    "itwinai.torch.reproducibility",
    "itwinai.torch.trainer",
    "itwinai.torch.type",
    "itwinai.torch.loggers",
    "itwinai.torch.tuning",
]

tensorflow_modules = [
    "itwinai.tensorflow",
    "itwinai.tensorflow.data",
    "itwinai.tensorflow.models",
    "itwinai.tensorflow.models.mnist",
    "itwinai.tensorflow.distributed",
    "itwinai.tensorflow.trainer",
    "itwinai.tensorflow.utils",
]


def run_sanity_check(modules: List[str]):
    """Run sanity checks by trying to import modules.

    Args:
        modules (List[str]): list of modules

    Raises:
        SanityCheckError: when some module cannot be imported.
    """
    failed_imports = []

    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ Successfully imported: {module}")
        except ImportError as e:
            failed_imports.append((module, str(e)))
            print(f"❌ Failed to import: {module} - {e}")

    if failed_imports:
        err_msg = "\nSummary of failed imports:\n"
        for module, error in failed_imports:
            err_msg += f"Module: {module}, Error: {error}\n"

        raise SanityCheckError(
            "Not all itwinai modules could be successfully imported!\n" + err_msg
        )
    else:
        print("\nAll modules imported successfully!")


def sanity_check_slim():
    """Run sanity check on the installation
    of core modules of itwinai (neither itwinai.torch,
    nor itwinai.tensorflow)."""

    run_sanity_check(modules=core_modules)


def sanity_check_torch():
    """Run sanity check on the installation of itwinai
    for a torch environment."""
    run_sanity_check(modules=core_modules + torch_modules)


def sanity_check_tensorflow():
    """Run sanity check on the installation of itwinai
    for a tensorflow environment."""
    run_sanity_check(modules=core_modules + tensorflow_modules)


def sanity_check_all():
    """Run all sanity checks."""
    run_sanity_check(modules=core_modules + torch_modules + tensorflow_modules)
