# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""constants used in the itwinai project"""
from pathlib import Path

# Directory names for logging and profiling data
PROFILER_TRACES_DIR_NAME = "profiler-traces"

# mlflow
RELATIVE_MLFLOW_PATH = Path("mllogs/mlflow")
BASE_EXP_NAME: str = "unnamed-experiment"
PROFILING_AVG_NAME: str = "torch_profiling_averages"

adjectives = [
    "quantum",
    "relativistic",
    "wavy",
    "entangled",
    "chiral",
    "tachyonic",
    "superluminal",
    "anomalous",
    "hypercharged",
    "fermionic",
    "hadronic",
    "quarky",
    "holographic",
    "dark",
    "force-sensitive",
    "chaotic",
]

names = [
    "neutrino",
    "graviton",
    "muon",
    "gluon",
    "tachyon",
    "quasar",
    "pulsar",
    "blazar",
    "meson",
    "boson",
    "hyperon",
    "starlord",
    "groot",
    "rocket",
    "yoda",
    "skywalker",
    "sithlord",
    "midichlorian",
    "womp-rat",
    "beskar",
    "mandalorian",
    "ewok",
    "vibranium",
    "nova",
    "gamora",
    "drax",
    "ronan",
    "thanos",
    "cosmo",
]
