# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import dagger
from dagger import enum_type


@enum_type
class Stage(dagger.Enum):
    """Image distribution stage"""

    DEV = "DEV", "Pushed as itwinai-dev"
    PRODUCTION = "PRODUCTION", "Pushed as itwinai"
    CVMFS = "CVMFS", "Pushed as itwinai-cvmfs"


@enum_type
class MLFramework(dagger.Enum):
    TORCH = "TORCH"
    TENSORFLOW = "TENSORFLOW"
