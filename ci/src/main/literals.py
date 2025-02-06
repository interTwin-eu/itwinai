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
class MLFramework(dagger.Enum):
    TORCH = "TORCH"
    TENSORFLOW = "TENSORFLOW"
