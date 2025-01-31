# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Framework-independent types."""


class MLArtifact:
    """A framework-independent machine learning artifact."""


class MLDataset(MLArtifact):
    """A framework-independent machine learning dataset."""


class MLModel(MLArtifact):
    """A framework-independent machine learning model."""
