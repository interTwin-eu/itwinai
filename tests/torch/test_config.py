# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from itwinai.torch.config import TrainingConfiguration


def test_values_parsing():
    """Check dynamic override and creation of new entries."""
    cfg = TrainingConfiguration(batch_size="11", param_abc="11", param_xyz=1.1)
    assert cfg.batch_size == 11
    assert cfg.param_abc == "11"
    assert cfg.param_xyz == 1.1
    assert isinstance(cfg.pin_gpu_memory, bool)

    # Check dict-like getitem
    assert cfg["batch_size"] == 11


def test_illegal_override():
    """Test that illegal type override fails."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfiguration(batch_size="hello")
    assert "batch_size" in str(exc_info.value)
