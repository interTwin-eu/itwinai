import pytest

from itwinai.components import Adapter
from itwinai.pipeline import Pipeline
from itwinai.tests import (
    FakeGetterExec, FakeSplitterExec, FakeTrainerExec, FakeSaverExec
)


def test_adapter():
    """Test Adapter component."""
    prefix = Adapter.INPUT_PREFIX
    adapter = Adapter(
        policy=[f"{prefix}{3-i}" for i in range(4)]
    )
    result = adapter.execute(0, 1, 2, 3)
    assert result == (3, 2, 1, 0)

    result = adapter.execute(*tuple(range(100)))
    assert result == (3, 2, 1, 0)

    adapter = Adapter(
        policy=[f"{prefix}0" for i in range(4)]
    )
    result = adapter.execute(0, 1, 2, 3)
    assert result == (0, 0, 0, 0)

    adapter = Adapter(
        policy=[f"{prefix}{i%2}" for i in range(4)]
    )
    result = adapter.execute(0, 1, 2, 3)
    assert result == (0, 1, 0, 1)

    adapter = Adapter(
        policy=[f"{prefix}2", "hello", "world", 3.14]
    )
    result = adapter.execute(0, 1, 2, 3)
    assert result == (2, "hello", "world", 3.14)

    adapter = Adapter(
        policy=[1, 3, 5, 7, 11]
    )
    result = adapter.execute(0, 1, 2, 3)
    assert result == (1, 3, 5, 7, 11)

    adapter = Adapter(
        policy=[f"{prefix}{9-i}" for i in range(10)]
    )
    with pytest.raises(IndexError) as exc_info:
        result = adapter.execute(0, 1)
    assert str(exc_info.value) == (
        "The args received as input by 'Adapter' are not consistent with "
        "the given adapter policy because input args are too few! Input "
        "args are 2 but the policy foresees at least 10 items."
    )

    adapter = Adapter(
        policy=[]
    )
    result = adapter.execute(*tuple(range(100)))
    assert result == ()


@pytest.mark.integration
def test_adapter_integration_pipeline():
    """Test integration of Adapter component in the pipeline,
    connecting other components.
    """
    pipeline = Pipeline([
        FakeGetterExec(data_uri='http://...'),
        FakeSplitterExec(train_prop=.7),
        FakeTrainerExec(lr=1e-3, batch_size=32),
        Adapter(policy=[f"{Adapter.INPUT_PREFIX}-1"]),
        FakeSaverExec(save_path="my_model.pth")
    ])
    saved_model = pipeline.execute()
    assert saved_model == FakeTrainerExec.model
