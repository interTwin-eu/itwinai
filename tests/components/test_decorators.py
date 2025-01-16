# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from unittest.mock import MagicMock, call, patch

import pytest

from itwinai.components import BaseComponent, monitor_exec
from itwinai.distributed import suppress_workers_print


class SilentComponent(BaseComponent):
    """Fake component class to test the decorator meant to suppress print()."""

    def __init__(self, max_items: int, name: str = "SilentComponent") -> None:
        super().__init__(name)
        self.save_parameters(max_items=max_items, name=name)
        self.max_items = max_items

    @suppress_workers_print
    def execute(self, foo, bar=123):
        print("Executing SilentComponent")
        return "silent component result"


class FakeComponent(BaseComponent):
    """Fake component class to test the use of two decorators at the same time."""

    def __init__(self, max_items: int, name: str = "FakeComponent") -> None:
        super().__init__(name)
        self.save_parameters(max_items=max_items, name=name)
        self.max_items = max_items

    @suppress_workers_print
    @monitor_exec
    def execute(self, foo, bar):
        print("Executing FakeComponent")
        return "fake component result"


@pytest.fixture
def mock_component():
    """Fixture to provide a mock BaseComponent."""
    component = MagicMock(spec=BaseComponent)
    component.name = "TestComponent"
    return component


@pytest.fixture
def mock_fake_component():
    """Fixture to provide a FakeComponent instance."""
    component = FakeComponent(max_items=10)
    component.cleanup = MagicMock(spec=FakeComponent.cleanup)
    return component


def test_suppress_workers_print_decorator():
    """Test suppress_workers_print decorator behavior."""
    with (
        patch("builtins.print", autospec=True) as mock_print,
        patch("itwinai.distributed.detect_distributed_environment", autospec=True) as mock_env,
        patch(
            "itwinai.distributed.distributed_patch_print", autospec=True
        ) as mock_patch_print,
    ):
        # Mock environment: global rank different from 0 (non-main worker)
        mock_env.return_value.global_rank = 1
        mock_patch_print.return_value = lambda *args, **kwargs: None  # Suppress print

        @suppress_workers_print
        def dummy_function():
            print("This should be suppressed")

        dummy_function()
        mock_print.assert_not_called()  # Ensure print was suppressed

        # Check that outside the decorated function print can still be used
        print("This should be printed")
        mock_print.assert_called_once_with("This should be printed")
        mock_print.reset_mock()

        # Mock environment: global rank is 0 (main worker)
        mock_env.return_value.global_rank = 0
        mock_patch_print.return_value = print  # Use default print

        dummy_function()
        mock_print.assert_called_once_with("This should be suppressed")


def test_suppress_workers_print_component():
    """Test suppress_workers_print decorator behavioron SilentComponent's
    execute method."""
    with (
        patch("builtins.print", autospec=True) as mock_print,
        patch("itwinai.distributed.detect_distributed_environment", autospec=True) as mock_env,
        patch(
            "itwinai.distributed.distributed_patch_print", autospec=True
        ) as mock_patch_print,
    ):
        # Initialize the SilentComponent instance
        silent_component = SilentComponent(max_items=10)

        # Mock environment: global rank different from 0 (non-main worker)
        mock_env.return_value.global_rank = 1
        mock_patch_print.return_value = lambda *args, **kwargs: None  # Suppress print

        # Call the execute method on a non-main worker
        result = silent_component.execute("foo", bar=123)

        # Ensure the execute method's print statement was suppressed
        mock_print.assert_not_called()
        assert result == "silent component result"

        # Check that outside the decorated function print can still be used
        print("This should be printed")
        mock_print.assert_called_once_with("This should be printed")
        mock_print.reset_mock()

        # Mock environment: global rank is 0 (main worker)
        mock_env.return_value.global_rank = 0
        mock_patch_print.return_value = print  # Use default print

        # Call the execute method on the main worker
        result = silent_component.execute("foo", bar=123)

        # Ensure the print statement in the execute method was executed correctly
        mock_print.assert_called_once_with("Executing SilentComponent")
        assert result == "silent component result"


def test_monitor_exec_decorator(mock_component):
    """Test monitor_exec decorator behavior."""
    with patch("time.time") as mock_time:
        mock_time.side_effect = [100.0, 105.0]  # Simulate 5 seconds execution time

        @monitor_exec
        def dummy_method(self):
            return "Execution result"

        result = dummy_method(mock_component)

        # Assert the result of the decorated function
        assert result == "Execution result"

        # Check that the start and end messages were logged
        mock_component._printout.assert_any_call("Starting execution of 'TestComponent'...")
        mock_component._printout.assert_any_call("'TestComponent' executed in 5.000s")

        # Ensure the cleanup method was called
        mock_component.cleanup.assert_called_once()

        # Check that execution time was set correctly
        assert mock_component.exec_t == 5.0


def test_combined_decorators_on_fake_component(mock_fake_component):
    """Test the combination of suppress_workers_print and monitor_exec decorators
    on FakeComponent."""

    with (
        patch("builtins.print", autospec=True) as mock_print,
        patch("itwinai.distributed.detect_distributed_environment", autospec=True) as mock_env,
        patch(
            "itwinai.distributed.distributed_patch_print", autospec=True
        ) as mock_patch_print,
        patch("time.time", autospec=True) as mock_time,
    ):
        # Simulate time progression for execution timing
        mock_time.side_effect = [100.0, 105.0]  # Simulate a 5-second execution time

        # Case 1: Non-main worker (print suppressed)
        mock_env.return_value.global_rank = 1
        mock_patch_print.return_value = lambda *args, **kwargs: None  # Suppress print

        result = mock_fake_component.execute("foo", "bar")
        assert result == "fake component result"

        # Ensure the print statement was suppressed for non-main worker
        mock_print.assert_not_called()

        assert mock_fake_component.exec_t == 5.0
        mock_fake_component.cleanup.assert_called_once()

        # Reset mock calls for next test case
        mock_print.reset_mock()
        mock_fake_component.cleanup.reset_mock()
        mock_time.side_effect = [100.0, 105.0]  # Simulate a 5-second execution time

        # Case 2: Main worker (print allowed)
        mock_env.return_value.global_rank = 0
        mock_patch_print.return_value = print  # Use standard print

        result = mock_fake_component.execute("foo", "bar")
        assert result == "fake component result"

        # Ensure the print statement was not suppressed for the main worker
        mock_print.assert_has_calls(
            [
                call("############################################"),
                call("# Starting execution of 'FakeComponent'... #"),
                call("############################################"),
                call("Executing FakeComponent"),
                call("######################################"),
                call("# 'FakeComponent' executed in 5.000s #"),
                call("######################################"),
            ]
        )

        assert mock_fake_component.exec_t == 5.0
        mock_fake_component.cleanup.assert_called_once()
