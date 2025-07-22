# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import os
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Literal

py_logger = logging.getLogger(__name__)


class GPUBackend(ABC):
    @property
    @abstractmethod
    def man_lib(self) -> ModuleType | None:
        """The library used for GPU management."""
        pass

    @property
    @abstractmethod
    def man_type(self) -> Literal["nvidia", "amd"] | None:
        """The type of GPU management library used."""
        pass

    @abstractmethod
    def get_handle_by_uuid(self, gpu_uuid: str) -> object:
        """Get the device handle for a specific GPU UUID."""
        pass

    @abstractmethod
    def get_handle_by_id(self, gpu_id: int) -> object:
        """Get the device handle for a specific GPU index (ID)."""
        pass

    @abstractmethod
    def get_gpu_utilization(self, handle) -> float:
        """Get the GPU utilization (%) for a given handle."""
        pass

    @abstractmethod
    def get_gpu_power_usage(self, handle) -> float:
        """Get the GPU power usage (W) for a given handle."""
        pass

    @abstractmethod
    def get_visible_gpu_ids(self) -> list[int]:
        """Get a list of visible GPU UUIDs."""
        pass


class NvidiaBackend(GPUBackend):
    def __init__(self):
        try:
            import pynvml as nv

            nv.nvmlInit()  # will raise if no NVIDIA driver
            py_logger.info("Monitoring: NVIDIA backend set up")
            self._man_lib: ModuleType = nv
            self._man_type: Literal["nvidia", "amd"] = "nvidia"
        except Exception:
            raise RuntimeError(
                "Monitoring: NVIDIA backend could not be set up."
                " (pynvml could not be initialized)"
            )

    @property
    def man_lib(self) -> ModuleType | None:
        return self._man_lib

    @property
    def man_type(self) -> Literal["nvidia", "amd"] | None:
        return self._man_type

    def get_handle_by_uuid(self, gpu_uuid: str) -> object:
        return self._man_lib.nvmlDeviceGetHandleByUUID(str(gpu_uuid))

    def get_handle_by_id(self, gpu_id: int) -> object:
        return self._man_lib.nvmlDeviceGetHandleByIndex(gpu_id)

    def get_gpu_utilization(self, handle) -> float:
        """Get the GPU utilization (%) for a given handle."""
        return float(self._man_lib.nvmlDeviceGetUtilizationRates(handle).gpu)

    def get_gpu_power_usage(self, handle) -> float:
        """Get the GPU power usage (W) for a given handle."""
        return float(self._man_lib.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # mW -> W

    def get_visible_gpu_ids(self) -> list[int]:
        """Get a list of visible GPU UUIDs."""
        visible_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_gpus_str:
            visible_gpus = visible_gpus_str.split(",")
            visible_gpus = [int(id) for id in visible_gpus]
        else:
            visible_gpus = []

        return visible_gpus


class AMDBackend(GPUBackend):
    def __init__(self):
        try:
            import amdsmi

            amdsmi.amdsmi_init()  # will raise if no AMD driver
            py_logger.info("Monitoring: AMD backend set up")
            self._man_lib: ModuleType = amdsmi
            self._man_type: Literal["nvidia", "amd"] = "amd"
            self._devices = amdsmi.amdsmi_get_processor_handles()
        except Exception:
            raise RuntimeError(
                "Monitoring: AMD backend could not be set up."
                " (amdsmi could not be initialized)"
            )

    @property
    def man_lib(self) -> ModuleType | None:
        return self._man_lib

    @property
    def man_type(self) -> Literal["nvidia", "amd"] | None:
        return self._man_type

    def get_handle_by_uuid(self, gpu_uuid: str) -> object:
        for dev in self._devices:
            if self._man_lib.amdsmi_get_gpu_device_uuid(dev) == gpu_uuid:
                return dev
        raise ValueError(f"GPU with UUID {gpu_uuid} not accessible.")

    def get_handle_by_id(self, gpu_id: int) -> object:
        try:
            return self._devices[gpu_id]
        except IndexError:
            raise ValueError(f"GPU with ID {gpu_id} not accessible.")

    def get_gpu_utilization(self, handle) -> float:
        """Get the GPU utilization (%) for a given handle."""
        return float(self._man_lib.amdsmi_get_gpu_activity(handle)["gfx_activity"])

    def get_gpu_power_usage(self, handle) -> float:
        """Get the GPU power usage (W) for a given handle."""
        return float(self._man_lib.amdsmi_get_power_info(handle)["average_socket_power"])  # W

    def get_visible_gpu_ids(self) -> list[int]:
        """Get a list of visible GPU UUIDs."""
        if os.environ.get("HIP_VISIBLE_DEVICES") is not None:
            visible_gpus_str = os.environ.get("HIP_VISIBLE_DEVICES", "")

        else:
            visible_gpus_str = os.environ.get("ROCR_VISIBLE_DEVICES", "")

        if visible_gpus_str:
            visible_gpus = visible_gpus_str.split(",")
            visible_gpus = [int(id) for id in visible_gpus]
        else:
            visible_gpus = []

        return visible_gpus


def detect_gpu_backend() -> GPUBackend:
    """Detects the available GPU backend and returns an instance of the corresponding class."""
    try:
        return NvidiaBackend()
    except RuntimeError as e:
        py_logger.warning(f"NVIDIA backend not available: {e}")

    try:
        return AMDBackend()
    except RuntimeError as e:
        py_logger.warning(f"AMD backend not available: {e}")

    raise RuntimeError("No compatible GPU backend found. Please install pynvml or amdsmi.")
