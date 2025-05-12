# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------
import logging
from types import ModuleType
from typing import Any, List, Tuple

py_logger = logging.getLogger(__name__)
Backend = Tuple[str, List[Any], ModuleType]  # ("nvidia" | "amd"), handles, module


def init_backend() -> Backend:
    try:
        import pynvml as nv

        nv.nvmlInit()  # will raise if no NVIDIA driver
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        py_logger.info("Monitoring: NVIDIA backend set up")
        return "nvidia", handles, nv
    except Exception:
        py_logger.warning("Monitoring: NVIDIA backend could not be set up")

    try:
        import amdsmi

        amdsmi.amdsmi_init()  # raises on non-AMD nodes
        handles = amdsmi.amdsmi_get_processor_handles()
        py_logger.info("Monitoring: AMD backend set up")
        return "amd", handles, amdsmi
    except Exception:
        py_logger.warning("Monitoring: AMD backend could not be set up")

    raise RuntimeError("Monitoring: No supported GPU backend found")
