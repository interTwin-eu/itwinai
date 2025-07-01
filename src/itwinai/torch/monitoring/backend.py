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
from typing import Tuple

py_logger = logging.getLogger(__name__)
Backend = Tuple[str, ModuleType]  # ("nvidia" | "amd"), module


def init_backend() -> Backend:
    try:
        import pynvml as nv

        nv.nvmlInit()  # will raise if no NVIDIA driver
        py_logger.info("Monitoring: NVIDIA backend set up")
        return "nvidia", nv
    except Exception:
        py_logger.info(
            "Monitoring: NVIDIA backend could not be set up. (pynvml could not be initialized)"
        )

    try:
        import amdsmi

        amdsmi.amdsmi_init()  # raises on non-AMD nodes
        py_logger.info("Monitoring: AMD backend set up")
        return "amd", amdsmi
    except Exception:
        py_logger.info(
            "Monitoring: AMD backend could not be set up. (amdsmi could not be initialized)"
        )
    raise RuntimeError(
        "Monitoring backend initialization failed: Neither NVIDIA (pynvml) nor AMD (amdsmi) "
        "GPU backends could be initialized. Ensure that the appropriate drivers and Python "
        "packages are installed, and that AMD or NVIDIA GPUs are available on this system."
    )
