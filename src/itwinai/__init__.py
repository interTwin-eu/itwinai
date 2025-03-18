# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import os

# Create a logger for your library
logger = logging.getLogger(__name__)

# Get log level from env variable (default to INFO)
log_level = os.getenv("ITWINAI_LOG_LEVEL", "INFO")

# Validate log level
valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if log_level.upper() not in valid_levels:
    logger.warning(
        f"ITWINAI_LOG_LEVEL was set to '{log_level}', which is not recognized. "
        f"Supported types are {', '.join(valid_levels)}."
    )
    log_level = "INFO"  # Default to INFO if invalid

# Apply log level
logger.setLevel(getattr(logging, log_level.upper()))

# Prevent duplicate handlers
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Prevent logs from bubbling up to the root logger
logger.propagate = False
