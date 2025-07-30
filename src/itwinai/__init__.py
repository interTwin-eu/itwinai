# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import os
import sys

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
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Prevent logs from bubbling up to the root logger
logger.propagate = False

class CLIFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"[ERROR]: {msg}"
        if record.levelno == logging.WARNING:
            return f"[WARNING]: {msg}"

        # INFO / DEBUG stay “plain”
        return msg

plain_logger = logging.getLogger("cli_logger")
plain_logger.setLevel(logging.INFO)
if not plain_logger.hasHandlers():
    # Moving the output to stdout instead of stderr
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(CLIFormatter())
    plain_logger.addHandler(h)
plain_logger.propagate = False

# Ray
if os.environ.get("TUNE_DISABLE_STRICT_METRIC_CHECKING", None) is None:
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# Set Ray V2 train to 1 as soon as Ray V2 train works:
# Our issue: https://github.com/ray-project/ray/issues/53921
if os.environ.get("RAY_TRAIN_V2_ENABLED", None) is None:
    os.environ["RAY_TRAIN_V2_ENABLED"] = "0"

