# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import io
import logging

import requests

cli_logger = logging.getLogger("cli_logger")


def retrieve_remote_file(url: str) -> str:
    """Fetches remote file from url.

    Args:
        url: URL to the raw configuration file (YAML/JSON format), e.g. raw GitHub link.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    response_io_stream = io.StringIO(response.text)
    return response_io_stream.getvalue()


def remove_indentation_from_multiline_string(multiline_string: str) -> str:
    """Removes *all* indentation from the start of each line in a multi-line string.

    If you want to remove only the shared indentation of all lines, thus preserving
    indentation for nested structures, use the builtin `textwrap.dedent` function instead.

    The main purpose of this function is allowing you to define multi-line strings that
    only appear indented in the code, thus increasing readability.
    """
    return "\n".join([line.lstrip() for line in multiline_string.split("\n")])
