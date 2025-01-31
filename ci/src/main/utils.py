# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


def get_codename(release_info: str) -> str:
    """
    Extracts the codename (VERSION_CODENAME or os_version) from release information.

    Args:
        release_info (str): The string containing the output of /etc/*-release.

    Returns:
        str: The extracted codename (e.g., "jammy" or "bookworm").
    """
    # Create a dictionary from the release info
    release_dict = {}
    for line in release_info.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release_dict[key.strip()] = value.strip().strip('"')

    # Attempt to extract the codename
    return release_dict.get("VERSION_CODENAME", release_dict.get("os_version", "Unknown"))
