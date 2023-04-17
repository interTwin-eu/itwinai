"""
Setup Python env by installing dependencies. 
"""

from typing import List
import sys
import subprocess


def install_pkg(pkg_name: str):
    """Install a pkg with pip."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg_name])
    except subprocess.CalledProcessError as err:
        print(f"Something went wrong when installing '{pkg_name}' pkg:")
        print(err)
        raise err


def install_deps_list(deps_list: List[str]):
    """Install a list of dependencies."""
    for dep in deps_list:
        install_pkg(dep)


def install_deps_file(deps_file_path: str):
    """Install a list of dependencies from file.
    Assume one package per line.
    """
    with open(deps_file_path, 'r', encoding='utf-8') as deps_file:
        deps_list = deps_file.read().splitlines()
        install_deps_list(deps_list)
