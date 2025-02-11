# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Kalliopi Tsolaki
#
# Credit:
# - Kalliopi Tsolaki <kalliopi.tsolaki@cren.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Killian Verder <killian.verder@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../tutorials/ml-workflows/"))
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../images"))

project = "itwinai"
copyright = "2024, interTwin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_tabs.tabs",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",  # ,
    # '.ipynb': 'nbsphinx'
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "requirements.txt",
    "README.md",
    "testing-with-pytest.md",
    "working-with-containers.md",
]


autodoc_mock_imports = ["tensorflow", "keras"]
suppress_warnings = ["myst.xref_missing", "myst.header"]

# Enable numref
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


def get_git_tag():
    try:
        return (
            subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


# Set the version to the latest tag
version = get_git_tag()
release = version

html_theme = "sphinx_rtd_theme"  # 'alabaster'
html_static_path = ["_static"]

html_logo = "../docs/images/icon-itwinai-orange-white-subtitle.png"
html_theme_options = {"logo_only": True, "style_nav_header_background": "black"}

html_context = {"display_version": True, "release": release}

html_footer = """
<div class="custom-footer">
    <strong>Version:</strong> {{ release }}
</div>
"""

html_sidebars = {"**": [html_footer]}  # Adds the custom footer with version information
