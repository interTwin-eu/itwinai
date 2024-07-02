# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import subprocess

exclude_patterns = 'requirements.txt'

# sys.path.insert(0, os.path.abspath('../use-cases/'))
# sys.path.insert(0, os.path.abspath('../use-cases/3dgan/'))
# sys.path.insert(0, os.path.abspath('../use-cases/mnist/torch-lightning/'))
# sys.path.insert(0, os.path.abspath('../use-cases/mnist/torch/'))
sys.path.insert(0, os.path.abspath('../tutorials/ml-workflows/'))
# sys.path.insert(0, os.path.abspath('../tutorials/distributed-ml/'))
# sys.path.insert(0, os.path.abspath('../src/itwinai'))
# sys.path.insert(0, os.path.abspath('../src/itwinai/tensorflow'))
# sys.path.insert(0, os.path.abspath('../src/itwinai/torch'))
sys.path.insert(0, os.path.abspath('../src'))
# sys.path.insert(0, os.path.abspath('../...'))

project = 'itwinai'
copyright = ('2024, Matteo Bunino, Kalliopi Tsolaki, '
             'Rakesh Sarma, Mario Ruettgers on behalf of CERN & JSC')
author = 'Matteo Bunino, Kalliopi Tsolaki, Rakesh Sarma, Mario Ruettgers'
version = '0.0'  # short version
release = '0.0.2'  # full version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.viewcode', 'myst_parser', 'nbsphinx',
              'sphinx.ext.napoleon']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown'  # ,
    # '.ipynb': 'nbsphinx'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ["mlflow"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


def get_git_tag():
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0']
        ).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return 'unknown'


# Set the version to the latest tag
version = get_git_tag()
release = version

html_theme = 'sphinx_rtd_theme'  # 'alabaster'
html_static_path = ['_static']

html_context = {
    'display_version': True,
    'release': release
}

html_footer = """
<div class="custom-footer">
    <strong>Version:</strong> {{ release }}
</div>
"""

html_sidebars = {
    '**': [
        html_footer  # Adds the custom footer with version information
    ]
}
