import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pulsar Segmentation using NeuralNets'
copyright = '2023, Tanumoy'
author = 'Tanumoy'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage', 'sphinx.ext.todo', 'sphinx_comments', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = []
# user starts in dark mode
default_dark_mode = True
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
autodoc_member_order = 'bysource'
