import os
import sys
from pathlib import Path
import re

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Dark Emulator2'
copyright = '2026, Satoshi Tanaka, Takahiro Nishimichi, Yosuke Kobayashi'
author = 'Satoshi Tanaka, Takahiro Nishimichi, Yosuke Kobayashi'

_root = Path(__file__).resolve().parents[1]
release = re.search(
    r'(?m)^version\s*=\s*"([^"]+)"\s*$',
    (_root / 'pyproject.toml').read_text(encoding='utf-8'),
).group(1)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

autodoc_preserve_defaults = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

htmlhelp_basename = 'DarkEmulator2Doc'
