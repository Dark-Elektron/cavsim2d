# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

project = 'cavsim2d'
copyright = '2024, Sosoho-Abasi Udongwo'
author = 'Sosoho-Abasi Udongwo'
release = '13.08.2024'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'numpydoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'matplotlib.sphinxext.mathmpl',
    'sphinx.ext.autosectionlabel',
    # 'sphinxcontrib.bibtex'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
bibtex_bibfiles = ['../references/refs.bib']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinxdoc'
# html_theme = 'bootstrap'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "../../qss/images/logo.png"
# html_static_path = sphinx_bootstrap_theme.get_html_theme_path()

# Enable numref
numfig = True
