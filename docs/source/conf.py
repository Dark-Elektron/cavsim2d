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
release = '1.0.0'

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
    'sphinxcontrib.bibtex',
    'myst_nb',
    'sphinx_immaterial'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
bibtex_bibfiles = ['../references/refs.bib']

# -- Notebooks (myst-nb) -----------------------------------------------------
# Example notebooks under source/examples/ are committed *with* their outputs
# and rendered as-is. They are not re-executed at build time: they need NGSolve
# and gmsh (which the docs runner does not have) and one of them solves 900
# eigenmodes, so a build would take ~10 minutes and could fail for reasons that
# have nothing to do with the docs.
#
# To re-execute a notebook after changing the library, run it yourself and
# commit the result:
#     jupyter nbconvert --to notebook --execute --inplace \
#         docs/source/examples/eigenmode_pillbox.ipynb
# (or set nb_execution_mode = "cache" to let Sphinx do it).
nb_execution_mode = 'off'
nb_render_markdown_format = 'myst'
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence']

# NGSolve's webgui (cav.show_mesh() / show_fields() with the default plotter)
# renders as a Jupyter *widget*, not an image. For it to appear in static HTML
# two things must both hold:
#   1. the page loads the ipywidgets JS (below), and
#   2. the notebook was saved WITH its widget state (nbclient's store_widget_state,
#      which writes a top-level `metadata.widgets` block).
# If a webgui cell renders blank, check (2) first — the state is what carries the
# scene. Passing plotter='matplotlib' sidesteps all of this and emits a plain PNG.
nb_ipywidgets_js = {
    'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js': {
        'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=',
        'crossorigin': 'anonymous',
    },
    'https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js': {
        'data-jupyter-widgets-cdn': 'https://cdn.jsdelivr.net/npm/',
        'crossorigin': 'anonymous',
    },
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_immaterial'

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/pencil",
    },
    "site_url": "https://github.com/Dark-Elektron/cavsim2d",
    "repo_url": "https://github.com/Dark-Elektron/cavsim2d",
    "repo_name": "cavsim2d",
    "globaltoc_collapse": False,
    "features": [
        "navigation.sections",
        "navigation.top",
        "search.share",
        "toc.follow",
        "content.code.copy",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            }
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "blue",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            }
        }
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "../images/cavsim2d_logo.svg"

# Enable numref
numfig = True

# Prefix document name to section labels to avoid duplicate warnings
autosectionlabel_prefix_document = True

