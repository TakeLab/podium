# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Quick fix for cross-reference warnings:
from sphinx.domains.python import PythonDomain


sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'Podium'
copyright = '2020, TakeLab, FER, Zagreb'
author = 'TakeLab, FER, Zagreb'

# The full version, including alpha/beta/rc tags
release = '2020'

# -- General configuration ---------------------------------------------------

# The master toctree document.
master_doc = 'index'
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'recommonmark',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

source_suffix = ['.rst', '.md']
autodoc_typehints = 'none'
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

copybutton_prompt_text = ">>> "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}


# Nbsphinx options

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
# We never execute notebooks to avoid problems if nbsphinx won't find all dependencies.
nbsphinx_execute = 'never'

# If True, the build process is continued even if an exception occurs:
nbsphinx_allow_errors = True

# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
nbsphinx_timeout = 180

# Default Pygments lexer for syntax highlighting in code cells:
nbsphinx_codecell_lexer = 'ipython3'

# This won't work until repo is public
nbsphinx_prolog = """
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        Interactive online version:
        :raw-html:`<a href="https://colab.research.google.com/github/mttk/podium/blob/master/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`
"""

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if 'refspecific' in node:
            del node['refspecific']
        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)


def setup(app):
    #sphinx.add_domain(PatchedPythonDomain, override=True)
    app.add_css_file('css/podium.css')
    app.add_js_file('js/custom.js')
