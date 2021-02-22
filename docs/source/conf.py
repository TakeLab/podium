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
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    "sphinx.ext.intersphinx",
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'recommonmark',
    'sphinx_copybutton',
]

source_suffix = ['.rst', '.md']
autodoc_typehints = 'none'
autoclass_content = 'both'

# Mock since the install is large, and it clogs the available space on GH Actions
autodoc_mock_imports = ['torch']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

copybutton_prompt_text = r'>>> |\.\.\. '

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']     # right now we don't use that

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

# only run doctests marked with a ".. doctest::" directive
doctest_test_doctest_blocks = ''

doctest_global_setup = '''
try:
    import transformers
except ImportError:
    transformers = None

try:
    import spacy
    spacy.load('en', disable=['parser', 'ner'])
except (ImportError, IOError):
    spacy = None

'''
