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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../Packages'))
sys.path.insert(0, os.path.abspath('../../Packages/gme/core'))
sys.path.insert(0, os.path.abspath('../../Packages/gme/ode'))
sys.path.insert(0, os.path.abspath('../../Packages/gme/plot'))
import recommonmark
from recommonmark.transform import AutoStructify


# -- Project information -----------------------------------------------------

project = 'GME'
copyright = '2021, CPS'
author = 'CPS'

# The full version, including alpha/beta/rc tags
release = '1.0'

# The master toctree document.
master_doc = 'index'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx.ext.inheritance_diagram'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}
source_suffix = ['.rst', '.md']

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'python_docs_theme'
# html_theme = 'alabaster'

# html_theme = 'guillotina'
# def setup(app):
#     import sphinx_guillotina_theme
#     sphinx_guillotina_theme.setup(app)

html_theme = 'sphinx_py3doc_enhanced_theme'
import sphinx_py3doc_enhanced_theme
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]

html_theme_options = {
    'githuburl': 'https://github.com/cstarkjp/GME/',
    'bodyfont': '"Lucida Grande",Arial,sans-serif',
    'headfont': '"Lucida Grande",Arial,sans-serif',
    'codefont': 'monospace,sans-serif',
    'linkcolor': '#0072AA',
    'visitedlinkcolor': '#6363bb',
    'extrastyling': False,
}
pygments_style = 'friendly'


# html_theme_options = {
#     'logo' : '../_images/H_icon1.png',
#     'logo_name' : 'false',
#     'description' : 'Classical mechanics & differential geometry applied to understanding landscape erosion',
#     'description_font_style' : 'italic',
#     'show_related' : 'true',
#     'code_font_size' : '0.7em',
#     'page_width': '950px',
#     'sidebar_width': '195px',
#     'sidebar_text': 'blah'
# #     'page_width': '1050px'
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_sidebars = {
    '**': [
        # 'about.html',
        'searchbox.html',
        # 'relations.html',
        'globaltoc.html',
        'localtoc.html',
        # 'sourcelink.html',
    ]
}

# -- Intersphinx mappings -------------------------------------------------

intersphinx_mapping = {
    'gmplib': ('https://cstarkjp.github.io/GMPLib', '/Users/colinstark/Projects/GMPLib/objects.inv'),
    'sphinx': ('http://www.sphinx-doc.org/en/master', None),
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org', None),
    'mpl_toolkits': ('https://matplotlib.org', None),
    'np': ('https://docs.scipy.org/doc/numpy', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'PIL': ('https://pillow.readthedocs.io/en/latest/', None),
    'IPython': ('http://ipython.org/ipython-doc/stable', None)
}


# -- Napoleon settings -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True



# app setup hook
def setup(app):
    app.add_config_value('recommonmark_config', {
        #'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_math': False,
        'enable_inline_math': False,
        'enable_eval_rst': True,
        # 'enable_auto_doc_ref': True,
    }, True)
    app.add_transform(AutoStructify)
