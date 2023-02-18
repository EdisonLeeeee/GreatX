import datetime
import os.path as osp
import sys

import pyg_sphinx_theme

import greatx

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

author = 'Jintang Li'
project = 'GreatX'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = greatx.__version__
release = greatx.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = '../../imgs/greatx.png'
html_favicon = '../../imgs/favicon.png'

add_module_names = False
autodoc_member_order = 'bysource'
suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'greatx': greatx}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')
