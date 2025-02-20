import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'beam_gas_collisions'
copyright = '2024, Elias Waagaard'
author = 'Elias Waagaard'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme' 