from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules=cythonize("trie.pyx"),
    zip_safe=False,
)