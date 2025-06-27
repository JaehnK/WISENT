from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
from setuptools import Extension

extensions = [
    Extension("co_occurence", ["co_occurence.pyx"]),
    Extension("trie", ["trie.pyx"]),
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
)