from distutils.core import setup
from Cython.Build import cythonize

setup(
   name='utils_cython',
   ext_modules = cythonize(['utils_cython.pyx']),
)
