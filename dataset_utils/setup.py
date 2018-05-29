from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension("text_gen_ru", ["text_gen_ru.pyx"],
    include_dirs = [numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language = 'c++')

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})