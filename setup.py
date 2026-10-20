import os
from setuptools import Extension,setup
import pybind11

ext_modules = [
    Extension(
        name = "micro_llm_engine",
        sources=[
            'csrc/engine.cpp',
            'csrc/bindings.cpp'
        ],
        include_dirs=[
            'csrc',
            pybind11.get_include()
        ],
        extra_compile_args=[
            '/O2', #instruct compiler to perform highly optimized math
            '/std:c++17'
        ],
    ),
]

setup(
    name = "micro_llm_engine",
    version = '0.1',
    author = 'kohjiajun',
    description = 'High-performance C++ math kernels bound to Python via pybind11',
    ext_modules = ext_modules,
    zip_safe = False,
)