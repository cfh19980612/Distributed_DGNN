import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["dgnn_group.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

r'''
Create the python extension by using cpp extension.
Importing dgnn_collectives makes torch.distributed recognize `GROUP_DGNN`
as a valid backend.
'''
if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name = "dgnn_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name = "dgnn_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )

setup(
    name = "Dgnn-Collectives",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)