import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

cxx_flags = []
ext_libs = []

authors = [
        'Fahao Chen', 
]

if __name__ == '__main__':
    setuptools.setup(
        name='Collectives',
        version='1.0.0',
        description='Customized collective backend for dgnn.',
        author=', '.join(authors),
        author_email='chenfh612@gmail.com',
        ext_modules=[
            CUDAExtension(
                name='dgnn_collectives',
                sources=['Collective/collective_gloo.cpp'],
            )
        ],
        cmdclass={'build_ext': BuildExtension}
    )