from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='primal_cuda',
    ext_modules=[
        CUDAExtension('primal_cuda', [
            os.path.join(_current_dir, 'primal_cuda.cpp'),
            os.path.join(_current_dir, 'primal_cuda_kernel.cu'),
        ],
        extra_compile_args={
            'cxx': ['-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
            'nvcc': ['-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
