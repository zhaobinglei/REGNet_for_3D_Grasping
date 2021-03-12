from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {'cxx': ['-g'],
                      'nvcc': ['-O2']}

setup(
    name='pn2_ext',
    ext_modules=[
        CUDAExtension(
            name='pn2_ext',
            sources=[
                'csrc/main.cpp',
                'csrc/ball_query_kernel.cu',
                'csrc/grouping_kernel.cu',
                'csrc/sampling_kernel.cu',
                'csrc/interpolate_kernel.cu',
            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
