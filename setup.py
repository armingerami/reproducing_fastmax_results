# from setuptools import setup, Extension
# from torch.utils import cpp_extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# setup(name='fastmax_cpp',
#       ext_modules=[cpp_extension.CppExtension('fastmax_cpp', ['fastmax.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})


# setup(
#     name='fastmax_cpp',
#     ext_modules=[
#         CUDAExtension('fastmax_cpp', [
#             'fastmax.cpp',
#             'fastmax_cuda_kernel.cu',
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

setup(
    name='fastmax_cuda',
    ext_modules=[
        CUDAExtension('fastmax_cuda', [
            'fastmax_cuda.cpp',
            'fastmax_cuda_forward.cu',
            'fastmax_cuda_backward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })