from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="matmul_cuda",
    ext_modules=[
        CUDAExtension("matmul_cuda", ["tlamacore/kernels/matmul_cuda.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension}
)
