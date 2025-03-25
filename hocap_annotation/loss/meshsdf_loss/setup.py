from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CURR_DIR = Path(__file__).parent.resolve()

# Compiler flags
c_flags = ["-O3", "-std=c++17"]

nvcc_flags = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-O3,-std=c++17",
]

setup(
    name="meshsdf_loss",
    version="0.0.1",
    author="Jikai Wang",
    author_email="jikai.wang@utdallas.edu",
    ext_modules=[
        CUDAExtension(
            name="meshsdf_loss_cuda",
            sources=[
                str(CURR_DIR / "meshsdf_loss_cuda.cpp"),
                str(CURR_DIR / "meshsdf_loss_cuda_kernel.cu"),
                str(CURR_DIR / "rbd" / "bvh.cu"),
                str(CURR_DIR / "rbd" / "util.cpp"),
            ],
            extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
            include_dirs=[str(CURR_DIR / "rbd")],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
