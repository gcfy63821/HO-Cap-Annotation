from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CURR_DIR = Path(__file__).parent.resolve()

# Compiler flags
# c_flags = ["-O3", "-std=c++17"]

# nvcc_flags = [
#     "-U__CUDA_NO_HALF_OPERATORS__",
#     "-U__CUDA_NO_HALF_CONVERSIONS__",
#     "-U__CUDA_NO_HALF2_OPERATORS__",
#     "-Xcompiler=-O3,-std=c++17",
# ]

# Compiler flags
c_flags = ["-O3", "-std=c++17"]

# Support multiple GPU architectures
# RTX 20 series: sm_75, RTX 30 series: sm_86, RTX 40 series: sm_89
nvcc_flags = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-Xcompiler", "-O3",
    "-Xcompiler", "-std=c++17",
    # Support multiple GPU architectures
    "-gencode=arch=compute_75,code=sm_75",  # RTX 20 series
    "-gencode=arch=compute_80,code=sm_80",  # A100
    "-gencode=arch=compute_86,code=sm_86",  # RTX 30 series
    "-gencode=arch=compute_89,code=sm_89",  # RTX 40 series (RTX 4090)
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
