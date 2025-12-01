from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.abspath(os.path.dirname(__file__))
setup(
    name='gru_quant',
    ext_modules=[
        CUDAExtension(
            name='gru_quant',
            sources=[
                'lib/GRUQuantWrapper.cpp',  # 仅 PyBind11 wrapper
            ],
            include_dirs=[os.path.join(here,'../include')],
            libraries=['gru_quant_shared'],           # 已经编译好的库名，不带 lib 前缀
            library_dirs=['lib'],   # 库所在目录
            extra_compile_args={
                'cxx': [
                    '-std=c++17',  # 关键：启用 C++17（解决 if constexpr 和 is_same_v 问题）
                    '-O3',         # 优化等级
                    '-fopenmp',    # 启用 OpenMP（修复 #pragma omp 警告）
                    '-Wno-unused-variable'  # 可选：忽略 unused variable 警告
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',  # NVCC 也需要指定 C++17（否则 CUDA 编译可能不兼容）
                    '-arch=sm_60'  # 根据你的 GPU 架构调整（例如 sm_75 对应 Turing 架构，sm_80 对应 Ampere）
                ]
            },
            extra_link_args=[
                '-fopenmp',  # 链接 OpenMP 库（与编译时对应）
                '-Wl,-rpath,$ORIGIN/lib'  # 可选：运行时优先从当前目录的 lib 文件夹找共享库
            ]
        ),
        CUDAExtension(
            name='gru_interface_binding',
            sources=[
                'lib/gru_interface_binding.cpp',  # GRU 接口 Python 绑定
            ],
            include_dirs=[os.path.join(here,'../include')],
            libraries=['gru_quant_shared'],           # 已经编译好的库名，不带 lib 前缀
            library_dirs=['lib'],   # 库所在目录
            extra_compile_args={
                'cxx': [
                    '-std=c++17',  # 关键：启用 C++17（解决 if constexpr 和 is_same_v 问题）
                    '-O3',         # 优化等级
                    '-fopenmp',    # 启用 OpenMP（修复 #pragma omp 警告）
                    '-Wno-unused-variable'  # 可选：忽略 unused variable 警告
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',  # NVCC 也需要指定 C++17（否则 CUDA 编译可能不兼容）
                    '-arch=sm_60'  # 根据你的 GPU 架构调整（例如 sm_75 对应 Turing 架构，sm_80 对应 Ampere）
                ]
            },
            extra_link_args=[
                '-fopenmp',  # 链接 OpenMP 库（与编译时对应）
                '-Wl,-rpath,$ORIGIN/lib'  # 可选：运行时优先从当前目录的 lib 文件夹找共享库
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
