from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='quant_gru',
    version='1.0.0',
    description='Quantized GRU implementation with C++ backend',
    py_modules=['quant_gru'],  # ✅ 将 quant_gru.py 作为 Python 模块安装
    ext_modules=[
        CUDAExtension(
            name='gru_interface_binding',
            sources=[
                'lib/gru_interface_binding.cc',  # GRU 接口 Python 绑定
            ],
            include_dirs=[os.path.join(here, '../include')],
            libraries=['gru_quant_shared'],      # 链接共享库
            library_dirs=[os.path.join(here, 'lib')],  # 使用绝对路径
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                    '-fopenmp',
                    '-Wno-unused-variable'
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_80'  # 根据 GPU 架构调整
                ]
            },
            extra_link_args=[
                '-fopenmp',
                # 设置 rpath 从扩展模块目录的 lib 子目录查找
                '-Wl,-rpath,$ORIGIN/lib',
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    # 安装后，quant_gru.py 和 gru_interface_binding.so 都会在同一个目录下
    # 这样 quant_gru.py 就能正确导入 gru_interface_binding 了
)
