from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil

here = os.path.abspath(os.path.dirname(__file__))
SHARED_LIB_NAME = 'libgru_quant_shared.so'
SOURCE_LIB = os.path.join(here, 'lib', SHARED_LIB_NAME)


def copy_shared_library(install_dir):
    """
    复制共享库到安装目录的 lib 子目录
    
    工作原理：
    - 扩展模块通过 rpath ($ORIGIN/lib) 在运行时查找共享库
    - $ORIGIN 指向扩展模块 .so 文件所在目录
    - 因此需要将共享库复制到安装目录的 lib/ 子目录中
    
    Args:
        install_dir: 安装目录路径（通常是 site-packages 或 dist-packages）
        
    Returns:
        bool: 是否成功复制
    """
    lib_dir = os.path.join(install_dir, 'lib')
    
    try:
        os.makedirs(lib_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating lib directory {lib_dir}: {e}")
        return False
    
    dest_lib = os.path.join(lib_dir, SHARED_LIB_NAME)
    
    if not os.path.exists(SOURCE_LIB):
        print(f"Warning: {SOURCE_LIB} not found, skipping copy")
        return False
    
    try:
        shutil.copy2(SOURCE_LIB, dest_lib)
        print(f"✓ Copied {SHARED_LIB_NAME} to {dest_lib}")
        return True
    except (OSError, shutil.Error) as e:
        print(f"Error copying {SOURCE_LIB} to {dest_lib}: {e}")
        return False


class InstallLibWithLib(install_lib):
    """自定义 install_lib 类，在安装库文件后复制共享库"""
    def run(self):
        install_lib.run(self)
        copy_shared_library(self.install_dir)


class InstallWithLib(install):
    """自定义安装类，在安装后复制共享库文件"""
    def run(self):
        install.run(self)
        copy_shared_library(self.install_lib)


class DevelopWithLib(develop):
    """自定义开发模式安装类，在安装后复制共享库文件"""
    def run(self):
        develop.run(self)
        # 开发模式下，install_dir 或 install_lib 可能不同
        install_dir = getattr(self, 'install_dir', None) or getattr(self, 'install_lib', None)
        if install_dir:
            copy_shared_library(install_dir)


setup(
    name='quant_gru',
    version='1.0.1',
    description='Quantized GRU implementation with C++ backend',
    py_modules=['quant_gru'],
    ext_modules=[
        CUDAExtension(
            name='gru_interface_binding',
            sources=['lib/gru_interface_binding.cc'],
            include_dirs=[os.path.join(here, '../include')],
            libraries=['gru_quant_shared'],
            library_dirs=[os.path.join(here, 'lib')],
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
                # rpath: 运行时从扩展模块目录的 lib 子目录查找共享库
                # $ORIGIN 是扩展模块 .so 文件所在目录
                '-Wl,-rpath,$ORIGIN/lib',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'install': InstallWithLib,
        'install_lib': InstallLibWithLib,
        'develop': DevelopWithLib,
    },
)
