from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib
import os
import shutil
import sys

try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
except ImportError as exc:
    raise RuntimeError(
        'PyTorch is required before building quant_gru. '
        'Please install a CUDA-enabled PyTorch package first.'
    ) from exc

here = os.path.abspath(os.path.dirname(__file__))
SHARED_LIB_NAME = 'libgru_quant_shared.so'
SOURCE_LIB = os.path.join(here, 'lib', SHARED_LIB_NAME)


def normalize_arch_list(arch_list):
    """标准化 TORCH_CUDA_ARCH_LIST 格式，统一为分号分隔。"""
    parts = []
    for arch in arch_list.replace(',', ';').replace(' ', ';').split(';'):
        arch = arch.strip()
        if arch:
            parts.append(arch)
    return ';'.join(parts)


def ensure_supported_platform():
    """当前 setup.py 仅支持 Linux 构建。"""
    if not sys.platform.startswith('linux'):
        raise RuntimeError(
            'quant_gru currently supports Linux builds only. '
        )


def ensure_cuda_build_environment():
    """校验当前是否满足 CUDA 构建前提。"""
    if torch.version.cuda is None:
        raise RuntimeError(
            'Detected CPU-only PyTorch. This project requires a CUDA-enabled PyTorch build.'
        )

    if CUDA_HOME is None and shutil.which('nvcc') is None:
        raise RuntimeError(
            'CUDA toolkit not found. Please install CUDA and ensure CUDA_HOME or nvcc is available.'
        )


def ensure_prebuilt_shared_library():
    """校验 CMake 预编译共享库是否已生成。"""
    if not os.path.exists(SOURCE_LIB):
        raise RuntimeError(
            f'Prebuilt shared library not found: {SOURCE_LIB}. '
            'Please build the C++ library first, for example: '
            '`mkdir -p build && cd build && cmake .. && make -j$(nproc)`.'
        )


def resolve_cuda_arch_list():
    """
    解析 CUDA 目标架构列表。

    优先使用用户显式设置的 TORCH_CUDA_ARCH_LIST；如果未设置，
    则遍历所有可见 GPU 自动生成唯一架构列表。
    当前项目要求必须在 CUDA 环境下构建，因此缺少 CUDA 相关条件时直接报错。
    """
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if arch_list:
        arch_list = normalize_arch_list(arch_list)
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
        return arch_list

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError(
            'No visible CUDA GPU detected and TORCH_CUDA_ARCH_LIST is not set. '
            'For GPU-less build environments, set TORCH_CUDA_ARCH_LIST explicitly '
            '(for example: "8.0;8.6").'
        )

    archs = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        archs.add(f'{major}.{minor}')

    arch_list = ';'.join(sorted(archs, key=lambda x: tuple(map(int, x.split('.')))))
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    return arch_list


ensure_supported_platform()
ensure_cuda_build_environment()
ensure_prebuilt_shared_library()
RESOLVED_CUDA_ARCH_LIST = resolve_cuda_arch_list()
print(f'Using TORCH_CUDA_ARCH_LIST={RESOLVED_CUDA_ARCH_LIST}')


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
        raise RuntimeError(f'Error creating lib directory {lib_dir}: {e}') from e
    
    dest_lib = os.path.join(lib_dir, SHARED_LIB_NAME)
    
    if not os.path.exists(SOURCE_LIB):
        raise RuntimeError(
            f'Prebuilt shared library not found during install: {SOURCE_LIB}. '
            'Please build the C++ library before installing the Python package.'
        )
    
    try:
        shutil.copy2(SOURCE_LIB, dest_lib)
        print(f"✓ Copied {SHARED_LIB_NAME} to {dest_lib}")
        return True
    except (OSError, shutil.Error) as e:
        raise RuntimeError(f'Error copying {SOURCE_LIB} to {dest_lib}: {e}') from e


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
    version='1.0.4',
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
