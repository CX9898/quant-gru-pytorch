#!/bin/bash
set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== 清理并重新编译 C++ 库 ==="
rm -rf "$PROJECT_ROOT/build"
mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"
cmake ..
make -j$(nproc)

echo "=== 重新编译 Python 绑定 ==="
cd "$PROJECT_ROOT/pytorch"
rm -rf build/  # 删除 Python 扩展的 build 缓存（包含 ninja 缓存）
rm -f *.so  # 强制删除旧的绑定
python setup.py build_ext --inplace

echo "=== 运行测试 ==="
export LD_LIBRARY_PATH="$PROJECT_ROOT/pytorch/lib:$LD_LIBRARY_PATH"
python test_quant_gru.py
