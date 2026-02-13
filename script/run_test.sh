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

# 开发模式：使用 pip install -e（可编辑安装，修改代码后立即生效）
# ✅ 使用 --no-build-isolation（快速，使用当前环境的 torch）
# 安装后可在任何地方直接导入：from quant_gru import QuantGRU（无需路径操作）
pip install -e . --no-deps --no-build-isolation

# 其他安装方式（备选）：
# 方案1：build_ext --inplace（本地编译，不安装到 site-packages）
# python setup.py build_ext --inplace
# 注意：需要 sys.path.insert() 才能导入

# 方案2：普通安装（生产环境，修改代码后需要重新安装）
# pip install . --no-deps --no-build-isolation

echo "=== 运行测试 ==="
export LD_LIBRARY_PATH="$PROJECT_ROOT/pytorch/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_ROOT/pytorch"
python test_quant_gru.py
