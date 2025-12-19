#!/bin/bash

# ============================================================================
# GRU 量化库发布打包脚本
# 
# 用法: ./pack_release.sh [版本号]
# 示例: ./pack_release.sh 1.0.0
# ============================================================================

set -e

# 版本号（默认为日期）
VERSION=${1:-$(date +%Y%m%d)}
RELEASE_NAME="quant-gru-pytorch-v${VERSION}"
RELEASE_DIR="release/${RELEASE_NAME}"

# 项目根目录（脚本在 script 子目录中）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  GRU 量化库打包脚本"
echo "  版本: ${VERSION}"
echo "=============================================="

# 清理旧的发布目录
if [ -d "release" ]; then
    rm -rf "release"
fi

# 创建发布目录结构
echo ""
echo "[1/6] 创建目录结构..."
mkdir -p "${RELEASE_DIR}/pytorch/lib"
mkdir -p "${RELEASE_DIR}/pytorch/config"
mkdir -p "${RELEASE_DIR}/pytorch/example"

# 复制核心 Python 文件
echo "[2/6] 复制 Python 接口..."
cp pytorch/custom_gru.py "${RELEASE_DIR}/pytorch/"

# 复制编译好的库文件
echo "[3/6] 复制编译库文件..."
if [ -f "pytorch/gru_interface_binding.cpython-310-x86_64-linux-gnu.so" ]; then
    cp pytorch/gru_interface_binding.cpython-310-x86_64-linux-gnu.so "${RELEASE_DIR}/pytorch/"
else
    echo "  ⚠️  警告: gru_interface_binding.*.so 不存在，请先编译"
fi

if [ -f "pytorch/lib/libgru_quant_shared.so" ]; then
    cp pytorch/lib/libgru_quant_shared.so "${RELEASE_DIR}/pytorch/lib/"
else
    echo "  ⚠️  警告: libgru_quant_shared.so 不存在，请先编译"
fi

# 复制配置文件
echo "[4/6] 复制配置文件..."
cp pytorch/config/gru_quant_bitwidth_config.json "${RELEASE_DIR}/pytorch/config/"
cp pytorch/config/README.md "${RELEASE_DIR}/pytorch/config/"

# 复制示例文件
echo "[5/6] 复制示例文件..."
if [ -f "pytorch/example/example_usage.py" ]; then
    cp pytorch/example/example_usage.py "${RELEASE_DIR}/pytorch/example/"
fi

# 创建简单的使用说明
cat > "${RELEASE_DIR}/INSTALL.md" << 'EOF'
# 安装与使用说明

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.0+

## 安装步骤

1. 将 `pytorch` 目录添加到 Python 路径：

```python
import sys
sys.path.append("/path/to/quant-gru-pytorch/pytorch")
```

2. 设置动态库路径（Linux）：

```bash
export LD_LIBRARY_PATH=/path/to/quant-gru-pytorch/pytorch/lib:$LD_LIBRARY_PATH
```

或在 Python 中：

```python
import os
os.environ['LD_LIBRARY_PATH'] = '/path/to/quant-gru-pytorch/pytorch/lib'
```

## 快速开始

### 推理

```python
from custom_gru import CustomGRU

# 创建 GRU 并加载配置
gru = CustomGRU(input_size=64, hidden_size=128)
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json")

# 校准（使用真实数据效果更好）
for batch in calibration_loader:
    gru.calibrate(batch)

# 推理
output, h_n = gru(input_data)
```

### 训练（量化感知训练）

```python
import torch

gru = CustomGRU(input_size=64, hidden_size=128)
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json")

# 校准
for batch in calibration_loader:
    gru.calibrate(batch)

# 训练循环（前向使用量化，反向使用浮点）
optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
gru.train()

for epoch in range(num_epochs):
    for x, target in train_loader:
        optimizer.zero_grad()
        output, _ = gru(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 更多示例

请参阅 `pytorch/example/example_usage.py`

## 详细配置说明

请参阅 `pytorch/config/README.md`
EOF

# 打包
echo "[6/6] 创建压缩包..."
cd release
tar -czvf "${RELEASE_NAME}.tar.gz" "${RELEASE_NAME}"
cd ..

# 打印结果
echo ""
echo "=============================================="
echo "  打包完成！"
echo "=============================================="
echo ""
echo "发布目录: ${RELEASE_DIR}/"
echo "压缩包:   release/${RELEASE_NAME}.tar.gz"
echo ""
echo "目录结构:"
tree "${RELEASE_DIR}" 2>/dev/null || find "${RELEASE_DIR}" -type f | sed "s|${RELEASE_DIR}|  ${RELEASE_NAME}|"
echo ""
echo "文件大小:"
du -sh "release/${RELEASE_NAME}.tar.gz"

